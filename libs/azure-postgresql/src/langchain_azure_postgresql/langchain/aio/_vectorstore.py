"""Asynchronous VectorStore integration for Azure Database for PostgreSQL using LangChain."""

import logging
import re
import sys
import uuid
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from itertools import cycle
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, utils
from pgvector.psycopg import register_vector_async  # type: ignore[import-untyped]
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from ...common import (
    HNSW,
    Algorithm,
    DiskANN,
    IVFFlat,
    VectorOpClass,
    VectorType,
)
from ...common._shared import _run_coroutine_in_sync
from .._shared import Filter, _filter_to_sql

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

_logger = logging.getLogger(__name__)


class AsyncAzurePGVectorStore(BaseModel, VectorStore):
    """LangChain VectorStore backed by Azure Database for PostgreSQL (async).

    The store validates or creates the backing table on initialization, and
    optionally discovers an existing vector index configuration. It supports
    inserting, deleting, fetching by id, similarity search, and MMR search.

    Fields such as ``schema_name``, ``table_name``, and column names control the
    schema layout. ``embedding_type``, ``embedding_dimension``, and
    ``embedding_index`` describe the vector column and its index behavior.

    Metadata can be stored in a single JSONB column by passing a string (default
    ``"metadata"``), in multiple typed columns via a list of strings/tuples, or
    disabled by setting ``metadata_columns=None``.

    :param embedding: The embedding model to use for embedding vector generation.
    :type embedding: Embeddings | None
    :param connection: The database connection or connection pool to use.
    :type connection: AsyncConnection | AsyncConnectionPool
    :param schema_name: The name of the database schema to use.
    :type schema_name: str
    :param table_name: The name of the database table to use.
    :type table_name: str
    :param id_column: The name of the column containing document IDs (UUIDs).
    :type id_column: str
    :param content_column: The name of the column containing document content.
    :type content_column: str
    :param embedding_column: The name of the column containing document embeddings.
    :type embedding_column: str
    :param embedding_type: The type of the embedding vectors.
    :type embedding_type: VectorType | None
    :param embedding_dimension: The dimensionality of the embedding vectors.
    :type embedding_dimension: PositiveInt | None
    :param embedding_index: The algorithm used for indexing the embedding vectors.
    :type embedding_index: Algorithm | None
    :param _embedding_index_name: (internal) The name of the discovered or created index.
    :type _embedding_index_name: str | None
    :param metadata_columns: The columns to use for storing metadata.
    :type metadata_columns: list[str] | list[tuple[str, str]] | str | None
    """

    embedding: Embeddings | None = None
    connection: AsyncConnection | AsyncConnectionPool
    schema_name: str = "public"
    table_name: str = "langchain"
    id_column: str = "id"
    content_column: str = "content"
    embedding_column: str = "embedding"
    embedding_type: VectorType | None = None
    embedding_dimension: PositiveInt | None = None
    embedding_index: Algorithm | None = None
    _embedding_index_name: str | None = None
    metadata_columns: list[str] | list[tuple[str, str]] | str | None = "metadata"

    model_config = ConfigDict(
        # Allow arbitrary types like Embeddings and AsyncConnection(Pool)
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def verify_and_init_store(self) -> Self:
        # verify that metadata_columns is not empty if provided
        if self.metadata_columns is not None and len(self.metadata_columns) == 0:
            raise ValueError("'metadata_columns' cannot be empty if provided.")

        _logger.debug(
            "checking if table '%s.%s' exists with the required columns",
            self.schema_name,
            self.table_name,
        )

        coroutine = self._ensure_table_verified()
        _run_coroutine_in_sync(coroutine)

        return self

    async def _ensure_table_verified(self) -> None:
        async with (
            self._connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                sql.SQL(
                    """
                      select  a.attname as column_name,
                              format_type(a.atttypid, a.atttypmod) as column_type
                        from  pg_attribute a
                              join pg_class c on a.attrelid = c.oid
                              join pg_namespace n on c.relnamespace = n.oid
                       where  a.attnum > 0
                              and not a.attisdropped
                              and n.nspname = %(schema_name)s
                              and c.relname = %(table_name)s
                    order by  a.attnum asc
                    """
                ),
                {"schema_name": self.schema_name, "table_name": self.table_name},
            )
            resultset = await cursor.fetchall()
            existing_columns: dict[str, str] = {
                row["column_name"]: row["column_type"] for row in resultset
            }

        # if table exists, verify that required columns exist and have correct types
        if len(existing_columns) > 0:
            _logger.debug(
                "table '%s.%s' exists with the following column mapping: %s",
                self.schema_name,
                self.table_name,
                existing_columns,
            )

            id_column_type = existing_columns.get(self.id_column)
            if id_column_type != "uuid":
                raise ValueError(
                    f"Table '{self.schema_name}.{self.table_name}' must have a column '{self.id_column}' of type 'uuid'."
                )

            content_column_type = existing_columns.get(self.content_column)
            if content_column_type is None or (
                content_column_type != "text"
                and not content_column_type.startswith("varchar")
            ):
                raise ValueError(
                    f"Table '{self.schema_name}.{self.table_name}' must have a column '{self.content_column}' of type 'text' or 'varchar'."
                )

            embedding_column_type = existing_columns.get(self.embedding_column)
            pattern = re.compile(r"(?P<type>\w+)(?:\((?P<dim>\d+)\))?")
            m = pattern.match(embedding_column_type if embedding_column_type else "")
            parsed_type: str | None = m.group("type") if m else None
            parsed_dim: PositiveInt | None = (
                PositiveInt(m.group("dim")) if m and m.group("dim") else None
            )

            vector_types = [t.value for t in VectorType.__members__.values()]
            if parsed_type not in vector_types:
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' must be one of the following types: {vector_types}."
                )
            elif (
                self.embedding_type is not None
                and parsed_type != self.embedding_type.value
            ):
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' has type '{parsed_type}', but the specified embedding_type is '{self.embedding_type.value}'. They must match."
                )
            elif self.embedding_type is None:
                _logger.info(
                    "embedding_type is not specified, but the column '%s' in table '%s.%s' has type '%s'. Overriding embedding_type accordingly.",
                    self.embedding_column,
                    self.schema_name,
                    self.table_name,
                    parsed_type,
                )
                self.embedding_type = VectorType(parsed_type)

            if parsed_dim is not None and self.embedding_dimension is None:
                _logger.info(
                    "embedding_dimension is not specified, but the column '%s' in table '%s.%s' has a dimension of %d. Overriding embedding_dimension accordingly.",
                    self.embedding_column,
                    self.schema_name,
                    self.table_name,
                    parsed_dim,
                )
                self.embedding_dimension = parsed_dim
            elif (
                parsed_dim is not None
                and self.embedding_dimension is not None
                and parsed_dim != self.embedding_dimension
            ):
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' has a dimension of {parsed_dim}, but the specified embedding_dimension is {self.embedding_dimension}. They must match."
                )

            if self.metadata_columns is not None:
                metadata_columns: list[tuple[str, str | None]]
                if isinstance(self.metadata_columns, str):
                    metadata_columns = [(self.metadata_columns, "jsonb")]
                else:
                    metadata_columns = [
                        (col[0], col[1]) if isinstance(col, tuple) else (col, None)
                        for col in self.metadata_columns
                    ]

                for col, col_type in metadata_columns:
                    existing_type = existing_columns.get(col)
                    if existing_type is None:
                        raise ValueError(
                            f"Column '{col}' does not exist in table '{self.schema_name}.{self.table_name}'."
                        )
                    elif col_type is not None and existing_type != col_type:
                        raise ValueError(
                            f"Column '{col}' in table '{self.schema_name}.{self.table_name}' must be of type '{col_type}', but found '{existing_type}'."
                        )

            async with (
                self._connection() as conn,
                conn.cursor(row_factory=dict_row) as cursor,
            ):
                _logger.debug(
                    "checking if table '%s.%s' has a vector index on column '%s'",
                    self.schema_name,
                    self.table_name,
                    self.embedding_column,
                )
                await cursor.execute(
                    sql.SQL(
                        """
                        with cte as (
                          select  n.nspname as schema_name,
                                  ct.relname as table_name,
                                  ci.relname as index_name,
                                  a.amname as index_type,
                                  pg_get_indexdef(
                                    ci.oid, -- index OID
                                    generate_series(1, array_length(ii.indkey, 1)), -- column no
                                    true -- pretty print
                                  ) as index_column,
                                  o.opcname as index_opclass,
                                  coalesce(ci.reloptions, array[]::text[]) as index_opts
                            from  pg_class ci
                                  join pg_index ii on ii.indexrelid = ci.oid
                                  join pg_am a on a.oid = ci.relam
                                  join pg_class ct on ct.oid = ii.indrelid
                                  join pg_namespace n on n.oid = ci.relnamespace
                                  join pg_opclass o on o.oid = any(ii.indclass)
                           where  ci.relkind = 'i'
                                  and ct.relkind = 'r'
                                  and ii.indisvalid
                                  and ii.indisready
                        ) select  schema_name, table_name, index_name, index_type,
                                  index_column, index_opclass, index_opts
                            from  cte
                           where  schema_name = %(schema_name)s
                                  and table_name = %(table_name)s
                                  and index_column like %(embedding_column)s
                                  and (
                                      index_opclass like '%%vector%%'
                                      or index_opclass like '%%halfvec%%'
                                      or index_opclass like '%%sparsevec%%'
                                      or index_opclass like '%%bit%%'
                                  )
                        order by  schema_name, table_name, index_name
                        """
                    ),
                    {
                        "schema_name": self.schema_name,
                        "table_name": self.table_name,
                        "embedding_column": f"%{self.embedding_column}%",
                    },
                )
                resultset = await cursor.fetchall()

            if len(resultset) > 0:
                _logger.debug(
                    "table '%s.%s' has %d vector index(es): %s",
                    self.schema_name,
                    self.table_name,
                    len(resultset),
                    resultset,
                )

                if self.embedding_index is None:
                    _logger.info(
                        "embedding_index is not specified, using the first found index: %s",
                        resultset[0],
                    )

                    index_name = resultset[0]["index_name"]
                    index_type = resultset[0]["index_type"]
                    index_opclass = VectorOpClass(resultset[0]["index_opclass"])
                    index_opts = {
                        opts.split("=")[0]: opts.split("=")[1]
                        for opts in resultset[0]["index_opts"]
                    }

                    index = (
                        DiskANN(op_class=index_opclass, **index_opts)
                        if index_type == "diskann"
                        else HNSW(op_class=index_opclass, **index_opts)
                        if index_type == "hnsw"
                        else IVFFlat(op_class=index_opclass, **index_opts)
                    )

                    self.embedding_index = index
                    self._embedding_index_name = index_name
                else:
                    _logger.info(
                        "embedding_index is specified as '%s'; will try to find a matching index.",
                        self.embedding_index,
                    )

                    index_opclass = self.embedding_index.op_class.value  # type: ignore[assignment]
                    if isinstance(self.embedding_index, DiskANN):
                        index_type = "diskann"
                    elif isinstance(self.embedding_index, HNSW):
                        index_type = "hnsw"
                    else:
                        index_type = "ivfflat"

                    found_matching_index = False
                    for row in resultset:
                        if (
                            row["index_type"] == index_type
                            and row["index_opclass"] == index_opclass
                        ):
                            index_opts = {
                                opts.split("=")[0]: opts.split("=")[1]
                                for opts in row["index_opts"]
                            }
                            index = (
                                DiskANN(op_class=index_opclass, **index_opts)
                                if index_type == "diskann"
                                else HNSW(op_class=index_opclass, **index_opts)
                                if index_type == "hnsw"
                                else IVFFlat(op_class=index_opclass, **index_opts)
                            )
                            if (
                                index.build_settings()
                                == self.embedding_index.build_settings()
                            ):
                                _logger.info("found a matching index: %s", row)
                                found_matching_index = True
                                self._embedding_index_name = row["index_name"]
                                break
                    if not found_matching_index:
                        raise ValueError(
                            f"Could not find a matching index for the specified embedding_index '{self.embedding_index}' in table '{self.schema_name}.{self.table_name}'. Found indexes: {resultset}"
                        )

            elif self.embedding_index is None:
                _logger.info(
                    "embedding_index is not specified, and no vector index found in table '%s.%s'. defaulting to 'DiskANN' with 'vector_cosine_ops' opclass.",
                    self.schema_name,
                    self.table_name,
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)
                self._embedding_index_name = None

        # if table does not exist, create it
        else:
            _logger.debug(
                "table '%s.%s' does not exist, creating it with the required columns",
                self.schema_name,
                self.table_name,
            )

            metadata_columns: list[tuple[str, str]] = []  # type: ignore[no-redef]
            if self.metadata_columns is None:
                _logger.warning(
                    "Metadata columns are not specified, defaulting to 'metadata' of type 'jsonb'."
                )
                metadata_columns = [("metadata", "jsonb")]
            elif isinstance(self.metadata_columns, str):
                _logger.warning(
                    "Metadata columns are specified as a string, defaulting to 'jsonb' type."
                )
                metadata_columns = [(self.metadata_columns, "jsonb")]
            elif isinstance(self.metadata_columns, list):
                _logger.warning(
                    "Metadata columns are specified as a list; defaulting to 'text' when type is not defined."
                )
                metadata_columns = [
                    (col[0], col[1]) if isinstance(col, tuple) else (col, "text")
                    for col in self.metadata_columns
                ]

            if self.embedding_type is None:
                _logger.warning(
                    "Embedding type is not specified, defaulting to 'vector'."
                )
                self.embedding_type = VectorType.vector

            if self.embedding_dimension is None:
                _logger.warning(
                    "Embedding dimension is not specified, defaulting to 1536."
                )
                self.embedding_dimension = PositiveInt(1_536)

            if self.embedding_index is None:
                _logger.warning(
                    "Embedding index is not specified, defaulting to 'DiskANN' with 'vector_cosine_ops' opclass."
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)

            self._embedding_index_name = None

            async with (
                self._connection() as conn,
                conn.cursor() as cursor,
            ):
                await cursor.execute(
                    sql.SQL(
                        """
                        create table {table_name} (
                            {id_column} uuid primary key,
                            {content_column} text,
                            {embedding_column} {embedding_type}({embedding_dimension}),
                            {metadata_columns}
                        )
                        """
                    ).format(
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        id_column=sql.Identifier(self.id_column),
                        content_column=sql.Identifier(self.content_column),
                        embedding_column=sql.Identifier(self.embedding_column),
                        embedding_type=sql.Identifier(self.embedding_type.value),
                        embedding_dimension=sql.Literal(self.embedding_dimension),
                        metadata_columns=sql.SQL(", ").join(
                            sql.SQL("{col} {type}").format(
                                col=sql.Identifier(col),
                                type=sql.SQL(type),  # type: ignore[arg-type]
                            )
                            for col, type in metadata_columns
                        ),
                    )
                )

    @asynccontextmanager
    async def _connection(self) -> AsyncGenerator[AsyncConnection, None]:
        if isinstance(self.connection, AsyncConnection):
            yield self.connection
        else:
            async with self.connection.connection() as conn:
                yield conn

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        return self.embedding

    async def create_index(self, *, concurrently: bool = False) -> bool:
        """Create the vector index on the embedding column (if not already exists).

        Builds a vector index for the configured ``embedding_column`` using the
        algorithm specified by ``embedding_index`` (``DiskANN``, ``HNSW`` or
        ``IVFFlat``). The effective index type name is inferred from the
        concrete ``Algorithm`` instance and the index name is generated as
        ``<table>_<column>_<type>_idx``. If an index has already been discovered
        (``_embedding_index_name`` is not ``None``) the operation is skipped.

        Prior to executing ``create index`` the per-build tuning parameters
        (returned by :meth:`Algorithm.index_settings`) are applied via ``set``
        GUCs so they only affect this session. Build-time options (returned by
        :meth:`Algorithm.build_settings`) are appended in a ``with (...)``
        clause.

        For quantized operator classes:
        - ``halfvec_*`` (scalar quantization) casts both the stored column and
          future query vectors to ``halfvec(dim)``.
        - ``bit_*`` (binary quantization) wraps the column with
          ``binary_quantize(col)::bit(dim)``.
        Otherwise the raw column is indexed.

        :param concurrently: When ``True`` uses ``create index concurrently`` to
            avoid long write-locks at the expense of a slower build.
        :type concurrently: bool
        :return: ``True`` if the index was created, ``False`` when an existing
            index prevented creation.
        :rtype: bool
        :raises AssertionError: If required attributes (``embedding_index`` or
            ``embedding_dimension``) are unexpectedly ``None``.
        """
        if self._embedding_index_name is not None:
            _logger.error(
                "Index '%s' already exists on table '%s.%s'; skipping index creation.",
                self._embedding_index_name,
                self.schema_name,
                self.table_name,
            )
            return False

        assert self.embedding_index is not None, (
            "embedding_index should have already been set"
        )
        assert self.embedding_dimension is not None, (
            "embedding_dimension should have already been set"
        )

        build_opts = self.embedding_index.build_settings()

        index_type_name = (
            "diskann"
            if isinstance(self.embedding_index, DiskANN)
            else "hnsw"
            if isinstance(self.embedding_index, HNSW)
            else "ivfflat"
        )

        index_name = f"{self.table_name}_{self.embedding_column}_{index_type_name}_idx"

        quantization = "other"
        if self.embedding_index.op_class in [
            VectorOpClass.halfvec_cosine_ops,
            VectorOpClass.halfvec_ip_ops,
            VectorOpClass.halfvec_l1_ops,
            VectorOpClass.halfvec_l2_ops,
        ]:
            quantization = "scalar"
        elif self.embedding_index.op_class in [
            VectorOpClass.bit_hamming_ops,
            VectorOpClass.bit_jaccard_ops,
        ]:
            quantization = "binary"

        async with self._connection() as conn, conn.cursor() as cursor:
            for opt, val in self.embedding_index.index_settings().items():
                _logger.debug("setting index option '%s' to '%s'", opt, val)
                await cursor.execute(
                    sql.SQL("set {option} to {value}").format(
                        option=sql.Identifier(opt), value=sql.Literal(val)
                    )
                )

            await cursor.execute(
                sql.SQL(
                    """
                    create index  {concurrently} {index_name}
                              on  {table_name}
                           using  {index_type} ({col_expr} {opclass})
                                  {embedding_opts}
                    """
                ).format(
                    concurrently=sql.SQL("concurrently" if concurrently else ""),
                    index_name=sql.Identifier(index_name),
                    table_name=sql.Identifier(self.schema_name, self.table_name),
                    index_type=sql.Identifier(index_type_name),
                    col_expr=sql.SQL("({col}::halfvec({dim}))").format(
                        col=sql.Identifier(self.embedding_column),
                        dim=sql.Literal(self.embedding_dimension),
                    )
                    if quantization == "scalar"
                    else sql.SQL("(binary_quantize({col})::bit({dim}))").format(
                        col=sql.Identifier(self.embedding_column),
                        dim=sql.Literal(self.embedding_dimension),
                    )
                    if quantization == "binary"
                    else sql.Identifier(self.embedding_column),
                    opclass=sql.Identifier(self.embedding_index.op_class.value),
                    embedding_opts=sql.SQL("with ({opts})").format(
                        opts=sql.SQL(", ").join(
                            sql.Identifier(k) + sql.SQL(" = ") + sql.Literal(v)
                            for k, v in build_opts.items()
                        )
                    )
                    if len(build_opts) > 0
                    else sql.SQL(""),
                )
            )

        self._embedding_index_name = index_name

        return True

    async def reindex(
        self, *, concurrently: bool = False, verbose: bool = False
    ) -> bool:
        """Reindex the existing vector index.

        Issues a ``reindex (concurrently <bool>, verbose <bool>) index`` command
        for the previously discovered or created index (tracked in
        ``_embedding_index_name``). The session-level index tuning GUCs
        (returned by :meth:`Algorithm.index_settings`) are applied beforehand to
        influence the reindex process (useful for algorithms whose maintenance
        cost or accuracy depends on these settings).

        :param concurrently: When ``True`` performs a concurrent reindex to
            minimize locking, trading speed for availability.
        :type concurrently: bool
        :param verbose: When ``True`` enables PostgreSQL verbose output, which
            may aid in diagnosing build performance issues.
        :type verbose: bool
        :return: ``True`` if reindex succeeded, ``False`` if no index existed.
        :rtype: bool
        :raises AssertionError: If ``embedding_index`` is unexpectedly ``None``.
        """
        if self._embedding_index_name is None:
            _logger.error(
                "No index exists on table '%s.%s'; skipping reindexing.",
                self.schema_name,
                self.table_name,
            )
            return False

        assert self.embedding_index is not None, (
            "embedding_index should have already been set"
        )

        async with self._connection() as conn, conn.cursor() as cursor:
            for opt, val in self.embedding_index.index_settings().items():
                _logger.debug("setting index option '%s' to '%s'", opt, val)
                await cursor.execute(
                    sql.SQL("set {option} to {value}").format(
                        option=sql.Identifier(opt), value=sql.Literal(val)
                    )
                )

            await cursor.execute(
                sql.SQL(
                    """
                    reindex (concurrently {concurrently}, verbose {verbose})
                      index {index_name}
                    """
                ).format(
                    concurrently=sql.Literal(concurrently),
                    verbose=sql.Literal(verbose),
                    index_name=sql.Identifier(
                        self.schema_name, self._embedding_index_name
                    ),
                )
            )

        return True

    @classmethod
    @override
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        """Create a store and add documents in one step.

        :param documents: The list of documents to add to the store.
        :type documents: list[Document]
        :param embedding: The embedding model to use for embedding vector generation.
        :type embedding: Embeddings

        Kwargs
        ------
        - connection: (required) psycopg AsyncConnection or AsyncConnectionPool
        - schema_name, table_name, id_column, content_column, embedding_column:
            customize table/column names
        - embedding_type: VectorType of the embedding column
        - embedding_dimension: dimension of the vector column
        - embedding_index: Algorithm describing the vector index
        - metadata_columns: str | list[str | (str, str)] | None to configure metadata storage
        - on_conflict_update (passed to add): bool to upsert existing rows

        :return: The created vector store instance.
        :rtype: Self
        """
        connection: AsyncConnection | AsyncConnectionPool = kwargs.pop("connection")
        schema_name: str = kwargs.pop("schema_name", "public")
        table_name: str = kwargs.pop("table_name", "langchain")
        id_column: str = kwargs.pop("id_column", "id")
        content_column: str = kwargs.pop("content_column", "content")
        embedding_column: str = kwargs.pop("embedding_column", "embedding")
        embedding_type: VectorType | None = kwargs.pop("embedding_type", None)
        embedding_dimension: PositiveInt | None = kwargs.pop(
            "embedding_dimension", None
        )
        embedding_index: Algorithm | None = kwargs.pop("embedding_index", None)
        metadata_columns: list[str] | list[tuple[str, str]] | str | None = kwargs.pop(
            "metadata_columns", "metadata"
        )
        vs = cls(
            embedding=embedding,
            connection=connection,
            schema_name=schema_name,
            table_name=table_name,
            id_column=id_column,
            content_column=content_column,
            embedding_column=embedding_column,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            embedding_index=embedding_index,
            metadata_columns=metadata_columns,
        )
        await vs.aadd_documents(documents, **kwargs)
        return vs

    @classmethod
    @override
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a store and add texts with optional metadata.

        :param texts: The list of texts to add to the store.
        :type texts: list[str]
        :param embedding: The embedding model to use for embedding vector generation.
        :type embedding: Embeddings
        :param metadatas: The list of metadata dictionaries corresponding to each text.
        :type metadatas: list[dict] | None
        :param ids: The list of custom IDs corresponding to each text. When ``ids``
            are not provided, UUIDs are generated.
        :type ids: list[str] | None

        Kwargs
        ------
        See :meth:`afrom_documents` for required and/or supported ``kwargs``.

        :return: The created vector store instance.
        :rtype: Self
        """
        connection: AsyncConnection | AsyncConnectionPool = kwargs.pop("connection")
        schema_name: str = kwargs.pop("schema_name", "public")
        table_name: str = kwargs.pop("table_name", "langchain")
        id_column: str = kwargs.pop("id_column", "id")
        content_column: str = kwargs.pop("content_column", "content")
        embedding_column: str = kwargs.pop("embedding_column", "embedding")
        embedding_type: VectorType | None = kwargs.pop("embedding_type", None)
        embedding_dimension: PositiveInt | None = kwargs.pop(
            "embedding_dimension", None
        )
        embedding_index: Algorithm | None = kwargs.pop("embedding_index", None)
        metadata_columns: list[str] | list[tuple[str, str]] | str | None = kwargs.pop(
            "metadata_columns", "metadata"
        )
        vs = cls(
            embedding=embedding,
            connection=connection,
            schema_name=schema_name,
            table_name=table_name,
            id_column=id_column,
            content_column=content_column,
            embedding_column=embedding_column,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            embedding_index=embedding_index,
            metadata_columns=metadata_columns,
        )
        await vs.aadd_texts(texts, metadatas, ids=ids, **kwargs)
        return vs

    @override
    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """Insert or upsert a batch of LangChain ``documents``.

        Kwargs
        ------
        - ids: list[str] custom ids, otherwise UUIDs or doc.id are used
        - on_conflict_update: bool to update existing rows on id conflict

        :return: Inserted ids.
        :rtype: list[str]
        """
        ids_: list[str] = kwargs.pop("ids", None) or [
            doc.id if doc.id is not None else str(uuid.uuid4()) for doc in documents
        ]
        texts_ = [doc.page_content for doc in documents]
        metadatas_ = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts_, metadatas_, ids=ids_, **kwargs)

    @override
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Insert or upsert ``texts`` with optional ``metadatas`` and embeddings.

        If an embeddings model is present, embeddings are computed and stored. When
        ``metadata_columns`` is a string, metadata is written as JSONB; otherwise only
        provided keys matching configured columns are stored.

        Kwargs
        ------
        - ids: list[str] custom ids, otherwise UUIDs are used
        - on_conflict_update: bool to update existing rows on id conflict

        :return: Inserted ids.
        :rtype: list[str]
        :raises ValueError: If the length of 'metadatas', 'texts', and 'ids' do not match.
        """
        texts_ = list(texts)
        if metadatas is not None and len(metadatas) != len(texts_):
            raise ValueError(
                "The length of 'metadatas' must match the length of 'texts'."
            )
        elif ids is not None and len(ids) != len(texts_):
            raise ValueError("The length of 'ids' must match the length of 'texts'.")

        metadatas_: list[dict] | cycle[dict] = metadatas or cycle([{}])
        ids_ = ids or [str(uuid.uuid4()) for _ in range(len(texts_))]

        on_conflict_update = bool(kwargs.pop("on_conflict_update", None))

        embeddings: np.ndarray | cycle[None] = cycle([None])
        embedding_column: list[str] = []
        if self.embeddings is not None:
            embeddings = np.array(
                await self.embeddings.aembed_documents([text for text in texts_]),
                dtype=np.float32,
            )
            embedding_column = [self.embedding_column]

        metadata_columns: list[str]
        if self.metadata_columns is None:
            metadata_columns = []
        elif isinstance(self.metadata_columns, list):
            metadata_columns = [
                col if isinstance(col, str) else col[0] for col in self.metadata_columns
            ]
        else:
            metadata_columns = [self.metadata_columns]

        async with self._connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.executemany(
                    sql.SQL(
                        """
                        insert into {table_name} ({columns})
                             values ({values})
                        on conflict ({id_column})
                                 do {update}
                          returning {id_column}
                        """
                    ).format(
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *embedding_column,
                                    *metadata_columns,
                                ],
                            )
                        ),
                        values=sql.SQL(", ").join(
                            map(
                                sql.Placeholder,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *embedding_column,
                                    *metadata_columns,
                                ],
                            )
                        ),
                        id_column=sql.Identifier(self.id_column),
                        update=sql.SQL(" ").join(
                            [
                                sql.SQL("update"),
                                sql.SQL("set"),
                                sql.SQL(", ").join(
                                    sql.SQL("{col} = {excluded_col}").format(
                                        col=sql.Identifier(col),
                                        excluded_col=sql.Identifier("excluded", col),
                                    )
                                    for col in [
                                        self.content_column,
                                        *embedding_column,
                                        *metadata_columns,
                                    ]
                                ),
                            ]
                        )
                        if on_conflict_update
                        else sql.SQL("nothing"),
                    ),
                    (
                        {
                            self.id_column: id_,
                            self.content_column: text_,
                            **(
                                {self.embedding_column: embedding_}
                                if embedding_ is not None
                                else {}
                            ),
                            **(
                                {metadata_columns[0]: Jsonb(metadata_)}
                                if isinstance(self.metadata_columns, str)
                                else {
                                    col: metadata_.get(col) for col in metadata_columns
                                }
                            ),
                        }
                        for id_, text_, embedding_, metadata_ in zip(
                            ids_,
                            texts_,
                            embeddings,
                            metadatas_,
                        )
                    ),
                    returning=True,
                )

                inserted_ids = []
                while True:
                    resultset = await cursor.fetchone()
                    if resultset is not None:
                        inserted_ids.append(str(resultset[self.id_column]))
                    if not cursor.nextset():
                        break
                return inserted_ids

    @override
    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by ids or truncate the table.

        If ``ids`` is ``None``, the table is truncated.

        Kwargs
        ------
        - restart: bool to restart (when True) or continue (when False) identity,
                   when truncating
        - cascade: bool to cascade (when True) or restrict (when False),
                   when truncating

        :return: True if the operation was successful, False otherwise.
        :rtype: bool | None
        """
        async with self._connection() as conn:
            try:
                async with conn.transaction() as _tx, conn.cursor() as cursor:
                    if ids is None:
                        restart = bool(kwargs.pop("restart", None))
                        cascade = bool(kwargs.pop("cascade", None))
                        await cursor.execute(
                            sql.SQL(
                                """
                                truncate table {table_name} {restart} {cascade}
                                """
                            ).format(
                                table_name=sql.Identifier(
                                    self.schema_name, self.table_name
                                ),
                                restart=sql.SQL(
                                    "restart identity"
                                    if restart
                                    else "continue identity"
                                ),
                                cascade=sql.SQL("cascade" if cascade else "restrict"),
                            )
                        )
                    else:
                        ids_ = [uuid.UUID(id) for id in ids]
                        await cursor.execute(
                            sql.SQL(
                                """
                                delete from {table_name}
                                      where {id_column} = any(%(id)s)
                                """
                            ).format(
                                table_name=sql.Identifier(
                                    self.schema_name, self.table_name
                                ),
                                id_column=sql.Identifier(self.id_column),
                            ),
                            {"id": ids_},
                        )
            except Exception:
                return False
            else:
                return True

    @override
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Fetch documents by their ``ids``.

        :param ids: Sequence of string ids.
        :type ids: Sequence[str]
        :return: Documents with metadata reconstructed from configured columns.
        :rtype: list[Document]
        """
        async with (
            self._connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            metadata_columns: list[str]
            if isinstance(self.metadata_columns, list):
                metadata_columns = [
                    col if isinstance(col, str) else col[0]
                    for col in self.metadata_columns
                ]
            elif isinstance(self.metadata_columns, str):
                metadata_columns = [self.metadata_columns]
            else:
                metadata_columns = []

            await cursor.execute(
                sql.SQL(
                    """
                    select {columns}
                      from {table_name}
                     where {id_column} = any(%(id)s)
                    """
                ).format(
                    columns=sql.SQL(", ").join(
                        map(
                            sql.Identifier,
                            [
                                self.id_column,
                                self.content_column,
                                *metadata_columns,
                            ],
                        )
                    ),
                    table_name=sql.Identifier(self.schema_name, self.table_name),
                    id_column=sql.Identifier(self.id_column),
                ),
                {"id": ids},
            )
            resultset = await cursor.fetchall()
            documents = [
                Document(
                    id=str(result[self.id_column]),
                    page_content=result[self.content_column],
                    metadata=(
                        result[metadata_columns[0]]
                        if isinstance(self.metadata_columns, str)
                        else {col: result[col] for col in metadata_columns}
                    ),
                )
                for result in resultset
            ]
            return documents

    async def _asimilarity_search_by_vector_with_distance(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float, np.ndarray | None]]:
        assert self.embedding_index is not None, (
            "embedding_index should have already been set"
        )
        return_embeddings = bool(kwargs.pop("return_embeddings", None))
        top_m = int(kwargs.pop("top_m", 5 * k))
        filter: Filter | None = kwargs.pop("filter", None)
        async with self._connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cursor:
                metadata_columns: list[str]
                if isinstance(self.metadata_columns, list):
                    metadata_columns = [
                        col if isinstance(col, str) else col[0]
                        for col in self.metadata_columns
                    ]
                elif isinstance(self.metadata_columns, str):
                    metadata_columns = [self.metadata_columns]
                else:
                    metadata_columns = []

                # do reranking for the following cases:
                #   - binary or scalar quantizations (for HNSW and IVFFlat), or
                #   - product quantization (for DiskANN)
                if (
                    self.embedding_index.op_class == VectorOpClass.bit_hamming_ops
                    or self.embedding_index.op_class == VectorOpClass.bit_jaccard_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_cosine_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_ip_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_l1_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_l2_ops
                    or (
                        isinstance(self.embedding_index, DiskANN)
                        and self.embedding_index.product_quantized
                    )
                ):
                    sql_query = sql.SQL(
                        """
                          select  {outer_columns},
                                  {embedding_column} {op} %(query)s as distance,
                                  {maybe_embedding_column}
                            from  (
                                     select {inner_columns}
                                       from {table_name}
                                      where {filter_expression}
                                   order by {expression} asc
                                      limit %(top_m)s
                                  ) i
                        order by  {embedding_column} {op} %(query)s asc
                           limit  %(top_k)s
                        """
                    ).format(
                        outer_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *metadata_columns,
                                ],
                            )
                        ),
                        embedding_column=sql.Identifier(self.embedding_column),
                        op=(
                            sql.SQL(
                                VectorOpClass.vector_cosine_ops.to_operator()
                            )  # TODO(arda): Think of getting this from outside
                            if (
                                self.embedding_index.op_class
                                in (
                                    VectorOpClass.bit_hamming_ops,
                                    VectorOpClass.bit_jaccard_ops,
                                )
                            )
                            else sql.SQL(self.embedding_index.op_class.to_operator())
                        ),
                        maybe_embedding_column=(
                            sql.Identifier(self.embedding_column)
                            if return_embeddings
                            else sql.SQL(" as ").join(
                                (sql.NULL, sql.Identifier(self.embedding_column))
                            )
                        ),
                        inner_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    self.embedding_column,
                                    *metadata_columns,
                                ],
                            )
                        ),
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        filter_expression=_filter_to_sql(filter),
                        expression=(
                            sql.SQL(
                                "binary_quantize({embedding_column})::bit({embedding_dim}) {op} binary_quantize({query})"
                            ).format(
                                embedding_column=sql.Identifier(self.embedding_column),
                                embedding_dim=sql.Literal(self.embedding_dimension),
                                op=sql.SQL(self.embedding_index.op_class.to_operator()),
                                query=sql.Placeholder("query"),
                            )
                            if self.embedding_index.op_class
                            in (
                                VectorOpClass.bit_hamming_ops,
                                VectorOpClass.bit_jaccard_ops,
                            )
                            else sql.SQL(
                                "{embedding_column}::halfvec({embedding_dim}) {op} {query}::halfvec({embedding_dim})"
                            ).format(
                                embedding_column=sql.Identifier(self.embedding_column),
                                embedding_dim=sql.Literal(self.embedding_dimension),
                                op=sql.SQL(self.embedding_index.op_class.to_operator()),
                                query=sql.Placeholder("query"),
                            )
                            if self.embedding_index.op_class
                            in (
                                VectorOpClass.halfvec_cosine_ops,
                                VectorOpClass.halfvec_ip_ops,
                                VectorOpClass.halfvec_l1_ops,
                                VectorOpClass.halfvec_l2_ops,
                            )
                            else sql.SQL("{embedding_column} {op} {query}").format(
                                embedding_column=sql.Identifier(self.embedding_column),
                                op=sql.SQL(self.embedding_index.op_class.to_operator()),
                                query=sql.Placeholder("query"),
                            )
                        ),
                    )
                # otherwise (i.e., no quantization), do not do reranking
                else:
                    sql_query = sql.SQL(
                        """
                          select  {outer_columns},
                                  {embedding_column} {op} %(query)s as distance,
                                  {maybe_embedding_column}
                            from  {table_name}
                           where  {filter_expression}
                        order by  {embedding_column} {op} %(query)s asc
                           limit  %(top_k)s
                        """
                    ).format(
                        outer_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *metadata_columns,
                                ],
                            )
                        ),
                        embedding_column=sql.Identifier(self.embedding_column),
                        op=sql.SQL(self.embedding_index.op_class.to_operator()),
                        maybe_embedding_column=(
                            sql.Identifier(self.embedding_column)
                            if return_embeddings
                            else sql.SQL(" as ").join(
                                (sql.NULL, sql.Identifier(self.embedding_column))
                            )
                        ),
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        filter_expression=_filter_to_sql(filter),
                    )

                await cursor.execute(
                    sql_query,
                    {
                        "query": np.array(embedding, dtype=np.float32),
                        "top_m": top_m,
                        "top_k": k,
                    },
                )

                resultset = await cursor.fetchall()

        return [
            (
                Document(
                    id=str(result[self.id_column]),
                    page_content=result[self.content_column],
                    metadata=(
                        result[metadata_columns[0]]
                        if isinstance(self.metadata_columns, str)
                        else {col: result[col] for col in metadata_columns}
                    ),
                ),
                result["distance"],
                result.get(self.embedding_column),  # type: ignore[return-value]
            )
            for result in resultset
        ]

    @override
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Similarity search for a query string using the configured index.

        :param query: Query text to embed and search.
        :type query: str
        :param k: Number of most similar documents.
        :type k: int

        Kwargs
        ------
        - filter: Filter | None; Optional filter to apply to the search.
        - top_m: int; Number of top results to prefetch when re-ranking (default: 5 * k).

        :return: Top-k documents.
        :rtype: list[Document]
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not set. Please provide an embeddings model to the AsyncAzurePGVectorStore."
            )
        embedding = await self.embeddings.aembed_query(query)
        return await self.asimilarity_search_by_vector(embedding, k=k, **kwargs)

    @override
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        assert self.embedding_index is not None, (
            "embedding_index should have already been set"
        )
        match self.embedding_index.op_class:
            case (
                VectorOpClass.vector_cosine_ops
                | VectorOpClass.halfvec_cosine_ops
                | VectorOpClass.sparsevec_cosine_ops
            ):
                return AsyncAzurePGVectorStore._cosine_relevance_score_fn
            case (
                VectorOpClass.vector_ip_ops
                | VectorOpClass.halfvec_ip_ops
                | VectorOpClass.sparsevec_ip_ops
            ):
                return AsyncAzurePGVectorStore._max_inner_product_relevance_score_fn
            case (
                VectorOpClass.vector_l2_ops
                | VectorOpClass.halfvec_l2_ops
                | VectorOpClass.sparsevec_l2_ops
            ):
                return AsyncAzurePGVectorStore._euclidean_relevance_score_fn
            case (
                VectorOpClass.vector_l1_ops
                | VectorOpClass.halfvec_l1_ops
                | VectorOpClass.sparsevec_l1_ops
            ):
                _logger.debug(
                    "Using the upper bound of 2 for the L1 distance, assuming unit-norm vectors"
                )
                return lambda x: 1.0 - x / 2.0
            case VectorOpClass.bit_hamming_ops:
                if self.embedding_dimension is None:
                    raise RuntimeError(
                        "Embedding dimension must be specified for bit_hamming_ops."
                    )
                embedding_dimension = int(self.embedding_dimension)
                return lambda x: 1.0 - x / embedding_dimension
            case VectorOpClass.bit_jaccard_ops:
                return lambda x: 1.0 - x
            case _:
                raise ValueError(
                    f"Unsupported vector op class: {self.embedding_index.op_class}. "
                    "Supported op classes are: "
                    f"{[t.value for t in VectorOpClass.__members__.values()]}"
                )

    @override
    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Similarity search returning (document, distance) pairs.

        :param query: Query text to embed and search.
        :type query: str
        :param k: Number of most similar documents.
        :type k: int

        Kwargs
        ------
        See :meth:`asimilarity_search` for supported ``kwargs``.

        :return: Top-k (document, distance) pairs.
        :rtype: list[tuple[Document, float]]
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not set. Please provide an embeddings model to the AsyncAzurePGVectorStore."
            )
        embedding = await self.embeddings.aembed_query(query)
        results = await self._asimilarity_search_by_vector_with_distance(
            embedding, k=k, **kwargs
        )
        return [(r[0], r[1]) for r in results]

    @override
    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Similarity search for a precomputed embedding vector.

        :param embedding: The precomputed embedding vector to search for.
        :type embedding: list[float]
        :param k: Number of most similar documents.
        :type k: int

        Kwargs
        ------
        See :meth:`asimilarity_search` for supported ``kwargs``.

        :return: Top-k documents.
        :rtype: list[Document]
        """
        return [
            doc
            for doc, *_ in await self._asimilarity_search_by_vector_with_distance(
                embedding, k=k, **kwargs
            )
        ]

    @override
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """MMR search for a query string.

        :param query: The query string to search for.
        :type query: str
        :param k: Number of most similar documents to return.
        :type k: int
        :param fetch_k: Candidate pool size before MMR reranking.
        :type fetch_k: int
        :param lambda_mult: Diversity vs. relevance trade-off parameter.
        :type lambda_mult: float

        Kwargs
        ------
        See :meth:`similarity_search` for supported ``kwargs``.

        :return: Top-k documents.
        :rtype: list[Document]
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not set. Please provide an embeddings model to the AsyncAzurePGVectorStore."
            )
        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    @override
    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """MMR search for a precomputed embedding vector.

        :param embedding: The precomputed embedding vector to search for.
        :type embedding: list[float]
        :param k: Number of most similar documents to return.
        :type k: int
        :param fetch_k: Candidate pool size before MMR reranking.
        :type fetch_k: int
        :param lambda_mult: Diversity vs. relevance trade-off parameter.
        :type lambda_mult: float

        Kwargs
        ------
        See :meth:`similarity_search` for supported ``kwargs``.

        :return: Top-k documents.
        :rtype: list[Document]
        """
        kwargs.update({"return_embeddings": True})
        results = await self._asimilarity_search_by_vector_with_distance(
            embedding, k=fetch_k, **kwargs
        )
        indices = utils.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [r[2] for r in results],
            lambda_mult,
            k,
        )
        return [results[i][0] for i in indices]

    # Synchronous methods are not implemented - use the sync version instead

    @classmethod
    @override
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def search(self, query: str, search_type: str, **kwargs: Any):
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def _similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )
