"""Asynchronous VectorStore integration for Azure Database for PostgreSQL using LangChain."""

import logging
import re
import sys
import uuid
from collections.abc import Callable, Iterable, Sequence
from itertools import cycle
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, utils
from pgvector.psycopg import register_vector_async  # type: ignore[import-untyped]
from psycopg import sql
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
from ...common._shared import run_coroutine_in_sync
from .._shared import Filter, filter_to_sql

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
    embedding: Embeddings | None = None
    connection_pool: AsyncConnectionPool
    schema_name: str = "public"
    table_name: str = "langchain"
    id_column: str = "id"
    content_column: str = "content"
    embedding_column: str = "embedding"
    embedding_type: VectorType | None = None
    embedding_dimension: PositiveInt | None = None
    embedding_index: Algorithm | None = None
    metadata_columns: list[str] | list[tuple[str, str]] | str | None = "metadata"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow arbitrary types like Embeddings and AsyncConnectionPool
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
        run_coroutine_in_sync(coroutine)

        return self

    async def _ensure_table_verified(self) -> None:
        async with (
            self.connection_pool.connection() as conn,
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
                self.connection_pool.connection() as conn,
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
                                  ci.reloptions as index_opts
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

                    for row in resultset:
                        if (
                            row["index_type"] == index_type
                            and row["index_opclass"] == index_opclass
                        ):
                            _logger.info(
                                "found a matching index: %s. overriding embedding_index.",
                                row,
                            )
                            index = (
                                DiskANN(op_class=index_opclass, **row["index_opts"])
                                if index_type == "diskann"
                                else HNSW(op_class=index_opclass, **row["index_opts"])
                                if index_type == "hnsw"
                                else IVFFlat(
                                    op_class=index_opclass, **row["index_opts"]
                                )
                            )
                            self.embedding_index = index
                            break
            elif self.embedding_index is None:
                _logger.info(
                    "embedding_index is not specified, and no vector index found in table '%s.%s'. defaulting to 'DiskANN' with 'vector_cosine_ops' opclass.",
                    self.schema_name,
                    self.table_name,
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)

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

            async with (
                self.connection_pool.connection() as conn,
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

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        return self.embedding

    @classmethod
    @override
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        connection_pool: AsyncConnectionPool = kwargs.pop("connection_pool")
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
            connection_pool=connection_pool,
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
        connection_pool: AsyncConnectionPool = kwargs.pop("connection_pool")
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
            connection_pool=connection_pool,
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
                self.embeddings.embed_documents([text for text in texts_]),
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

        async with self.connection_pool.connection() as conn:
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
        async with self.connection_pool.connection() as conn:
            await conn.set_autocommit(True)
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
        async with (
            self.connection_pool.connection() as conn,
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
        async with self.connection_pool.connection() as conn:
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
                        filter_expression=filter_to_sql(filter),
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
                        filter_expression=filter_to_sql(filter),
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
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not set. Please provide an embeddings model to the AsyncAzurePGVectorStore."
            )
        embedding = self.embeddings.embed_query(query)
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
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not set. Please provide an embeddings model to the AsyncAzurePGVectorStore."
            )
        embedding = self.embeddings.embed_query(query)
        results = await self._asimilarity_search_by_vector_with_distance(
            embedding, k=k, **kwargs
        )
        return [(r[0], r[1]) for r in results]

    @override
    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
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
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings are not set. Please provide an embeddings model to the AsyncAzurePGVectorStore."
            )
        embedding = self.embeddings.embed_query(query)
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

    @override
    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError(
            "asearch method is not implemented for AsyncAzurePGVectorStore."
        )

    @override
    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await self.asimilarity_search_with_score(query, k, **kwargs)

    @override
    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return await self.asimilarity_search_with_score(query, k, **kwargs)
