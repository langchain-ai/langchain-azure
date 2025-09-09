import re
from contextlib import nullcontext
from itertools import cycle
from typing import Any

import pytest
from langchain_core.documents import Document
from pgvector.psycopg import (  # type: ignore[import-untyped]
    register_vector,
    register_vector_async,
)
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pydantic import PositiveInt

from langchain_azure_postgresql.langchain import (
    AsyncAzurePGVectorStore,
    AzurePGVectorStore,
)

from .conftest import MockEmbedding, MockUUID, Table

# SQL constants to be used in tests
_GET_TABLE_COLUMNS_AND_TYPES = sql.SQL(
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
)


# Utility/assertion functions to be used in tests
def verify_table_created(table: Table, resultset: list[dict[str, Any]]) -> None:
    """Verify that the table has been created with the correct columns and types.

    :param table: Expected table to be created
    :type table: Table
    :param resultset: Actual result set from the database
    :type resultset: list[dict[str, Any]]
    """
    # Verify that the ID column has been created correctly
    result = next((r for r in resultset if r["column_name"] == table.id_column), None)
    assert result is not None, "ID column was not created in the table."
    assert result["column_type"] == "uuid", "ID column type is incorrect."

    # Verify that the content column has been created correctly
    result = next(
        (r for r in resultset if r["column_name"] == table.content_column), None
    )
    assert result is not None, "Content column was not created in the table."
    assert result["column_type"] == "text", "Content column type is incorrect."

    # Verify that the embedding column has been created correctly
    result = next(
        (r for r in resultset if r["column_name"] == table.embedding_column), None
    )
    assert result is not None, "Embedding column was not created in the table."
    embedding_column_type = result["column_type"]
    pattern = re.compile(r"(?P<type>\w+)(?:\((?P<dim>\d+)\))?")
    m = pattern.match(embedding_column_type if embedding_column_type else "")
    parsed_type: str | None = m.group("type") if m else None
    parsed_dim: PositiveInt | None = (
        PositiveInt(m.group("dim")) if m and m.group("dim") else None
    )
    assert parsed_type == table.embedding_type.value, (
        "Embedding column type is incorrect."
    )
    assert parsed_dim == table.embedding_dimension, (
        "Embedding column dimension is incorrect."
    )

    # Verify that metadata columns have been created correctly
    for column in table.metadata_columns:
        assert isinstance(column, tuple), (
            "Expecting a tuple for metadata columns (in the fixture)."
        )
        col_name, col_type = column[0], column[1]
        result = next((r for r in resultset if r["column_name"] == col_name), None)
        assert result is not None, (
            f"Metadata column '{col_name}' was not created in the table."
        )
        assert result["column_type"] == col_type, (
            f"Metadata column '{col_name}' type is incorrect."
        )


def verify_documents_inserted(
    documents: list[Document],
    resultset: list[dict[str, Any]],
) -> None:
    result_by_id = {str(r["id"]): r for r in resultset}

    for document in documents:
        assert document.id is not None, "Document ID is missing."
        result = result_by_id.get(document.id)
        assert result is not None, (
            f"Document with id '{document.id}' was not found in the result set."
        )

        assert result["content"] == document.page_content, (
            "Document content does not match."
        )

        assert result["embedding"] is not None, "Document embedding is missing."

        for key, value in document.metadata.items():
            assert result["metadata"][key] == value, (
                f"Document metadata '{key}' does not match."
            )


class TestAzurePGVectorStore:
    @pytest.mark.xfail(
        reason="Table creation failure tests not yet implemented",
        raises=AssertionError,
    )
    def test_table_creation_failure(self):
        assert False

    def test_table_creation_success(
        self, vectorstore: AzurePGVectorStore, table: Table
    ):
        with (
            vectorstore._connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                _GET_TABLE_COLUMNS_AND_TYPES,
                {
                    "schema_name": table.schema_name,
                    "table_name": table.table_name,
                },
            )
            resultset = cursor.fetchall()
        verify_table_created(table, resultset)

    @pytest.mark.parametrize(
        ["documents_ids", "expected"],
        [
            ("documents-ids-success", nullcontext(AzurePGVectorStore)),
            ("documents-no-ids-success", nullcontext(AzurePGVectorStore)),
            ("documents-ids-overridden-success", nullcontext(AzurePGVectorStore)),
            ("documents-ids-overridden-failure", pytest.raises(ValueError)),
        ],
        indirect=["documents_ids"],
        ids=[
            "documents-ids-success",
            "documents-no-ids-success",
            "documents-ids-overridden-success",
            "documents-ids-overridden-failure",
        ],
    )
    def test_vectorstore_initialization_from_documents(
        self,
        connection_pool: ConnectionPool,
        schema: str,
        mock_uuid: MockUUID,
        documents_ids: tuple[list[Document], list[str] | None],
        expected: nullcontext[AzurePGVectorStore] | pytest.RaisesExc,
    ):
        table_name = "vs_init_from_documents"
        embedding_dimension = 3
        embedding = MockEmbedding(dimension=embedding_dimension)

        documents, ids = documents_ids

        with expected as e:
            vectorstore = AzurePGVectorStore.from_documents(
                documents,
                embedding,
                ids=ids,
                connection=connection_pool,
                schema_name=schema,
                table_name=table_name,
                id_column="id",
                content_column="content",
                embedding_column="embedding",
                embedding_type=None,
                embedding_dimension=embedding_dimension,
                embedding_index=None,
                metadata_columns="metadata",
            )
            assert isinstance(vectorstore, e)  # type: ignore[arg-type]

        with connection_pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cur:
                if not isinstance(e, pytest.ExceptionInfo):
                    cur.execute(
                        sql.SQL(
                            """
                              select  id, content, embedding, metadata
                                from  {table}
                            order by  content asc
                            """
                        ).format(table=sql.Identifier(schema, table_name))
                    )
                    resultset = cur.fetchall()

                cur.execute(
                    sql.SQL(
                        """
                        drop table if exists {table} cascade
                        """
                    ).format(table=sql.Identifier(schema, table_name))
                )

        if not isinstance(e, pytest.ExceptionInfo):
            ids_ = ids or [str(id) for id in mock_uuid.generated_uuids]
            for id, document in zip(ids_, documents):
                document.id = id
            verify_documents_inserted(documents, resultset)

    @pytest.mark.parametrize(
        ["texts_ids_metadatas", "expected"],
        [
            ("texts-success", nullcontext(AzurePGVectorStore)),
            ("texts-ids-success", nullcontext(AzurePGVectorStore)),
            ("texts-metadatas-success", nullcontext(AzurePGVectorStore)),
            ("texts-ids-metadatas-success", nullcontext(AzurePGVectorStore)),
            ("texts-ids-failure", pytest.raises(ValueError)),
            ("texts-metadatas-failure", pytest.raises(ValueError)),
        ],
        indirect=["texts_ids_metadatas"],
        ids=[
            "texts-success",
            "texts-ids-success",
            "texts-metadatas-success",
            "texts-ids-metadatas-success",
            "texts-ids-failure",
            "texts-metadatas-failure",
        ],
    )
    def test_vectorstore_initialization_from_texts(
        self,
        connection_pool: ConnectionPool,
        schema: str,
        mock_uuid: MockUUID,
        texts_ids_metadatas: tuple[
            list[str], list[str] | None, list[dict[str, Any]] | None
        ],
        expected: nullcontext[AzurePGVectorStore] | pytest.RaisesExc,
    ):
        table_name = "vs_init_from_texts"
        embedding_dimension = 3
        embedding = MockEmbedding(dimension=embedding_dimension)

        texts, ids, metadatas = texts_ids_metadatas

        with expected as e:
            vectorstore = AzurePGVectorStore.from_texts(
                texts,
                embedding,
                metadatas=metadatas,
                ids=ids,
                connection=connection_pool,
                schema_name=schema,
                table_name=table_name,
                id_column="id",
                content_column="content",
                embedding_column="embedding",
                embedding_type=None,
                embedding_dimension=embedding_dimension,
                embedding_index=None,
                metadata_columns="metadata",
            )
            assert isinstance(vectorstore, e)  # type: ignore[arg-type]

        with connection_pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cur:
                if not isinstance(e, pytest.ExceptionInfo):
                    cur.execute(
                        sql.SQL(
                            """
                              select  id, content, embedding, metadata
                                from  {table}
                            order by  content asc
                            """
                        ).format(table=sql.Identifier(schema, table_name))
                    )
                    resultset = cur.fetchall()

                cur.execute(
                    sql.SQL(
                        """
                        drop table if exists {table} cascade
                        """
                    ).format(table=sql.Identifier(schema, table_name))
                )

        if not isinstance(e, pytest.ExceptionInfo):
            ids_ = ids or [str(id) for id in mock_uuid.generated_uuids]
            metadatas_: list[dict[str, Any]] | cycle[dict] = metadatas or cycle([{}])
            documents = [
                Document(id=id, page_content=text, metadata=metadata)
                for id, text, metadata in zip(ids_, texts, metadatas_)
            ]
            verify_documents_inserted(documents, resultset)

    @pytest.mark.parametrize(
        ["documents_ids", "expected"],
        [
            ("documents-ids-success", nullcontext(AzurePGVectorStore)),
            ("documents-ids-overridden-failure", pytest.raises(ValueError)),
        ],
        indirect=["documents_ids"],
        ids=["success", "failure"],
    )
    def test_add_documents(
        self,
        vectorstore: AzurePGVectorStore,
        documents_ids: tuple[list[Document], list[str] | None],
        expected: nullcontext[AzurePGVectorStore] | pytest.RaisesExc,
    ):
        documents, ids = documents_ids
        with expected:
            returned_ids = vectorstore.add_documents(documents, ids=ids)
            expected_ids = set(doc.id for doc in documents)
            actual_ids = set(returned_ids)
            assert actual_ids == expected_ids, "Inserted document IDs do not match"

    @pytest.mark.parametrize(
        ["documents_ids"],
        [
            ("documents-ids-success",),
            ("documents-ids-overridden-failure",),
        ],
        indirect=["documents_ids"],
        ids=["existing", "non-existing"],
    )
    def test_get_by_ids(
        self,
        vectorstore: AzurePGVectorStore,
        documents_ids: tuple[list[Document], list[str] | None],
    ):
        documents, ids = documents_ids
        assert all(doc.id is not None for doc in documents), (
            "All documents must have IDs"
        )
        ids_ = ids or [doc.id for doc in documents]  # type: ignore[misc]
        retrieved = vectorstore.get_by_ids(ids_)

        documents_set = set(ids_)
        retrieved_set = set(doc.id for doc in retrieved)
        assert retrieved_set <= documents_set, (
            "Retrieved documents must be a subset of sought-after documents"
        )

        if len(ids_) != len(documents):  # failure case; no documents should be there
            assert len(retrieved) == 0, "Retrieved documents should be empty"

    @pytest.mark.parametrize(
        ["documents_ids", "truncate"],
        [
            ("documents-ids-success", False),
            ("documents-ids-overridden-failure", False),
            ("documents-ids-success", True),
        ],
        indirect=["documents_ids"],
        ids=["some", "non-existing", "all"],
    )
    def test_delete(
        self,
        vectorstore: AzurePGVectorStore,
        documents_ids: tuple[list[Document], list[str] | None],
        truncate: bool,
    ):
        documents, ids = documents_ids
        assert all(doc.id is not None for doc in documents), (
            "All documents must have IDs"
        )
        ids_ = (
            None if truncate else (list(ids) if ids else [doc.id for doc in documents])
        )  # type: ignore[misc]

        if ids_ is not None:
            ids_.pop()

        assert vectorstore.delete(ids_), "Failed to delete documents"

        with (
            vectorstore._connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                sql.SQL(
                    """
                    select  {id_column} as id
                      from  {table_name}
                    """
                ).format(
                    id_column=sql.Identifier(vectorstore.id_column),
                    table_name=sql.Identifier(
                        vectorstore.schema_name, vectorstore.table_name
                    ),
                )
            )
            resultset = cursor.fetchall()

        deleted_set = set(ids_) if ids_ is not None else set()
        remaining_set = set(str(r["id"]) for r in resultset)

        assert len(deleted_set & remaining_set) == 0, (
            "Deleted document IDs should not exist in the remaining set"
        )

        if truncate:
            assert len(remaining_set) == 0, (
                "No documents should exist when the table is truncated"
            )
        else:
            assert len(remaining_set) > 0, (
                "Some documents should still exist when not all are deleted"
            )

    @pytest.mark.parametrize(
        ["texts_ids_metadatas", "expected"],
        [
            ("texts-ids-success", nullcontext(AzurePGVectorStore)),
            ("texts-ids-failure", pytest.raises(ValueError)),
        ],
        indirect=["texts_ids_metadatas"],
        ids=["success", "failure"],
    )
    def test_add_texts(
        self,
        vectorstore: AzurePGVectorStore,
        texts_ids_metadatas: tuple[
            list[str], list[str] | None, list[dict[str, Any]] | None
        ],
        expected: nullcontext[AzurePGVectorStore] | pytest.RaisesExc,
    ):
        texts, ids, metadatas = texts_ids_metadatas
        assert ids is not None, "IDs must be provided for this test"
        with expected:
            returned_ids = vectorstore.add_texts(texts, ids=ids, metadatas=metadatas)
            expected_ids = set(ids)
            actual_ids = set(returned_ids)
            assert actual_ids == expected_ids, "Inserted text IDs do not match"

    @pytest.mark.parametrize(
        ["query", "k"],
        [
            ("query about cats", 2),
            ("query about animals", 3),
        ],
        ids=["search-cats", "search-animals"],
    )
    def test_similarity_search(
        self, vectorstore: AzurePGVectorStore, query: str, k: int
    ):
        documents = vectorstore.similarity_search(query, k)

        if ("cats" in query) or ("animals" in query):
            assert len(documents) == k, f"Expected {k} results"
            assert any("cats" in doc.page_content for doc in documents) or any(
                "tigers" in doc.page_content for doc in documents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in doc.page_content for doc in documents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in doc.page_content for doc in documents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in doc.page_content for doc in documents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )

    @pytest.mark.parametrize(
        ["query", "k"],
        [
            ("query about cats", 2),
            ("query about animals", 3),
        ],
        ids=["search-cats", "search-animals"],
    )
    def test_similarity_search_with_score(
        self, vectorstore: AzurePGVectorStore, query: str, k: int
    ):
        retrieved = vectorstore.similarity_search_with_score(query, k)

        documents = [doc for doc, _score in retrieved]
        scores = [score for _doc, score in retrieved]

        if ("cats" in query) or ("animals" in query):
            assert len(documents) == k, f"Expected {k} results"
            assert any("cats" in doc.page_content for doc in documents) or any(
                "tigers" in doc.page_content for doc in documents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in doc.page_content for doc in documents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in doc.page_content for doc in documents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in doc.page_content for doc in documents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )

        assert all(x <= y for x, y in zip(scores, scores[1:])), (
            "Scores must be non-decreasing"
        )

    @pytest.mark.parametrize(
        ["query", "k"],
        [
            ("query about cats", 2),
            ("query about animals", 3),
        ],
        ids=["search-cats", "search-animals"],
    )
    def test_similarity_search_by_vector(
        self, vectorstore: AzurePGVectorStore, query: str, k: int
    ):
        assert vectorstore.embedding is not None, (
            "Vectorstore's embedding is not initialized"
        )
        embedding = vectorstore.embedding.embed_query(query)
        documents = vectorstore.similarity_search_by_vector(embedding, k)

        if ("cats" in query) or ("animals" in query):
            assert len(documents) == k, f"Expected {k} results"
            assert any("cats" in doc.page_content for doc in documents) or any(
                "tigers" in doc.page_content for doc in documents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in doc.page_content for doc in documents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in doc.page_content for doc in documents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in doc.page_content for doc in documents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )

    @pytest.mark.xfail(
        reason="MMR search tests rely on random number generation (can be flaky)",
        raises=AssertionError,
        strict=False,
    )
    @pytest.mark.parametrize(
        ["query", "k", "lambda_mult"],
        [
            ("query about dogs", 3, 1.0),
            ("query about dogs", 3, 0.5),
            ("query about dogs", 3, 0.0),
        ],
        ids=["search-dogs-accurate", "search-dogs-balanced", "search-dogs-diverse"],
    )
    def test_max_marginal_relevance_search(
        self, vectorstore: AzurePGVectorStore, query: str, k: int, lambda_mult: float
    ):
        documents = vectorstore.max_marginal_relevance_search(
            query, k, lambda_mult=lambda_mult
        )

        assert len(documents) == k, f"Expected {k} results"

        if lambda_mult == 1.0:  # accurate
            expected_words = ["dogs", "cats", "tigers"]
        elif lambda_mult == 0.5:  # balanced
            expected_words = ["dogs", "tigers", "plants"]
        elif lambda_mult == 0.0:  # diverse
            expected_words = ["dogs", "plants", "tigers"]

        for expected_word, document in zip(expected_words, documents):
            assert expected_word in document.page_content, (
                f"Expected word '{expected_word}' to be in retrieved document's contents for query: {query}"
            )

    @pytest.mark.xfail(
        reason="MMR search tests rely on random number generation (can be flaky)",
        raises=AssertionError,
        strict=False,
    )
    @pytest.mark.parametrize(
        ["query", "k", "lambda_mult"],
        [
            ("query about dogs", 3, 1.0),
            ("query about dogs", 3, 0.5),
            ("query about dogs", 3, 0.0),
        ],
        ids=["search-dogs-accurate", "search-dogs-balanced", "search-dogs-diverse"],
    )
    def test_max_marginal_relevance_search_by_vector(
        self, vectorstore: AzurePGVectorStore, query: str, k: int, lambda_mult: float
    ):
        assert vectorstore.embedding is not None, (
            "Vectorstore's embedding is not initialized"
        )
        embedding = vectorstore.embedding.embed_query(query)
        documents = vectorstore.max_marginal_relevance_search_by_vector(
            embedding, k, lambda_mult=lambda_mult
        )

        assert len(documents) == k, f"Expected {k} results"

        if lambda_mult == 1.0:  # accurate
            expected_words = ["dogs", "cats", "tigers"]
        elif lambda_mult == 0.5:  # balanced
            expected_words = ["dogs", "tigers", "plants"]
        elif lambda_mult == 0.0:  # diverse
            expected_words = ["dogs", "plants", "tigers"]

        for expected_word, document in zip(expected_words, documents):
            assert expected_word in document.page_content, (
                f"Expected word '{expected_word}' to be in retrieved document's contents for query: {query}"
            )


class TestAsyncAzurePGVectorStore:
    @pytest.mark.xfail(
        reason="Table creation failure tests not yet implemented",
        raises=AssertionError,
    )
    async def test_table_creation_failure(self):
        assert False

    async def test_table_creation_success(
        self, async_vectorstore: AsyncAzurePGVectorStore, async_table: Table
    ):
        async with (
            async_vectorstore._connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                _GET_TABLE_COLUMNS_AND_TYPES,
                {
                    "schema_name": async_table.schema_name,
                    "table_name": async_table.table_name,
                },
            )
            resultset = await cursor.fetchall()
        verify_table_created(async_table, resultset)

    @pytest.mark.parametrize(
        ["documents_ids", "expected"],
        [
            ("documents-ids-success", nullcontext(AsyncAzurePGVectorStore)),
            ("documents-no-ids-success", nullcontext(AsyncAzurePGVectorStore)),
            ("documents-ids-overridden-success", nullcontext(AsyncAzurePGVectorStore)),
            ("documents-ids-overridden-failure", pytest.raises(ValueError)),
        ],
        indirect=["documents_ids"],
        ids=[
            "documents-ids-success",
            "documents-no-ids-success",
            "documents-ids-overridden-success",
            "documents-ids-overridden-failure",
        ],
    )
    async def test_vectorstore_initialization_from_documents(
        self,
        async_connection_pool: AsyncConnectionPool,
        async_schema: str,
        mock_uuid: MockUUID,
        documents_ids: tuple[list[Document], list[str] | None],
        expected: nullcontext[AsyncAzurePGVectorStore] | pytest.RaisesExc,
    ):
        table_name = "async_vs_init_from_documents"
        embedding_dimension = 3
        embedding = MockEmbedding(dimension=embedding_dimension)

        documents, ids = documents_ids

        with expected as e:
            vectorstore = await AsyncAzurePGVectorStore.afrom_documents(
                documents,
                embedding,
                ids=ids,
                connection=async_connection_pool,
                schema_name=async_schema,
                table_name=table_name,
                id_column="id",
                content_column="content",
                embedding_column="embedding",
                embedding_type=None,
                embedding_dimension=embedding_dimension,
                embedding_index=None,
                metadata_columns="metadata",
            )
            assert isinstance(vectorstore, e)  # type: ignore[arg-type]

        async with async_connection_pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cur:
                if not isinstance(e, pytest.ExceptionInfo):
                    await cur.execute(
                        sql.SQL(
                            """
                              select  id, content, embedding, metadata
                                from  {table}
                            order by  content asc
                            """
                        ).format(table=sql.Identifier(async_schema, table_name))
                    )
                    resultset = await cur.fetchall()

                await cur.execute(
                    sql.SQL(
                        """
                        drop table if exists {table} cascade
                        """
                    ).format(table=sql.Identifier(async_schema, table_name))
                )

        if not isinstance(e, pytest.ExceptionInfo):
            ids_ = ids or [str(id) for id in mock_uuid.generated_uuids]
            for id, document in zip(ids_, documents):
                document.id = id
            verify_documents_inserted(documents, resultset)

    @pytest.mark.parametrize(
        ["texts_ids_metadatas", "expected"],
        [
            ("texts-success", nullcontext(AsyncAzurePGVectorStore)),
            ("texts-ids-success", nullcontext(AsyncAzurePGVectorStore)),
            ("texts-metadatas-success", nullcontext(AsyncAzurePGVectorStore)),
            ("texts-ids-metadatas-success", nullcontext(AsyncAzurePGVectorStore)),
            ("texts-ids-failure", pytest.raises(ValueError)),
            ("texts-metadatas-failure", pytest.raises(ValueError)),
        ],
        indirect=["texts_ids_metadatas"],
        ids=[
            "texts-success",
            "texts-ids-success",
            "texts-metadatas-success",
            "texts-ids-metadatas-success",
            "texts-ids-failure",
            "texts-metadatas-failure",
        ],
    )
    async def test_vectorstore_initialization_from_texts(
        self,
        async_connection_pool: AsyncConnectionPool,
        async_schema: str,
        mock_uuid: MockUUID,
        texts_ids_metadatas: tuple[
            list[str], list[str] | None, list[dict[str, Any]] | None
        ],
        expected: nullcontext[AsyncAzurePGVectorStore] | pytest.RaisesExc,
    ):
        table_name = "async_vs_init_from_texts"
        embedding_dimension = 3
        embedding = MockEmbedding(dimension=embedding_dimension)

        texts, ids, metadatas = texts_ids_metadatas

        with expected as e:
            vectorstore = await AsyncAzurePGVectorStore.afrom_texts(
                texts,
                embedding,
                metadatas=metadatas,
                ids=ids,
                connection=async_connection_pool,
                schema_name=async_schema,
                table_name=table_name,
                id_column="id",
                content_column="content",
                embedding_column="embedding",
                embedding_type=None,
                embedding_dimension=embedding_dimension,
                embedding_index=None,
                metadata_columns="metadata",
            )
            assert isinstance(vectorstore, e)  # type: ignore[arg-type]

        async with async_connection_pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cur:
                if not isinstance(e, pytest.ExceptionInfo):
                    await cur.execute(
                        sql.SQL(
                            """
                              select  id, content, embedding, metadata
                                from  {table}
                            order by  content asc
                            """
                        ).format(table=sql.Identifier(async_schema, table_name))
                    )
                    resultset = await cur.fetchall()

                await cur.execute(
                    sql.SQL(
                        """
                        drop table if exists {table} cascade
                        """
                    ).format(table=sql.Identifier(async_schema, table_name))
                )

        if not isinstance(e, pytest.ExceptionInfo):
            ids_ = ids or [str(id) for id in mock_uuid.generated_uuids]
            metadatas_: list[dict[str, Any]] | cycle[dict] = metadatas or cycle([{}])
            documents = [
                Document(id=id, page_content=text, metadata=metadata)
                for id, text, metadata in zip(ids_, texts, metadatas_)
            ]
            verify_documents_inserted(documents, resultset)

    @pytest.mark.parametrize(
        ["documents_ids", "expected"],
        [
            ("documents-ids-success", nullcontext(AsyncAzurePGVectorStore)),
            ("documents-ids-overridden-failure", pytest.raises(ValueError)),
        ],
        indirect=["documents_ids"],
        ids=["success", "failure"],
    )
    async def test_add_documents(
        self,
        async_vectorstore: AsyncAzurePGVectorStore,
        documents_ids: tuple[list[Document], list[str] | None],
        expected: nullcontext[AsyncAzurePGVectorStore] | pytest.RaisesExc,
    ):
        documents, ids = documents_ids
        with expected:
            returned_ids = await async_vectorstore.aadd_documents(documents, ids=ids)
            expected_ids = set(doc.id for doc in documents)
            actual_ids = set(returned_ids)
            assert actual_ids == expected_ids, "Inserted document IDs do not match"

    @pytest.mark.parametrize(
        ["documents_ids"],
        [
            ("documents-ids-success",),
            ("documents-ids-overridden-failure",),
        ],
        indirect=["documents_ids"],
        ids=["existing", "non-existing"],
    )
    async def test_get_by_ids(
        self,
        async_vectorstore: AsyncAzurePGVectorStore,
        documents_ids: tuple[list[Document], list[str] | None],
    ):
        documents, ids = documents_ids
        assert all(doc.id is not None for doc in documents), (
            "All documents must have IDs"
        )
        ids_ = ids or [doc.id for doc in documents]  # type: ignore[misc]
        retrieved = await async_vectorstore.aget_by_ids(ids_)

        documents_set = set(ids_)
        retrieved_set = set(doc.id for doc in retrieved)
        assert retrieved_set <= documents_set, (
            "Retrieved documents must be a subset of sought-after documents"
        )

        if len(ids_) != len(documents):  # failure case; no documents should be there
            assert len(retrieved) == 0, "Retrieved documents should be empty"

    @pytest.mark.parametrize(
        ["documents_ids", "truncate"],
        [
            ("documents-ids-success", False),
            ("documents-ids-overridden-failure", False),
            ("documents-ids-success", True),
        ],
        indirect=["documents_ids"],
        ids=["some", "non-existing", "all"],
    )
    async def test_delete(
        self,
        async_vectorstore: AsyncAzurePGVectorStore,
        documents_ids: tuple[list[Document], list[str] | None],
        truncate: bool,
    ):
        documents, ids = documents_ids
        assert all(doc.id is not None for doc in documents), (
            "All documents must have IDs"
        )
        ids_ = (
            None if truncate else (list(ids) if ids else [doc.id for doc in documents])
        )  # type: ignore[misc]

        if ids_ is not None:
            ids_.pop()

        assert await async_vectorstore.adelete(ids_), "Failed to delete documents"

        async with (
            async_vectorstore._connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                sql.SQL(
                    """
                    select  {id_column} as id
                      from  {table_name}
                    """
                ).format(
                    id_column=sql.Identifier(async_vectorstore.id_column),
                    table_name=sql.Identifier(
                        async_vectorstore.schema_name, async_vectorstore.table_name
                    ),
                )
            )
            resultset = await cursor.fetchall()

        deleted_set = set(ids_) if ids_ is not None else set()
        remaining_set = set(str(r["id"]) for r in resultset)

        assert len(deleted_set & remaining_set) == 0, (
            "Deleted document IDs should not exist in the remaining set"
        )

        if truncate:
            assert len(remaining_set) == 0, (
                "No documents should exist when the table is truncated"
            )
        else:
            assert len(remaining_set) > 0, (
                "Some documents should still exist when not all are deleted"
            )

    @pytest.mark.parametrize(
        ["texts_ids_metadatas", "expected"],
        [
            ("texts-ids-success", nullcontext(AsyncAzurePGVectorStore)),
            ("texts-ids-failure", pytest.raises(ValueError)),
        ],
        indirect=["texts_ids_metadatas"],
        ids=["success", "failure"],
    )
    async def test_add_texts(
        self,
        async_vectorstore: AsyncAzurePGVectorStore,
        texts_ids_metadatas: tuple[
            list[str], list[str] | None, list[dict[str, Any]] | None
        ],
        expected: nullcontext[AsyncAzurePGVectorStore] | pytest.RaisesExc,
    ):
        texts, ids, metadatas = texts_ids_metadatas
        assert ids is not None, "IDs must be provided for this test"
        with expected:
            returned_ids = await async_vectorstore.aadd_texts(
                texts, ids=ids, metadatas=metadatas
            )
            expected_ids = set(ids)
            actual_ids = set(returned_ids)
            assert actual_ids == expected_ids, "Inserted text IDs do not match"

    @pytest.mark.parametrize(
        ["query", "k"],
        [
            ("query about cats", 2),
            ("query about animals", 3),
        ],
        ids=["search-cats", "search-animals"],
    )
    async def test_similarity_search(
        self, async_vectorstore: AsyncAzurePGVectorStore, query: str, k: int
    ):
        documents = await async_vectorstore.asimilarity_search(query, k)

        if ("cats" in query) or ("animals" in query):
            assert len(documents) == k, f"Expected {k} results"
            assert any("cats" in doc.page_content for doc in documents) or any(
                "tigers" in doc.page_content for doc in documents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in doc.page_content for doc in documents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in doc.page_content for doc in documents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in doc.page_content for doc in documents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )

    @pytest.mark.parametrize(
        ["query", "k"],
        [
            ("query about cats", 2),
            ("query about animals", 3),
        ],
        ids=["search-cats", "search-animals"],
    )
    async def test_similarity_search_with_score(
        self, async_vectorstore: AsyncAzurePGVectorStore, query: str, k: int
    ):
        retrieved = await async_vectorstore.asimilarity_search_with_score(query, k)

        documents = [doc for doc, _score in retrieved]
        scores = [score for _doc, score in retrieved]

        if ("cats" in query) or ("animals" in query):
            assert len(documents) == k, f"Expected {k} results"
            assert any("cats" in doc.page_content for doc in documents) or any(
                "tigers" in doc.page_content for doc in documents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in doc.page_content for doc in documents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in doc.page_content for doc in documents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in doc.page_content for doc in documents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )

        assert all(x <= y for x, y in zip(scores, scores[1:])), (
            "Scores must be non-decreasing"
        )

    @pytest.mark.parametrize(
        ["query", "k"],
        [
            ("query about cats", 2),
            ("query about animals", 3),
        ],
        ids=["search-cats", "search-animals"],
    )
    async def test_similarity_search_by_vector(
        self, async_vectorstore: AsyncAzurePGVectorStore, query: str, k: int
    ):
        assert async_vectorstore.embedding is not None, (
            "Vectorstore's embedding is not initialized"
        )
        embedding = async_vectorstore.embedding.embed_query(query)
        documents = await async_vectorstore.asimilarity_search_by_vector(embedding, k)

        if ("cats" in query) or ("animals" in query):
            assert len(documents) == k, f"Expected {k} results"
            assert any("cats" in doc.page_content for doc in documents) or any(
                "tigers" in doc.page_content for doc in documents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in doc.page_content for doc in documents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in doc.page_content for doc in documents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in doc.page_content for doc in documents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )

    @pytest.mark.xfail(
        reason="MMR search tests rely on random number generation (can be flaky)",
        raises=AssertionError,
        strict=False,
    )
    @pytest.mark.parametrize(
        ["query", "k", "lambda_mult"],
        [
            ("query about dogs", 3, 1.0),
            ("query about dogs", 3, 0.5),
            ("query about dogs", 3, 0.0),
        ],
        ids=["search-dogs-accurate", "search-dogs-balanced", "search-dogs-diverse"],
    )
    async def test_max_marginal_relevance_search(
        self,
        async_vectorstore: AsyncAzurePGVectorStore,
        query: str,
        k: int,
        lambda_mult: float,
    ):
        documents = await async_vectorstore.amax_marginal_relevance_search(
            query, k, lambda_mult=lambda_mult
        )

        assert len(documents) == k, f"Expected {k} results"

        if lambda_mult == 1.0:  # accurate
            expected_words = ["dogs", "cats", "tigers"]
        elif lambda_mult == 0.5:  # balanced
            expected_words = ["dogs", "tigers", "plants"]
        elif lambda_mult == 0.0:  # diverse
            expected_words = ["dogs", "plants", "tigers"]

        for expected_word, document in zip(expected_words, documents):
            assert expected_word in document.page_content, (
                f"Expected word '{expected_word}' to be in retrieved document's contents for query: {query}"
            )

    @pytest.mark.xfail(
        reason="MMR search tests rely on random number generation (can be flaky)",
        raises=AssertionError,
        strict=False,
    )
    @pytest.mark.parametrize(
        ["query", "k", "lambda_mult"],
        [
            ("query about dogs", 3, 1.0),
            ("query about dogs", 3, 0.5),
            ("query about dogs", 3, 0.0),
        ],
        ids=["search-dogs-accurate", "search-dogs-balanced", "search-dogs-diverse"],
    )
    async def test_max_marginal_relevance_search_by_vector(
        self,
        async_vectorstore: AsyncAzurePGVectorStore,
        query: str,
        k: int,
        lambda_mult: float,
    ):
        assert async_vectorstore.embedding is not None, (
            "Vectorstore's embedding is not initialized"
        )
        embedding = async_vectorstore.embedding.embed_query(query)
        documents = await async_vectorstore.amax_marginal_relevance_search_by_vector(
            embedding, k, lambda_mult=lambda_mult
        )

        assert len(documents) == k, f"Expected {k} results"

        if lambda_mult == 1.0:  # accurate
            expected_words = ["dogs", "cats", "tigers"]
        elif lambda_mult == 0.5:  # balanced
            expected_words = ["dogs", "tigers", "plants"]
        elif lambda_mult == 0.0:  # diverse
            expected_words = ["dogs", "plants", "tigers"]

        for expected_word, document in zip(expected_words, documents):
            assert expected_word in document.page_content, (
                f"Expected word '{expected_word}' to be in retrieved document's contents for query: {query}"
            )
