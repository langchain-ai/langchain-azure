import re
from typing import Any

import pytest
from langchain_core.embeddings import FakeEmbeddings
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pydantic import PositiveInt

from langchain_azure_postgresql.langchain import (
    AsyncAzurePGVectorStore,
    AzurePGVectorStore,
)

from .conftest import Table

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


class TestAzurePGVectorStore:
    def test_vectorstore_creates_table(
        self, connection_pool: ConnectionPool, table: Table
    ):
        if table.existing:
            pytest.skip(reason="Table already exists, skipping creation test.")

        embedding = FakeEmbeddings(size=table.embedding_dimension)

        _vector_store = AzurePGVectorStore(
            embedding=embedding,
            connection_pool=connection_pool,
            schema_name=table.schema_name,
            table_name=table.table_name,
            id_column=table.id_column,
            content_column=table.content_column,
            embedding_column=table.embedding_column,
            embedding_type=table.embedding_type,
            embedding_dimension=table.embedding_dimension,
            metadata_columns=table.metadata_columns,
        )

        # Check if the table was created successfully
        with (
            connection_pool.connection() as conn,
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


class TestAsyncAzurePGVectorStore:
    async def test_vectorstore_creates_table(
        self, async_connection_pool: AsyncConnectionPool, async_table: Table
    ):
        if async_table.existing:
            pytest.skip(reason="Table already exists, skipping creation test.")

        embedding = FakeEmbeddings(size=async_table.embedding_dimension)

        _async_vector_store = AsyncAzurePGVectorStore(
            embedding=embedding,
            connection_pool=async_connection_pool,
            schema_name=async_table.schema_name,
            table_name=async_table.table_name,
            id_column=async_table.id_column,
            content_column=async_table.content_column,
            embedding_column=async_table.embedding_column,
            embedding_type=async_table.embedding_type,
            embedding_dimension=async_table.embedding_dimension,
            metadata_columns=async_table.metadata_columns,
        )

        # Check if the table was created successfully
        async with (
            async_connection_pool.connection() as conn,
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
