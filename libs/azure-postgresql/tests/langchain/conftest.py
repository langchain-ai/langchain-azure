"""Pytest fixtures for LangChain PostgreSQL vectorstore tests.

This module provides async and sync table fixtures plus helpers used by
tests under the LangChain integration folder.
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from psycopg import sql
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pydantic import BaseModel, PositiveInt


def transform_metadata_columns(
    columns: list[str] | list[tuple[str, str]] | str,
) -> list[tuple[str, str]]:
    """Normalize metadata column definitions to a list of (name, type) tuples.

    :param columns: A single column name (string), a list of column names
                    (strings), or a list of (name, type) tuples.
    :type columns: list[str] | list[tuple[str, str]] | str
    :return: A list of (column_name, column_type) tuples. Strings are mapped to
             "text", except a single-string input which maps to type "jsonb".
    :rtype: list[tuple[str, str]]
    """
    if isinstance(columns, str):
        return [(columns, "jsonb")]
    else:
        return [(col, "text") if isinstance(col, str) else col for col in columns]


class Table(BaseModel):
    """Table configuration for test parameterization.

    :param existing: Whether the table should be created before running a test.
    :param schema_name: Schema where the table resides.
    :param table_name: Name of the table.
    :param id_column: Primary key column name (uuid).
    :param content_column: Text content column name.
    :param embedding_column: Vector/embedding column name.
    :param embedding_type: Embedding type (e.g., "vector").
    :param embedding_dimension: Embedding dimension length.
    :param metadata_columns: List of metadata column names or (name, type) tuples.
    """

    existing: bool
    schema_name: str
    table_name: str
    id_column: str
    content_column: str
    embedding_column: str
    embedding_type: str
    embedding_dimension: PositiveInt
    metadata_columns: list[str] | list[tuple[str, str]] | str


@pytest.fixture(
    scope="class",
    params=[
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": "metadata_column",
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": [
                ("metadata_column1", "text"),
                ("metadata_column2", "double precision"),
            ],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": "metadata_column",
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": [
                ("metadata_column1", "text"),
                ("metadata_column2", "double precision"),
            ],
        },
    ],
    ids=[
        "non-existing-table-metadata-str",
        "non-existing-table-metadata-list",
        "non-existing-table-metadata-list-tuple",
        "existing-table-metadata-str",
        "existing-table-metadata-list",
        "existing-table-metadata-list-tuple",
    ],
)
async def async_table(
    async_connection_pool: AsyncConnectionPool,
    async_schema: str,
    request: pytest.FixtureRequest,
) -> AsyncGenerator[Table, Any]:
    """Fixture to provide a parametrized table configuration for asynchronous tests.

    This fixture yields a `Table` model with normalized metadata columns. When
    the parameter `existing` is `True`, it creates the table in the provided
    schema before yielding and drops it after the test class completes.

    :param async_connection_pool: The asynchronous connection pool to use for DDL.
    :type async_connection_pool: AsyncConnectionPool
    :param async_schema: The schema name where the table should be created.
    :type async_schema: str
    :param request: The pytest request object providing parametrization.
    :type request: pytest.FixtureRequest
    :return: An asynchronous generator yielding a `Table` configuration.
    :rtype: AsyncGenerator[Table, Any]
    """
    assert isinstance(request.param, dict), "Request param must be a dictionary"

    table = Table(
        existing=request.param.get("existing", None),
        schema_name=async_schema,
        table_name=request.param.get("table_name", "langchain"),
        id_column=request.param.get("id_column", "id_column"),
        content_column=request.param.get("content_column", "content_column"),
        embedding_column=request.param.get("embedding_column", "embedding_column"),
        embedding_type=request.param.get("embedding_type", "vector"),
        embedding_dimension=request.param.get("embedding_dimension", 1_536),
        metadata_columns=request.param.get("metadata_columns", "metadata_column"),
    )

    # Needed to make mypy happy during type checking
    metadata_columns = transform_metadata_columns(table.metadata_columns)
    table.metadata_columns = metadata_columns

    if table.existing:
        async with async_connection_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
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
                    table_name=sql.Identifier(async_schema, table.table_name),
                    id_column=sql.Identifier(table.id_column),
                    content_column=sql.Identifier(table.content_column),
                    embedding_column=sql.Identifier(table.embedding_column),
                    embedding_type=sql.Identifier(table.embedding_type),
                    embedding_dimension=sql.Literal(table.embedding_dimension),
                    metadata_columns=sql.SQL(", ").join(
                        sql.SQL("{col} {type}").format(
                            col=sql.Identifier(col), type=sql.Identifier(type)
                        )
                        for col, type in table.metadata_columns
                    ),
                )
            )

    yield table

    async with async_connection_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            sql.SQL("drop table {table} cascade").format(
                table=sql.Identifier(async_schema, table.table_name)
            )
        )


@pytest.fixture(
    scope="class",
    params=[
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": "metadata_column",
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": [
                ("metadata_column1", "text"),
                ("metadata_column2", "double precision"),
            ],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": "metadata_column",
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "embedding_type": "vector",
            "embedding_dimension": 1_536,
            "metadata_columns": [
                ("metadata_column1", "text"),
                ("metadata_column2", "double precision"),
            ],
        },
    ],
    ids=[
        "non-existing-table-metadata-str",
        "non-existing-table-metadata-list",
        "non-existing-table-metadata-list-tuple",
        "existing-table-metadata-str",
        "existing-table-metadata-list",
        "existing-table-metadata-list-tuple",
    ],
)
def table(
    connection_pool: ConnectionPool,
    schema: str,
    request: pytest.FixtureRequest,
) -> Generator[Table, Any, None]:
    """Fixture to provide a parametrized table configuration for synchronous tests.

    This fixture yields a `Table` model with normalized metadata columns. When
    the parameter `existing` is `True`, it creates the table in the provided
    schema before yielding and drops it after the test class completes.

    :param connection_pool: The synchronous connection pool to use for DDL.
    :type connection_pool: ConnectionPool
    :param schema: The schema name where the table should be created.
    :type schema: str
    :param request: The pytest request object providing parametrization.
    :type request: pytest.FixtureRequest
    :return: A generator yielding a `Table` configuration.
    :rtype: Generator[Table, Any, None]
    """
    assert isinstance(request.param, dict), "Request param must be a dictionary"

    table = Table(
        existing=request.param.get("existing", None),
        schema_name=schema,
        table_name=request.param.get("table_name", "langchain"),
        id_column=request.param.get("id_column", "id_column"),
        content_column=request.param.get("content_column", "content_column"),
        embedding_column=request.param.get("embedding_column", "embedding_column"),
        embedding_type=request.param.get("embedding_type", "vector"),
        embedding_dimension=request.param.get("embedding_dimension", 1_536),
        metadata_columns=request.param.get("metadata_columns", "metadata_column"),
    )

    # Needed to make mypy happy during type checking
    metadata_columns = transform_metadata_columns(table.metadata_columns)
    table.metadata_columns = metadata_columns

    if table.existing:
        with connection_pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
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
                    table_name=sql.Identifier(schema, table.table_name),
                    id_column=sql.Identifier(table.id_column),
                    content_column=sql.Identifier(table.content_column),
                    embedding_column=sql.Identifier(table.embedding_column),
                    embedding_type=sql.Identifier(table.embedding_type),
                    embedding_dimension=sql.Literal(table.embedding_dimension),
                    metadata_columns=sql.SQL(", ").join(
                        sql.SQL("{col} {type}").format(
                            col=sql.Identifier(col), type=sql.Identifier(type)
                        )
                        for col, type in table.metadata_columns
                    ),
                )
            )

    yield table

    with connection_pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("drop table {table} cascade").format(
                table=sql.Identifier(schema, table.table_name)
            )
        )
