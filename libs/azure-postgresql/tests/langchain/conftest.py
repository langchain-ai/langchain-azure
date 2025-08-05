from collections.abc import AsyncGenerator
from typing import Any

import pytest
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel


class Table(BaseModel):
    existing: bool
    schema_name: str
    table_name: str
    id_column: str
    content_column: str
    embedding_column: str
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
            "metadata_columns": "metadata_column",
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": False,
            "table_name": "non_existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
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
            "metadata_columns": "metadata_column",
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
            "metadata_columns": ["metadata_column1", "metadata_column2"],
        },
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id_column",
            "content_column": "content_column",
            "embedding_column": "embedding_column",
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
async def async_table_name(
    async_connection_pool: AsyncConnectionPool,
    async_schema_name: str,
    request: pytest.FixtureRequest,
) -> AsyncGenerator[Table, Any]:
    assert isinstance(request.param, dict), "Request param must be a dictionary"

    existing = request.param.get("existing", False)
    table_name = request.param.get("table_name", "langchain")
    id_column = request.param.get("id_column", "id")
    content_column = request.param.get("content_column", "content")
    embedding_column = request.param.get("embedding_column", "embedding")
    metadata_columns = request.param.get("metadata_columns", "metadata")

    table = Table(
        existing=existing,
        schema_name=async_schema_name,
        table_name=table_name,
        id_column=id_column,
        content_column=content_column,
        embedding_column=embedding_column,
        metadata_columns=metadata_columns,
    )

    if existing:
        pass

    yield table
