"""Live tests against a real Kusto database.

Set ``KUSTO_CLUSTER_URI`` and ``KUSTO_DATABASE`` (see ``.env.example``) and run
``az login``. Without them every test here is skipped.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_azure_kusto import (
    KustoDatabase,
    KustoLoader,
    KustoQueryTool,
    KustoToolkit,
)

# importorskip, not a plain import: `python-dotenv` only comes with the
# `test_integration` group.
dotenv = pytest.importorskip("dotenv")
dotenv.load_dotenv()

CLUSTER_URI = os.environ.get("KUSTO_CLUSTER_URI", "")
DATABASE = os.environ.get("KUSTO_DATABASE", "")

pytestmark = pytest.mark.skipif(
    not (CLUSTER_URI and DATABASE),
    reason="KUSTO_CLUSTER_URI and KUSTO_DATABASE are not set",
)


@pytest.fixture(scope="module")
def db() -> Iterator[KustoDatabase]:
    with KustoDatabase(CLUSTER_URI, DATABASE, max_rows=5) as database:
        yield database


@pytest.fixture(scope="module")
def table_name(db: KustoDatabase) -> str:
    names = db.get_table_names()
    assert names, "KUSTO_DATABASE must contain at least one table"
    return names[0]


def test_toolkit_end_to_end(db: KustoDatabase, table_name: str) -> None:
    list_tables, schema, query = KustoToolkit(db=db).get_tools()
    assert table_name in list_tables.invoke({}).split(", ")
    assert f"Table: {table_name}" in schema.invoke({"table_names": table_name})
    assert query.invoke({"query": f"['{table_name}'] | count"}).startswith("Count\n")


def test_query_errors_are_returned(db: KustoDatabase) -> None:
    tool = KustoQueryTool(db=db)
    assert tool.invoke({"query": "ThisTableDoesNotExist_x | take 1"})


async def test_async_end_to_end(table_name: str) -> None:
    async with KustoDatabase(CLUSTER_URI, DATABASE, max_rows=5) as database:
        assert table_name in await database.aget_table_names()
        result = await database.arun(f"['{table_name}'] | take 2")
        assert result.count("\n") >= 1


def test_loader(db: KustoDatabase, table_name: str) -> None:
    docs = KustoLoader(f"['{table_name}'] | take 2", db).load()
    assert len(docs) <= 2


class TestKustoQueryToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[KustoQueryTool]:
        return KustoQueryTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"db": KustoDatabase(CLUSTER_URI or "https://unset", DATABASE or "x")}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "print x = 1"}
