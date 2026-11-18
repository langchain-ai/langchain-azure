"""LangChain standard unit tests for the Kusto tools."""

from __future__ import annotations

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_azure_kusto import (
    KustoDatabase,
    KustoListTablesTool,
    KustoQueryTool,
    KustoTableSchemaTool,
)

_DB = KustoDatabase("https://mycluster.westus.kusto.windows.net", "mydb")


class TestKustoListTablesToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[KustoListTablesTool]:
        return KustoListTablesTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"db": _DB}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {}


class TestKustoTableSchemaToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[KustoTableSchemaTool]:
        return KustoTableSchemaTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"db": _DB}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"table_names": "StormEvents"}


class TestKustoQueryToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[KustoQueryTool]:
        return KustoQueryTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"db": _DB}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "StormEvents | count"}
