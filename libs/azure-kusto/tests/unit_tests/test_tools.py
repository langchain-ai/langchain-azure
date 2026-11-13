from __future__ import annotations

from unittest import mock

import pytest
from azure.kusto.data.exceptions import (
    KustoApiError,
    KustoClientError,
    KustoMultiApiError,
    KustoNetworkError,
    KustoServiceError,
)
from langchain_core.tools import BaseTool, ToolException

from langchain_azure_kusto import (
    InvalidKustoInputError,
    KustoDatabase,
    KustoListTablesTool,
    KustoQueryTool,
    KustoTableSchemaTool,
    KustoToolkit,
)
from langchain_azure_kusto.tools import _to_tool_exception


@pytest.fixture
def fake_db() -> mock.MagicMock:
    return mock.create_autospec(KustoDatabase, instance=True)


def test_toolkit_returns_tools_sharing_the_database(fake_db: mock.MagicMock) -> None:
    tools = KustoToolkit(db=fake_db).get_tools()
    assert [tool.name for tool in tools] == [
        "kusto_list_tables",
        "kusto_table_schema",
        "kusto_query",
    ]
    assert all(getattr(tool, "db") is fake_db for tool in tools)


def test_tools_call_the_database(fake_db: mock.MagicMock) -> None:
    fake_db.get_table_names.return_value = ["A", "B"]
    fake_db.get_table_info.return_value = "info"
    fake_db.run.return_value = "N\n1"

    assert KustoListTablesTool(db=fake_db).invoke({}) == "A, B"
    assert KustoTableSchemaTool(db=fake_db).invoke({"table_names": "A, B"}) == "info"
    fake_db.get_table_info.assert_called_once_with(["A", " B"])
    assert KustoQueryTool(db=fake_db).invoke({"query": "A | count"}) == "N\n1"
    fake_db.run.assert_called_once_with("A | count")


async def test_tools_call_the_database_async(fake_db: mock.MagicMock) -> None:
    fake_db.aget_table_names.return_value = ["A"]
    fake_db.aget_table_info.return_value = "info"
    fake_db.arun.return_value = "N\n1"

    assert await KustoListTablesTool(db=fake_db).ainvoke({}) == "A"
    schema_tool = KustoTableSchemaTool(db=fake_db)
    assert await schema_tool.ainvoke({"table_names": "A"}) == "info"
    query_tool = KustoQueryTool(db=fake_db)
    assert await query_tool.ainvoke({"query": "A | count"}) == "N\n1"
    fake_db.get_table_names.assert_not_called()
    fake_db.run.assert_not_called()


@pytest.mark.parametrize(
    ("tool_cls", "method", "tool_input"),
    [
        (KustoListTablesTool, "get_table_names", {}),
        (KustoTableSchemaTool, "get_table_info", {"table_names": "A"}),
        (KustoQueryTool, "run", {"query": "A"}),
    ],
)
def test_query_errors_are_returned_to_the_agent(
    fake_db: mock.MagicMock,
    tool_cls: type[BaseTool],
    method: str,
    tool_input: dict[str, str],
) -> None:
    getattr(fake_db, method).side_effect = KustoServiceError("Syntax error")
    assert tool_cls(db=fake_db).invoke(tool_input) == "Syntax error"  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("tool_cls", "method", "tool_input"),
    [
        (KustoListTablesTool, "aget_table_names", {}),
        (KustoTableSchemaTool, "aget_table_info", {"table_names": "A"}),
        (KustoQueryTool, "arun", {"query": "A"}),
    ],
)
async def test_query_errors_are_returned_to_the_agent_async(
    fake_db: mock.MagicMock,
    tool_cls: type[BaseTool],
    method: str,
    tool_input: dict[str, str],
) -> None:
    getattr(fake_db, method).side_effect = InvalidKustoInputError("Bad")
    assert await tool_cls(db=fake_db).ainvoke(tool_input) == "Bad"  # type: ignore[call-arg]


def test_network_errors_propagate(fake_db: mock.MagicMock) -> None:
    fake_db.run.side_effect = KustoNetworkError("https://x")
    with pytest.raises(KustoNetworkError):
        KustoQueryTool(db=fake_db).invoke({"query": "A"})


def test_error_mapping() -> None:
    api_error = KustoApiError({"error": {"code": "c", "message": "m", "@message": "d"}})
    assert str(_to_tool_exception(api_error)) == "d"

    multi = KustoMultiApiError(
        [
            {
                "OneApiErrors": [
                    {"error": {"code": "c", "message": "first", "@message": "one"}},
                    {"error": {"code": "c", "message": "second"}},
                ]
            }
        ]
    )
    assert str(_to_tool_exception(multi)) == "one\nsecond"

    long_error = _to_tool_exception(KustoServiceError("x" * 5000))
    assert isinstance(long_error, ToolException)
    assert len(str(long_error)) == 2000

    other = KustoClientError("client")
    assert _to_tool_exception(other) is other
