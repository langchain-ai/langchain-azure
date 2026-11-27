from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, cast
from unittest import mock

import pytest

from langchain_azure_kusto import InvalidKustoInputError, KustoDatabase
from langchain_azure_kusto.database import (
    _BorrowedAsyncCredential,
    _BorrowedCredential,
)
from tests.unit_tests.conftest import CLUSTER, DATABASE, response, table


class _AsyncCredential:
    def __init__(self) -> None:
        self.closed = False

    async def get_token(self, *scopes: str, **kwargs: Any) -> str:
        return "token"

    async def close(self) -> None:
        self.closed = True


def _async_credential() -> Any:
    return _AsyncCredential()


def _call(client_cls: mock.MagicMock, method: str = "execute_query") -> Any:
    return getattr(client_cls.return_value, method).call_args


def _options(client_cls: mock.MagicMock, method: str = "execute_query") -> Any:
    properties = _call(client_cls, method).args[2]
    return properties._options, properties.client_request_id


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cluster_uri": "http://x.kusto.windows.net"}, "https://"),
        ({"database": ""}, "database"),
        ({"max_rows": 0}, "max_rows"),
        ({"sample_rows_in_table_info": -1}, "sample_rows"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {"cluster_uri": CLUSTER, "database": DATABASE}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        KustoDatabase(**arguments)


def test_properties_and_repr() -> None:
    db = KustoDatabase(CLUSTER + "/", DATABASE, max_rows=7)
    assert db.cluster_uri == CLUSTER
    assert db.database == DATABASE
    assert db.max_rows == 7
    assert repr(db) == f"KustoDatabase(cluster_uri='{CLUSTER}', database='{DATABASE}')"


def test_run_is_read_only_and_capped(
    db: KustoDatabase, sync_client: mock.MagicMock
) -> None:
    sync_client.return_value.execute_query.return_value = response(
        table(["Name", "Count"], [["a", 1], ["b", 2]])
    )

    assert db.run("T | take 5") == "Name,Count\na,1\nb,2"

    call = _call(sync_client)
    assert call.args[:2] == (DATABASE, "T | take 5")
    options, request_id = _options(sync_client)
    assert options["request_readonly"] is True
    assert options["query_take_max_records"] == 3
    assert options["servertimeout"] == timedelta(minutes=1)
    assert request_id.startswith("LangChain;")


def test_run_reports_truncation(db: KustoDatabase, sync_client: mock.MagicMock) -> None:
    sync_client.return_value.execute_query.return_value = response(
        table(["N"], [[1], [2], [3]])
    )
    result = db.run("T")
    assert result.startswith("N\n1\n2\n(Showing the first 2 rows only.")


def test_run_formats_values_and_multiple_results(
    db: KustoDatabase, sync_client: mock.MagicMock
) -> None:
    sync_client.return_value.execute_query.return_value = response(
        table(["D", "B", "Null", "S"], [[{"k": "v"}, True, None, "a,b"]]),
        table(["Empty"], []),
    )
    assert db.run("T; U") == (
        'Result 1:\nD,B,Null,S\n"{""k"":""v""}",true,,"a,b"\n\n'
        "Result 2:\nEmpty\n(no rows)"
    )


@pytest.mark.parametrize(
    "query",
    [
        ".drop table T",
        "  \n.set-or-append T <| print 1",
        "// comment\n.show tables",
        "// a\n  // b\n\t.alter table T",
    ],
)
def test_run_rejects_management_commands(
    db: KustoDatabase, sync_client: mock.MagicMock, query: str
) -> None:
    with pytest.raises(InvalidKustoInputError, match="Management commands"):
        db.run(query)
    sync_client.assert_not_called()


def test_run_allows_dots_inside_queries(
    db: KustoDatabase, sync_client: mock.MagicMock
) -> None:
    sync_client.return_value.execute_query.return_value = response(table(["x"], []))
    db.run("// .drop is just a comment here\nT | where x == '.'")
    sync_client.return_value.execute_query.assert_called_once()


def test_get_table_names(db: KustoDatabase, sync_client: mock.MagicMock) -> None:
    sync_client.return_value.execute_mgmt.return_value = response(
        table(["TableName"], [["A"], ["B"]])
    )
    assert db.get_table_names() == ["A", "B"]
    call = _call(sync_client, "execute_mgmt")
    assert call.args[:2] == (DATABASE, ".show tables | project TableName")
    options, _ = _options(sync_client, "execute_mgmt")
    assert "request_readonly" not in options
    assert "query_take_max_records" not in options


def test_get_table_info(db: KustoDatabase, sync_client: mock.MagicMock) -> None:
    sync_client.return_value.execute_query.return_value = response(
        table(["ColumnName", "ColumnType"], [["Name", "string"], ["N", "long"]]),
        table(["Name", "N"], [["a", 1]]),
        table(["ColumnName", "ColumnType"], [["X", "real"]]),
        table(["X"], []),
    )

    info = db.get_table_info([" Storm Events ", "", "other.table-1"])

    assert info == (
        "Table: Storm Events\nColumns:\n- Name: string\n- N: long\n"
        "Sample rows:\nName,N\na,1\n\n"
        "Table: other.table-1\nColumns:\n- X: real\nSample rows:\nX\n(no rows)"
    )
    assert _call(sync_client).args[1] == (
        "['Storm Events'] | getschema | project ColumnName, ColumnType;\n"
        "['Storm Events'] | take 3;\n"
        "['other.table-1'] | getschema | project ColumnName, ColumnType;\n"
        "['other.table-1'] | take 3"
    )
    options, _ = _options(sync_client)
    assert "request_readonly" not in options


def test_get_table_info_without_samples(sync_client: mock.MagicMock) -> None:
    db = KustoDatabase(CLUSTER, DATABASE, sample_rows_in_table_info=0)
    sync_client.return_value.execute_query.return_value = response(
        table(["ColumnName", "ColumnType"], [["Name", "string"]])
    )
    with mock.patch("azure.identity.DefaultAzureCredential"):
        info = db.get_table_info(["T"])
    assert info == "Table: T\nColumns:\n- Name: string"
    assert _call(sync_client).args[1] == (
        "['T'] | getschema | project ColumnName, ColumnType"
    )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([" ", ""], "at least one"),
        (["T'] | take 1; ['U"], "Invalid table name"),
        (["T\\"], "Invalid table name"),
        (["x" * 1025], "Invalid table name"),
    ],
)
def test_get_table_info_rejects_bad_names(
    db: KustoDatabase, sync_client: mock.MagicMock, names: list[str], message: str
) -> None:
    with pytest.raises(InvalidKustoInputError, match=message):
        db.get_table_info(names)
    sync_client.assert_not_called()


def test_sync_client_is_reused_stamped_and_borrows_credential(
    db: KustoDatabase, sync_client: mock.MagicMock, credential: mock.MagicMock
) -> None:
    sync_client.return_value.execute_query.return_value = response(table(["x"], []))
    db.run("T")
    db.run("T")

    sync_client.assert_called_once()
    kcsb = sync_client.call_args.args[0]
    assert kcsb.application_for_tracing.startswith("Kusto.LangChain:")
    assert isinstance(kcsb.azure_credential, _BorrowedCredential)

    kcsb.azure_credential.get_token("scope")
    credential.get_token.assert_called_once_with("scope")
    kcsb.azure_credential.close()
    credential.close.assert_not_called()


def test_sync_default_credential(sync_client: mock.MagicMock) -> None:
    db = KustoDatabase(CLUSTER, DATABASE)
    sync_client.return_value.execute_query.return_value = response(table(["x"], []))
    with mock.patch("azure.identity.DefaultAzureCredential") as default:
        db.run("T")
    assert sync_client.call_args.args[0].azure_credential is default.return_value


def test_sync_method_rejects_async_credential(sync_client: mock.MagicMock) -> None:
    db = KustoDatabase(CLUSTER, DATABASE, credential=_async_credential())
    with pytest.raises(ValueError, match="AsyncTokenCredential"):
        db.run("T")
    sync_client.assert_not_called()


def test_close_and_context_manager(
    db: KustoDatabase, sync_client: mock.MagicMock
) -> None:
    db.close()  # nothing to close yet
    sync_client.return_value.execute_query.return_value = response(table(["x"], []))
    with db as entered:
        assert entered is db
        db.run("T")
    sync_client.return_value.close.assert_called_once()
    db.close()
    sync_client.return_value.close.assert_called_once()


async def test_async_methods(
    db: KustoDatabase, async_client: mock.MagicMock, sync_client: mock.MagicMock
) -> None:
    client = async_client.return_value
    client.execute_query.return_value = response(table(["N"], [[1]]))
    assert await db.arun("T") == "N\n1"
    options, _ = _options(async_client)
    assert options["request_readonly"] is True

    client.execute_mgmt.return_value = response(table(["TableName"], [["A"]]))
    assert await db.aget_table_names() == ["A"]

    client.execute_query.return_value = response(
        table(["ColumnName", "ColumnType"], [["N", "long"]]),
        table(["N"], [[1]]),
    )
    assert await db.aget_table_info(["T"]) == (
        "Table: T\nColumns:\n- N: long\nSample rows:\nN\n1"
    )
    async_client.assert_called_once()
    sync_client.assert_not_called()

    with pytest.raises(InvalidKustoInputError):
        await db.arun(".show tables")
    with pytest.raises(InvalidKustoInputError):
        await db.aget_table_info(["bad'name"])


async def test_async_borrows_sync_credential(
    db: KustoDatabase, async_client: mock.MagicMock
) -> None:
    async_client.return_value.execute_query.return_value = response(table(["x"], []))
    await db.arun("T")
    kcsb = async_client.call_args.args[0]
    assert isinstance(kcsb.azure_credential, _BorrowedCredential)


async def test_async_borrows_async_credential(async_client: mock.MagicMock) -> None:
    credential = _AsyncCredential()
    db = KustoDatabase(CLUSTER, DATABASE, credential=cast(Any, credential))
    async_client.return_value.execute_query.return_value = response(table(["x"], []))
    await db.arun("T")

    borrowed = async_client.call_args.args[0].azure_credential
    assert isinstance(borrowed, _BorrowedAsyncCredential)
    assert await borrowed.get_token("scope") == "token"
    await borrowed.close()
    assert not credential.closed


async def test_async_default_credential(async_client: mock.MagicMock) -> None:
    db = KustoDatabase(CLUSTER, DATABASE)
    async_client.return_value.execute_query.return_value = response(table(["x"], []))
    with mock.patch("azure.identity.aio.DefaultAzureCredential") as default:
        await db.arun("T")
    assert async_client.call_args.args[0].azure_credential is default.return_value


async def test_aclose_and_async_context_manager(
    db: KustoDatabase, async_client: mock.MagicMock, sync_client: mock.MagicMock
) -> None:
    await db.aclose()  # nothing to close yet
    async_client.return_value.execute_query.return_value = response(table(["x"], []))
    sync_client.return_value.execute_query.return_value = response(table(["x"], []))
    async with db as entered:
        assert entered is db
        await db.arun("T")
        db.run("T")
    async_client.return_value.close.assert_awaited_once()
    sync_client.return_value.close.assert_called_once()


def test_new_event_loop_gets_new_async_client(
    db: KustoDatabase, async_client: mock.MagicMock
) -> None:
    first, second, third = (mock.MagicMock() for _ in range(3))
    for client in (first, second, third):
        client.execute_query = mock.AsyncMock(return_value=response(table(["x"], [])))
        client.close = mock.AsyncMock()
    second.close.side_effect = RuntimeError("Event loop is closed")
    async_client.side_effect = [first, second, third]

    asyncio.run(db.arun("T"))
    asyncio.run(db.arun("T"))
    first.close.assert_awaited_once()
    asyncio.run(db.arun("T"))  # a failed close does not stop the query
    second.close.assert_awaited_once()
    assert async_client.call_count == 3


async def test_same_event_loop_reuses_async_client(
    db: KustoDatabase, async_client: mock.MagicMock
) -> None:
    async_client.return_value.execute_query.return_value = response(table(["x"], []))
    await db.arun("T")
    await db.arun("T")
    async_client.assert_called_once()
