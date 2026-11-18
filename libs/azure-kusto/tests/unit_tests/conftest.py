"""Shared fakes for the unit tests. No test talks to a real cluster."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import pytest

from langchain_azure_kusto import KustoDatabase

CLUSTER = "https://mycluster.westus.kusto.windows.net"
DATABASE = "mydb"


@dataclass
class FakeColumn:
    column_name: str
    column_type: str = "string"


@dataclass
class FakeTable:
    columns: list[FakeColumn]
    raw_rows: list[list[Any]]


@dataclass
class FakeResponse:
    primary_results: list[FakeTable] = field(default_factory=list)


def table(columns: list[str], rows: list[list[Any]]) -> FakeTable:
    return FakeTable([FakeColumn(name) for name in columns], rows)


def response(*tables: FakeTable) -> FakeResponse:
    return FakeResponse(list(tables))


@pytest.fixture
def sync_client() -> Iterator[mock.MagicMock]:
    with mock.patch("langchain_azure_kusto.database.KustoClient") as client_cls:
        yield client_cls


@pytest.fixture
def async_client() -> Iterator[mock.MagicMock]:
    with mock.patch("langchain_azure_kusto.database.AsyncKustoClient") as client_cls:
        instance = client_cls.return_value
        instance.execute_query = mock.AsyncMock()
        instance.execute_mgmt = mock.AsyncMock()
        instance.close = mock.AsyncMock()
        yield client_cls


@pytest.fixture
def credential() -> mock.MagicMock:
    """A synchronous credential supplied by the caller."""
    return mock.MagicMock(spec=["get_token", "close"])


@pytest.fixture
def db(credential: mock.MagicMock) -> KustoDatabase:
    return KustoDatabase(CLUSTER, DATABASE, credential=credential, max_rows=2)
