"""A database error must not read as "no such documents" or "deleted"."""

from unittest import mock

import pytest
from sqlalchemy.exc import DBAPIError

from langchain_sqlserver.vectorstores import SQLServerVectorStore
from tests.utils.fake_embeddings import DeterministicFakeEmbedding

_CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};Server=tcp:host,1433;"
    "Database=mydb;Uid=user;Pwd=pwd;TrustServerCertificate=yes;"
)
EMBEDDING_LENGTH = 32


def _db_error() -> DBAPIError:
    return DBAPIError("SELECT ...", {}, Exception("Communication link failure"))


def _make_store() -> SQLServerVectorStore:
    with (
        mock.patch("langchain_sqlserver.vectorstores.create_engine"),
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServerVectorStore."
            "_prepare_json_data_type"
        ),
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServerVectorStore."
            "_create_table_if_not_exists"
        ),
    ):
        return SQLServerVectorStore(
            connection_string=_CONNECTION_STRING,
            embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
            embedding_length=EMBEDDING_LENGTH,
        )


def _patch_session(session: mock.MagicMock) -> mock._patch:
    factory = mock.MagicMock()
    factory.return_value.__enter__.return_value = session
    return mock.patch("langchain_sqlserver.vectorstores.Session", factory)


def _patch_async_session(session: mock.MagicMock) -> mock._patch:
    factory = mock.MagicMock()
    factory.return_value.__aenter__.return_value = session
    return mock.patch("langchain_sqlserver.vectorstores.AsyncSession", factory)


def test_get_by_ids_raises_when_query_fails() -> None:
    store = _make_store()
    session = mock.MagicMock()
    session.execute.side_effect = _db_error()

    with _patch_session(session), pytest.raises(DBAPIError):
        store.get_by_ids(["1", "2"])


def test_get_by_ids_with_no_matches_is_still_empty() -> None:
    store = _make_store()
    session = mock.MagicMock()
    session.execute.return_value.fetchall.return_value = []

    with _patch_session(session):
        assert store.get_by_ids(["1", "2"]) == []


@pytest.mark.asyncio
async def test_aget_by_ids_raises_when_query_fails() -> None:
    store = _make_store()
    session = mock.MagicMock()
    session.execute = mock.AsyncMock(side_effect=_db_error())

    with (
        mock.patch.object(store, "_aensure_table_exists", mock.AsyncMock()),
        mock.patch.object(store, "_get_async_engine"),
        _patch_async_session(session),
        pytest.raises(DBAPIError),
    ):
        await store.aget_by_ids(["1", "2"])


def test_delete_returns_false_when_commit_fails() -> None:
    store = _make_store()
    session = mock.MagicMock()
    session.query.return_value.filter.return_value.delete.return_value = 2
    session.commit.side_effect = _db_error()

    with _patch_session(session):
        assert store.delete(["1", "2"]) is False


def test_delete_returns_false_when_query_fails() -> None:
    store = _make_store()
    session = mock.MagicMock()
    session.query.return_value.filter.return_value.delete.side_effect = _db_error()

    with _patch_session(session):
        assert store.delete(["1", "2"]) is False


def test_delete_that_removes_rows_is_still_true() -> None:
    store = _make_store()
    session = mock.MagicMock()
    session.query.return_value.filter.return_value.delete.return_value = 2

    with _patch_session(session):
        assert store.delete(["1", "2"]) is True
