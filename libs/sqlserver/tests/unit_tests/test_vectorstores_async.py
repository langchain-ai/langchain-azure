"""Unit tests for the async surface on SQLServer_VectorStore."""

from typing import Any
from unittest import mock

import pytest

from langchain_sqlserver.vectorstores import SQLServer_VectorStore
from tests.utils.fake_embeddings import DeterministicFakeEmbedding

_CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};Server=tcp:host,1433;"
    "Database=mydb;Uid=user;Pwd=pwd;TrustServerCertificate=yes;"
)
EMBEDDING_LENGTH = 32


def _make_store() -> SQLServer_VectorStore:
    """Build a vector store while suppressing engine + table creation, so
    tests can drive the async public methods in isolation."""
    with (
        mock.patch("langchain_sqlserver.vectorstores.create_engine"),
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServer_VectorStore."
            "_prepare_json_data_type"
        ),
        mock.patch(
            "langchain_sqlserver.vectorstores.SQLServer_VectorStore."
            "_create_table_if_not_exists"
        ),
    ):
        return SQLServer_VectorStore(
            connection_string=_CONNECTION_STRING,
            embedding_function=DeterministicFakeEmbedding(size=EMBEDDING_LENGTH),
            embedding_length=EMBEDDING_LENGTH,
        )


def test_get_async_engine_rewrites_driver_and_caches() -> None:
    """The async engine uses ``mssql+aioodbc`` (not ``mssql+pyodbc``) and is
    cached for the lifetime of the store, so repeated calls don't rebuild
    the engine on every request."""
    store = _make_store()
    captured: dict[str, Any] = {}

    def fake_create(url: str, **_: Any) -> object:
        captured.setdefault("urls", []).append(url)
        return mock.Mock(name="AsyncEngine")

    with mock.patch(
        "langchain_sqlserver.vectorstores.create_async_engine", side_effect=fake_create
    ) as create_async:
        engine_1 = store._get_async_engine()
        engine_2 = store._get_async_engine()

    assert engine_1 is engine_2
    create_async.assert_called_once()
    assert captured["urls"][0].startswith("mssql+aioodbc://")
    assert "mssql+pyodbc" not in captured["urls"][0]


@pytest.mark.asyncio
async def test_aadd_texts_with_none_short_circuits() -> None:
    """Mirrors the sync ``add_texts`` contract: ``None`` input returns ``[]``
    without touching the database, so callers can pass through filtered
    iterables defensively."""
    store = _make_store()
    with mock.patch.object(store, "_get_async_engine") as engine:
        result = await store.aadd_texts(None)  # type: ignore[arg-type]
        engine.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_adelete_empty_list_returns_false_without_engine() -> None:
    """``adelete([])`` is a no-op that returns ``False`` and never opens the
    async engine, matching the sync behavior of :meth:`delete`."""
    store = _make_store()
    with mock.patch.object(store, "_get_async_engine") as engine:
        result = await store.adelete([])
    assert result is False
    engine.assert_not_called()


@pytest.mark.asyncio
async def test_aget_by_ids_empty_returns_empty_list() -> None:
    """``aget_by_ids([])`` short-circuits without contacting the engine."""
    store = _make_store()
    with mock.patch.object(store, "_get_async_engine") as engine:
        result = await store.aget_by_ids([])
    assert result == []
    engine.assert_not_called()
