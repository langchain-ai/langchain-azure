"""Unit tests for id handling in ``add_texts`` (no live DB).

``add_texts`` / ``add_documents`` must return ``List[str]`` per the
``VectorStore`` contract. When ids are auto-generated (or taken from metadata),
the returned values must be strings so they match the stringified ``custom_id``
persisted in the table and round-trip through ``get_by_ids`` / ``delete``.
"""

from typing import Any, List

import pytest

from langchain_sqlserver import vectorstores
from langchain_sqlserver.vectorstores import SQLServerVectorStore


class _FakeEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _FakeSession:
    """No-op stand-in for ``sqlalchemy.orm.Session`` used by ``add_texts``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def commit(self) -> None:
        return None


def _make_store(monkeypatch: pytest.MonkeyPatch) -> SQLServerVectorStore:
    monkeypatch.setattr(vectorstores, "Session", _FakeSession)
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    store._batch_size = 100
    store._embedding_length = 4
    store._use_binary_collation = False
    store.embedding_function = _FakeEmbeddings()  # type: ignore[assignment]
    store._bind = object()  # type: ignore[assignment]
    store._embedding_store = store._get_embedding_store("add_texts_ids_test", None)
    return store


def test_add_texts_returns_str_ids_when_generated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(monkeypatch)
    ids = store.add_texts(["alpha", "beta"])
    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)


def test_add_texts_stringifies_ids_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(monkeypatch)
    ids = store.add_texts(["alpha"], metadatas=[{"id": 123}])
    assert ids == ["123"]
    assert all(isinstance(i, str) for i in ids)
