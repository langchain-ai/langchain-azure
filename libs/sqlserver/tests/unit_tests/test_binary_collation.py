"""Unit tests for the `force_binary_collation` option on the `custom_id` column.

These exercise the table definition produced by `_get_embedding_store` directly,
so they do not require a live SQL Server connection.
"""

from langchain_sqlserver.vectorstores import (
    CUSTOM_ID_COLLATION,
    SQLServerVectorStore,
)


def _make_store(force_binary_collation: bool) -> SQLServerVectorStore:
    """Build a store instance without running `__init__` (which connects to a DB).

    Only the attributes used by `_get_embedding_store` are set.
    """
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    store._embedding_length = 1536
    store._force_binary_collation = force_binary_collation
    return store


def test_custom_id_uses_binary_collation_by_default() -> None:
    store = _make_store(True)
    embedding_store = store._get_embedding_store("bin_default", None)
    custom_id = embedding_store.__table__.c.custom_id
    assert custom_id.type.collation == CUSTOM_ID_COLLATION


def test_custom_id_keeps_default_collation_when_opted_out() -> None:
    store = _make_store(False)
    embedding_store = store._get_embedding_store("opt_out", None)
    custom_id = embedding_store.__table__.c.custom_id
    assert custom_id.type.collation is None
