"""Unit tests for the `use_binary_collation` option on the `custom_id` column.

These exercise the table definition produced by `_get_embedding_store` directly,
so they do not require a live SQL Server connection.
"""

from langchain_sqlserver.vectorstores import (
    CUSTOM_ID_COLLATION,
    SQLServerVectorStore,
)


def _make_store(use_binary_collation: bool) -> SQLServerVectorStore:
    """Build a store instance without running `__init__` (which connects to a DB).

    Only the attributes used by `_get_embedding_store` are set.
    """
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    store._embedding_length = 1536
    store._use_binary_collation = use_binary_collation
    return store


def test_custom_id_keeps_default_collation_by_default() -> None:
    store = _make_store(False)
    embedding_store = store._get_embedding_store("coll_default", None)
    custom_id = embedding_store.__table__.c.custom_id
    assert custom_id.type.collation is None


def test_custom_id_uses_binary_collation_when_opted_in() -> None:
    store = _make_store(True)
    embedding_store = store._get_embedding_store("bin_opt_in", None)
    custom_id = embedding_store.__table__.c.custom_id
    assert custom_id.type.collation == CUSTOM_ID_COLLATION
