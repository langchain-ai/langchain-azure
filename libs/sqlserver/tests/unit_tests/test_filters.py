"""Unit tests for the metadata filter builder (no live DB required).

These compile the SQLAlchemy expression produced by ``_handle_field_filter``
so operators can be checked without connecting to SQL Server.
"""

from typing import cast

from sqlalchemy import ColumnElement
from sqlalchemy.dialects import mssql

from langchain_sqlserver.vectorstores import SQLServerVectorStore


def _make_store() -> SQLServerVectorStore:
    """Build a store without running ``__init__`` (which connects to a DB).

    Only the attributes used by ``_handle_field_filter`` are set.
    """
    store = SQLServerVectorStore.__new__(SQLServerVectorStore)
    store._embedding_length = 4
    store._use_binary_collation = False
    store._embedding_store = store._get_embedding_store("t", None)
    return store


def _compile(store: SQLServerVectorStore, field: str, value: object) -> str:
    clause = cast(ColumnElement, store._handle_field_filter(field, value))
    return str(
        clause.compile(dialect=mssql.dialect(), compile_kwargs={"literal_binds": True})
    )


def test_nin_operator_builds_not_in_clause() -> None:
    """`$nin` must produce a NOT IN clause. It previously called a
    non-existent `nin_` method and raised AttributeError at query time."""
    sql = _compile(_make_store(), "color", {"$nin": ["red", "blue"]})
    assert "NOT IN" in sql.upper()
    assert "'red'" in sql and "'blue'" in sql


def test_in_operator_builds_in_clause() -> None:
    sql = _compile(_make_store(), "color", {"$in": ["red", "blue"]})
    assert " IN " in sql.upper()
    assert "NOT IN" not in sql.upper()


def test_like_operator_builds_like_clause() -> None:
    sql = _compile(_make_store(), "name", {"$like": "%foo%"})
    assert "LIKE" in sql.upper()
    assert "'%foo%'" in sql
