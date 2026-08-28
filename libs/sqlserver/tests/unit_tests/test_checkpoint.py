"""Unit tests for SQLServerSaver (no live DB required)."""

from unittest import mock

import pytest
from sqlalchemy import Integer, LargeBinary
from sqlalchemy.dialects import mssql

from langchain_sqlserver.checkpoint import SQLServerSaver

_CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};Server=tcp:host,1433;"
    "Database=mydb;Uid=user;Pwd=pwd;TrustServerCertificate=yes;"
)
_ENTRA_ID_CONNECTION_STRING = (
    "mssql+pyodbc://host,1433/mydb?driver=ODBC+Driver+18+for+SQL+Server"
)


def test_missing_connection_and_connection_string_raises() -> None:
    """If neither a connection nor a connection_string is given, construction
    fails with a clear error rather than producing a half-initialized object."""
    with pytest.raises(ValueError):
        SQLServerSaver()


def test_qualified_table_names_respect_schema() -> None:
    """When ``db_schema`` is provided, the saver should bracket the schema
    and the table name in the SQL it builds — never leak unquoted names."""
    saver = _make_saver(db_schema="lc")
    assert saver._cps_t == "[lc].[checkpoints]"
    assert saver._wrt_t == "[lc].[checkpoint_writes]"


def test_qualified_table_names_without_schema() -> None:
    """Without a schema, only the table name is bracketed."""
    saver = _make_saver()
    assert saver._cps_t == "[checkpoints]"
    assert saver._wrt_t == "[checkpoint_writes]"


def test_custom_table_names_are_used() -> None:
    """``checkpoints_table`` / ``writes_table`` overrides flow through to the
    fully-qualified identifiers used in SQL."""
    saver = _make_saver(checkpoints_table="cps", writes_table="wrt")
    assert saver._cps_t == "[cps]"
    assert saver._wrt_t == "[wrt]"


def test_uid_pwd_connection_string_does_not_use_entra_id() -> None:
    """A username/password connection string must report that Entra ID is
    not usable so the engine layer skips registering the token callback."""
    instance = SQLServerSaver.__new__(SQLServerSaver)
    instance.connection_string = instance._get_connection_url(_CONNECTION_STRING)
    assert instance._can_connect_with_entra_id() is False


def test_credentialless_connection_string_uses_entra_id() -> None:
    """A connection string with no credentials reports that Entra ID auth
    should be used."""
    instance = SQLServerSaver.__new__(SQLServerSaver)
    instance.connection_string = _ENTRA_ID_CONNECTION_STRING
    assert instance._can_connect_with_entra_id() is True


def test_put_writes_empty_is_a_noop() -> None:
    """put_writes([]) must short-circuit without opening a DB session, so it
    is safe to call defensively from the graph runtime."""
    saver = _make_saver()

    cfg = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "x",
        }
    }
    with mock.patch("langchain_sqlserver.checkpoint.Session") as session_cls:
        saver.put_writes(cfg, [], task_id="t")  # type: ignore[arg-type]
        session_cls.assert_not_called()


def test_list_with_invalid_metadata_filter_key_raises() -> None:
    """The metadata filter is interpolated into the SQL JSON_VALUE path; an
    invalid identifier must be rejected up-front to avoid SQL injection."""
    saver = _make_saver()

    with mock.patch("langchain_sqlserver.checkpoint.Session"):
        with pytest.raises(ValueError):
            list(saver.list(None, filter={"bad key": "x"}))


def test_writes_idx_column_is_integer() -> None:
    """`idx` must be an integer column so ORDER BY idx sorts numerically;
    some writes use negative indices (WRITES_IDX_MAP) and text ordering would
    place '-1'/'10' before '2'."""
    saver = _make_saver()
    idx_col = saver._writes_model.__table__.c.idx
    assert isinstance(idx_col.type, Integer)


def test_checkpoint_metadata_column_is_json_text_not_binary() -> None:
    """`metadata` must be NVARCHAR text (not binary) so JSON_VALUE filtering
    works; casting binary to NVARCHAR does not round-trip UTF-8 JSON."""
    saver = _make_saver()
    meta_col = saver._checkpoints_model.__table__.c.metadata
    assert not isinstance(meta_col.type, LargeBinary)
    rendered = meta_col.type.compile(dialect=mssql.dialect()).upper()
    assert "NVARCHAR" in rendered


def test_put_writes_defaults_missing_checkpoint_ns_and_binds_int_idx() -> None:
    """put_writes must not KeyError when `checkpoint_ns` is omitted (other
    methods default it to ""), and must bind `idx` as an integer."""
    saver = _make_saver()
    saver.is_setup = True  # skip DDL; we only care about the bound params
    cfg = {"configurable": {"thread_id": "1", "checkpoint_id": "c"}}  # no ns

    with mock.patch("langchain_sqlserver.checkpoint.Session") as session_cls:
        session = session_cls.return_value
        saver.put_writes(cfg, [("mychannel", "value")], task_id="t")  # type: ignore[arg-type]
        assert session.execute.call_args is not None
        params = session.execute.call_args.args[1]
        assert params["ns"] == ""
        assert isinstance(params["idx"], int)


def test_list_non_positive_limit_omits_top_clause() -> None:
    """A non-positive `limit` must not emit `TOP` (TOP with a negative value
    is invalid T-SQL)."""
    saver = _make_saver()
    saver.is_setup = True

    with mock.patch("langchain_sqlserver.checkpoint.Session") as session_cls:
        session = session_cls.return_value
        session.execute.return_value.fetchall.return_value = []
        list(saver.list(None, limit=0))
        query = session.execute.call_args.args[0].text
        assert "TOP" not in query.upper()


def test_list_positive_limit_emits_top_clause() -> None:
    """A positive `limit` must emit a `TOP N` clause."""
    saver = _make_saver()
    saver.is_setup = True

    with mock.patch("langchain_sqlserver.checkpoint.Session") as session_cls:
        session = session_cls.return_value
        session.execute.return_value.fetchall.return_value = []
        list(saver.list(None, limit=5))
        query = session.execute.call_args.args[0].text
        assert "TOP 5" in query.upper()


def _make_saver(**kwargs: object) -> SQLServerSaver:
    """Build a saver while suppressing engine + table creation, so tests can
    drive the public methods in isolation."""
    with mock.patch("langchain_sqlserver.checkpoint.create_engine"):
        return SQLServerSaver(connection_string=_CONNECTION_STRING, **kwargs)  # type: ignore[arg-type]
