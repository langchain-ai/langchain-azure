"""Unit tests for SQLServerSaver (no live DB required)."""

from unittest import mock

import pytest

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


def test_list_with_non_positive_limit_raises() -> None:
    """list() must reject limit=0 or negative values before building SQL."""
    saver = _make_saver()

    with mock.patch("langchain_sqlserver.checkpoint.Session"):
        with pytest.raises(ValueError):
            list(saver.list(None, limit=0))

        with pytest.raises(ValueError):
            list(saver.list(None, limit=-1))

    with mock.patch("langchain_sqlserver.checkpoint.Session"):
        with pytest.raises(ValueError):
            list(saver.list(None, filter={"bad key": "x"}))


def _make_saver(**kwargs: object) -> SQLServerSaver:
    """Build a saver while suppressing engine + table creation, so tests can
    drive the public methods in isolation."""
    with mock.patch("langchain_sqlserver.checkpoint.create_engine"):
        return SQLServerSaver(connection_string=_CONNECTION_STRING, **kwargs)  # type: ignore[arg-type]
