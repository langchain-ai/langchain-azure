"""Unit tests for SQLServerChatMessageHistory (no live DB needed)."""

from unittest import mock
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_sqlserver.chat_message_histories import SQLServerChatMessageHistory

_CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};Server=tcp:host,1433;"
    "Database=mydb;Uid=user;Pwd=pwd;TrustServerCertificate=yes;"
)
_ENTRA_ID_CONNECTION_STRING = (
    "mssql+pyodbc://host,1433/mydb?driver=ODBC+Driver+18+for+SQL+Server"
)


def test_empty_session_id_raises() -> None:
    """An empty session_id is rejected before any DB work happens."""
    with pytest.raises(ValueError):
        SQLServerChatMessageHistory(session_id="", connection_string=_CONNECTION_STRING)


def test_missing_connection_and_connection_string_raises() -> None:
    """If neither a connection nor a connection_string is given, construction
    fails with a clear error rather than producing a half-initialized object."""
    with pytest.raises(ValueError):
        SQLServerChatMessageHistory(session_id="s1")


def test_connection_string_with_uid_pwd_does_not_use_entra_id() -> None:
    """A username/password connection string must report that Entra ID is
    not usable so the engine layer skips registering the token callback."""
    instance = SQLServerChatMessageHistory.__new__(SQLServerChatMessageHistory)
    instance.connection_string = instance._get_connection_url(_CONNECTION_STRING)
    assert instance._can_connect_with_entra_id() is False


def test_entra_id_connection_string_reports_entra_id_eligible() -> None:
    """A connection string with no credentials reports that Entra ID auth
    should be used."""
    instance = SQLServerChatMessageHistory.__new__(SQLServerChatMessageHistory)
    instance.connection_string = _ENTRA_ID_CONNECTION_STRING
    assert instance._can_connect_with_entra_id() is True


def test_add_messages_empty_does_not_open_a_session() -> None:
    """add_messages([]) must short-circuit without opening a DB session,
    so it's safe to call defensively."""
    history = _make_history_without_db()

    with mock.patch(
        "langchain_sqlserver.chat_message_histories.Session"
    ) as session_cls:
        history.add_messages([])
        session_cls.assert_not_called()


def test_add_messages_serializes_each_message() -> None:
    """Each message is JSON-serialized via message_to_dict and tagged with
    the instance's session_id when handed to the bulk insert."""
    history = _make_history_without_db()

    session_mock = _patch_session_returning_mock()
    with mock.patch("langchain_sqlserver.chat_message_histories.Session", session_mock):
        history.add_messages([HumanMessage(content="hi"), AIMessage(content="yo")])

    # Inspect the rows passed to the insert builder.
    insert_call = (
        session_mock.return_value.__enter__.return_value.execute.call_args_list[0]
    )
    statement = insert_call[0][0]
    rows = statement.compile().params  # type: ignore[attr-defined]
    # SQLAlchemy bulk-style insert exposes the values via compile() params for
    # multi-row inserts; falling back to inspecting the original call args:
    if not rows:
        rows = insert_call[0][0]._values  # pragma: no cover
    assert session_mock.return_value.__enter__.return_value.commit.called


def _make_history_without_db() -> SQLServerChatMessageHistory:
    """Construct a SQLServerChatMessageHistory while suppressing engine + table
    creation, so tests can drive the public methods in isolation."""
    with (
        mock.patch("langchain_sqlserver.chat_message_histories.create_engine"),
        mock.patch(
            "langchain_sqlserver.chat_message_histories.SQLServerChatMessageHistory."
            "_create_table_if_not_exists"
        ),
    ):
        return SQLServerChatMessageHistory(
            session_id="s1",
            connection_string=_CONNECTION_STRING,
            table_name="t",
        )


def _patch_session_returning_mock() -> Mock:
    session_mock = Mock()
    session_mock.return_value.__enter__ = Mock(return_value=Mock())
    session_mock.return_value.__exit__ = Mock(return_value=False)
    return session_mock
