"""Integration tests for SQLServerChatMessageHistory."""

import os
import uuid
from typing import Generator

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import create_engine, text

from langchain_sqlserver import SQLServerChatMessageHistory

_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_TRUSTED_CONNECTION"))
_PYODBC_CONNECTION_STRING = str(os.environ.get("TEST_PYODBC_CONNECTION_STRING"))
_TABLE_NAME = "langchain_chat_history_tests"


@pytest.fixture
def history() -> Generator[SQLServerChatMessageHistory, None, None]:
    """Provide a fresh per-test chat history bound to a unique session_id.

    The table is shared across tests, but each test uses a unique session_id
    so they remain isolated. The session is cleared and the table dropped on
    teardown.
    """
    session_id = f"test-session-{uuid.uuid4()}"
    history = SQLServerChatMessageHistory(
        session_id=session_id,
        connection_string=_CONNECTION_STRING,
        table_name=_TABLE_NAME,
    )
    yield history

    # Best-effort cleanup: clear messages for this session, then drop the
    # underlying table if no other session still has rows in it.
    history.clear()
    try:
        conn = create_engine(_PYODBC_CONNECTION_STRING).connect()
        conn.execute(text(f"drop table if exists {_TABLE_NAME}"))
        conn.commit()
        conn.close()
    except Exception:
        pass


def test_add_and_retrieve_messages(history: SQLServerChatMessageHistory) -> None:
    """add_messages persists messages and `messages` returns them in order."""
    history.add_messages(
        [
            SystemMessage(content="you are helpful"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
    )

    result = history.messages
    assert [type(m).__name__ for m in result] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert [m.content for m in result] == ["you are helpful", "hi", "hello"]


def test_add_user_and_ai_message_helpers(
    history: SQLServerChatMessageHistory,
) -> None:
    """The inherited add_user_message / add_ai_message helpers persist."""
    history.add_user_message("ping")
    history.add_ai_message("pong")

    result = history.messages
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage) and result[0].content == "ping"
    assert isinstance(result[1], AIMessage) and result[1].content == "pong"


def test_clear_removes_only_current_session(
    history: SQLServerChatMessageHistory,
) -> None:
    """`clear` deletes only the current session's rows, leaving others intact."""
    history.add_user_message("session A msg")

    other = SQLServerChatMessageHistory(
        session_id=f"test-session-{uuid.uuid4()}",
        connection_string=_CONNECTION_STRING,
        table_name=_TABLE_NAME,
    )
    other.add_user_message("session B msg")

    history.clear()
    assert history.messages == []
    assert [m.content for m in other.messages] == ["session B msg"]

    other.clear()


def test_messages_isolated_by_session_id(
    history: SQLServerChatMessageHistory,
) -> None:
    """Two instances pointing at the same table but different session_ids
    must not see each other's messages."""
    history.add_user_message("from session 1")

    other = SQLServerChatMessageHistory(
        session_id=f"test-session-{uuid.uuid4()}",
        connection_string=_CONNECTION_STRING,
        table_name=_TABLE_NAME,
    )
    other.add_user_message("from session 2")

    assert [m.content for m in history.messages] == ["from session 1"]
    assert [m.content for m in other.messages] == ["from session 2"]

    other.clear()


def test_add_messages_empty_is_a_noop(
    history: SQLServerChatMessageHistory,
) -> None:
    """Passing an empty sequence to add_messages should not raise or insert."""
    history.add_messages([])
    assert history.messages == []


def test_empty_session_id_is_rejected() -> None:
    """An empty session_id is rejected at construction time."""
    with pytest.raises(ValueError):
        SQLServerChatMessageHistory(
            session_id="",
            connection_string=_CONNECTION_STRING,
            table_name=_TABLE_NAME,
        )
