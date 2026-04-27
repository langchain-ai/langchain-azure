"""Unit tests for CosmosDBChatMessageHistory."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from azure.core import MatchConditions


def test_missing_credential_and_connection_string_raises() -> None:
    with patch("langchain_azure_cosmosdb._chat_history.CosmosClient"):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        with pytest.raises(
            ValueError,
            match="Either a connection string or a credential must be set",
        ):
            CosmosDBChatMessageHistory(
                cosmos_endpoint="https://fake.documents.azure.com:443/",
                cosmos_database="testdb",
                cosmos_container="testcontainer",
                session_id="s1",
                user_id="u1",
                credential=None,
                connection_string=None,
            )


def test_init_with_connection_string() -> None:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb._chat_history.CosmosClient.from_connection_string",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        history = CosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            connection_string="AccountEndpoint=https://fake;AccountKey=fakekey==;",
        )
        assert history.session_id == "s1"
        assert history.user_id == "u1"


def test_init_with_credential() -> None:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb._chat_history.CosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        history = CosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        assert history.session_id == "s1"
        assert history.messages == []


# ---------------------------------------------------------------------------
# ETag-based concurrency on upsert_messages
# ---------------------------------------------------------------------------


def _make_history() -> Any:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb._chat_history.CosmosClient", return_value=mock_client
    ):
        from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

        history = CosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        history._container = MagicMock()
        return history


def test_load_messages_captures_etag() -> None:
    history = _make_history()
    history._container.read_item.return_value = {
        "id": "s1",
        "user_id": "u1",
        "messages": [],
        "_etag": '"etag-123"',
    }
    history.load_messages()
    assert history._etag == '"etag-123"'


def test_load_messages_no_session_no_etag() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_history()
    history._container.read_item.side_effect = CosmosHttpResponseError(
        status_code=404, message="not found"
    )
    history.load_messages()
    assert not hasattr(history, "_etag") or history._etag is None


def test_upsert_passes_etag_when_available() -> None:
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history._etag = '"etag-abc"'
    history.messages = [HumanMessage(content="hi")]
    history._container.read_item.return_value = {"_etag": '"etag-new"'}

    history.upsert_messages()

    call_kwargs = history._container.upsert_item.call_args[1]
    assert call_kwargs["etag"] == '"etag-abc"'
    assert call_kwargs["match_condition"] == MatchConditions.IfNotModified


def test_upsert_retries_on_412_conflict() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history._etag = '"stale"'
    history.messages = [HumanMessage(content="m1")]

    history._container.upsert_item.side_effect = [
        CosmosHttpResponseError(status_code=412, message="Precondition Failed"),
        None,
    ]
    history._container.read_item.return_value = {
        "id": "s1",
        "user_id": "u1",
        "messages": [],
        "_etag": '"fresh"',
    }
    history.upsert_messages()
    assert history._container.upsert_item.call_count == 2


def test_upsert_raises_on_non_412_error() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history._etag = '"e"'
    history.messages = [HumanMessage(content="x")]
    history._container.upsert_item.side_effect = CosmosHttpResponseError(
        status_code=500, message="Server Error"
    )
    with pytest.raises(CosmosHttpResponseError):
        history.upsert_messages()


def test_upsert_412_merge_preserves_local_message() -> None:
    """A 412 conflict must not drop the local in-flight message."""
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from langchain_core.messages import HumanMessage

    history = _make_history()
    # Simulate a previously-loaded session containing one server message.
    history.messages = [HumanMessage(content="server-msg-1")]
    history._loaded_count = 1
    history._etag = '"stale"'

    # Caller appends a new local message (mirrors add_message behavior).
    history.messages.append(HumanMessage(content="local-new"))

    # First upsert sees a 412; load_messages returns server state with an
    # additional message added by another writer.
    history._container.upsert_item.side_effect = [
        CosmosHttpResponseError(status_code=412, message="Precondition Failed"),
        None,
    ]
    history._container.read_item.side_effect = [
        # load_messages() refresh after 412
        {
            "id": "s1",
            "user_id": "u1",
            "messages": [
                {"type": "human", "data": {"content": "server-msg-1"}},
                {"type": "human", "data": {"content": "other-writer-msg"}},
            ],
            "_etag": '"fresh"',
        },
        # post-success refresh
        {"_etag": '"final"'},
    ]

    history.upsert_messages()

    # Body sent on the retry must contain server state plus local-new.
    final_body = history._container.upsert_item.call_args_list[-1].kwargs["body"]
    contents = [m["data"]["content"] for m in final_body["messages"]]
    assert contents == ["server-msg-1", "other-writer-msg", "local-new"]


def test_upsert_412_retry_uses_refreshed_etag() -> None:
    """Retry after 412 must send the freshly-loaded ETag, not the stale one."""
    from azure.cosmos.exceptions import CosmosHttpResponseError
    from langchain_core.messages import HumanMessage

    history = _make_history()
    history.messages = [HumanMessage(content="m")]
    history._loaded_count = 0
    history._etag = '"stale"'

    history._container.upsert_item.side_effect = [
        CosmosHttpResponseError(status_code=412, message="Precondition Failed"),
        None,
    ]
    history._container.read_item.side_effect = [
        {"id": "s1", "user_id": "u1", "messages": [], "_etag": '"fresh"'},
        {"_etag": '"final"'},
    ]

    history.upsert_messages()

    first_etag = history._container.upsert_item.call_args_list[0].kwargs["etag"]
    second_etag = history._container.upsert_item.call_args_list[1].kwargs["etag"]
    assert first_etag == '"stale"'
    assert second_etag == '"fresh"'


# ---------------------------------------------------------------------------
# Warning logging on failures
# ---------------------------------------------------------------------------


def test_upsert_logs_warning_on_failure(caplog: Any) -> None:
    """upsert_messages logs a warning before re-raising."""
    import logging

    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_history()
    history._container.upsert_item.side_effect = CosmosHttpResponseError(
        status_code=500, message="Server error"
    )

    with caplog.at_level(logging.WARNING), pytest.raises(CosmosHttpResponseError):
        history.upsert_messages()

    assert "Failed to upsert messages for session s1" in caplog.text


def test_clear_logs_warning_on_failure(caplog: Any) -> None:
    """clear logs a warning before re-raising."""
    import logging

    history = _make_history()
    history._container.delete_item.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="boom"):
        history.clear()

    assert "Failed to delete session s1" in caplog.text
