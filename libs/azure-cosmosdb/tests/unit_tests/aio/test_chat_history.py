"""Unit tests for AsyncCosmosDBChatMessageHistory."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core import MatchConditions
from langchain_core.messages import HumanMessage

# ---- init validation -------------------------------------------------------


def test_missing_credential_and_connection_string_raises() -> None:
    with patch("langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient"):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        with pytest.raises(
            ValueError,
            match="Either a connection string or a credential must be set",
        ):
            AsyncCosmosDBChatMessageHistory(
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
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient.from_connection_string",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
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
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        assert history.session_id == "s1"
        assert history.messages == []


# ---- sync stubs raise NotImplementedError ----------------------------------


def test_add_message_sync_raises() -> None:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        with pytest.raises(NotImplementedError):
            history.add_message(HumanMessage(content="hello"))


def test_clear_sync_raises() -> None:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        with pytest.raises(NotImplementedError):
            history.clear()


# ---- aadd_messages (mock-based) --------------------------------------------


async def test_aadd_messages_upserts() -> None:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        # Manually set the container to a mock
        history._container = AsyncMock()

        msgs = [HumanMessage(content="hi")]
        await history.aadd_messages(msgs)

        assert len(history.messages) == 1
        history._container.upsert_item.assert_awaited_once()
        call_body = history._container.upsert_item.call_args[1]["body"]
        assert call_body["id"] == "s1"
        assert call_body["user_id"] == "u1"


# ---- aclear (mock-based) --------------------------------------------------


async def test_aclear_deletes_item() -> None:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        history._container = AsyncMock()
        history.messages = [HumanMessage(content="old")]

        await history.aclear()

        assert history.messages == []
        history._container.delete_item.assert_awaited_once_with(
            item="s1", partition_key="u1"
        )


# ---------------------------------------------------------------------------
# Async ETag concurrency on upsert_messages
# ---------------------------------------------------------------------------


def _make_async_history_helper() -> Any:
    mock_client = MagicMock()
    with patch(
        "langchain_azure_cosmosdb.aio._chat_history.AsyncCosmosClient",
        return_value=mock_client,
    ):
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory(
            cosmos_endpoint="https://fake.documents.azure.com:443/",
            cosmos_database="testdb",
            cosmos_container="testcontainer",
            session_id="s1",
            user_id="u1",
            credential="fake-credential",
        )
        history._container = AsyncMock()
        return history


async def test_async_load_messages_captures_etag() -> None:
    history = _make_async_history_helper()
    history._container.read_item.return_value = {
        "id": "s1",
        "user_id": "u1",
        "messages": [],
        "_etag": '"async-etag"',
    }
    await history.load_messages()
    assert history._etag == '"async-etag"'


async def test_async_upsert_passes_etag() -> None:
    history = _make_async_history_helper()
    history._etag = '"etag-old"'
    history.messages = [HumanMessage(content="hi")]
    history._container.read_item.return_value = {"_etag": '"etag-new"'}

    await history.upsert_messages()

    call_kwargs = history._container.upsert_item.call_args[1]
    assert call_kwargs["etag"] == '"etag-old"'
    assert call_kwargs["match_condition"] == MatchConditions.IfNotModified


async def test_async_upsert_retries_on_412() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_async_history_helper()
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
    await history.upsert_messages()
    assert history._container.upsert_item.call_count == 2


async def test_async_upsert_raises_on_non_412() -> None:
    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_async_history_helper()
    history._etag = '"e"'
    history.messages = [HumanMessage(content="x")]
    history._container.upsert_item.side_effect = CosmosHttpResponseError(
        status_code=500, message="Server Error"
    )
    with pytest.raises(CosmosHttpResponseError):
        await history.upsert_messages()


async def test_async_upsert_412_merge_preserves_local_message() -> None:
    """A 412 conflict must not drop the local in-flight message."""
    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_async_history_helper()
    history.messages = [HumanMessage(content="server-msg-1")]
    history._loaded_count = 1
    history._etag = '"stale"'
    history.messages.append(HumanMessage(content="local-new"))

    history._container.upsert_item.side_effect = [
        CosmosHttpResponseError(status_code=412, message="Precondition Failed"),
        None,
    ]
    history._container.read_item.side_effect = [
        {
            "id": "s1",
            "user_id": "u1",
            "messages": [
                {"type": "human", "data": {"content": "server-msg-1"}},
                {"type": "human", "data": {"content": "other-writer-msg"}},
            ],
            "_etag": '"fresh"',
        },
        {"_etag": '"final"'},
    ]

    await history.upsert_messages()

    final_body = history._container.upsert_item.call_args_list[-1].kwargs["body"]
    contents = [m["data"]["content"] for m in final_body["messages"]]
    assert contents == ["server-msg-1", "other-writer-msg", "local-new"]


async def test_async_upsert_412_retry_uses_refreshed_etag() -> None:
    """Retry after 412 must send the freshly-loaded ETag, not the stale one."""
    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_async_history_helper()
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

    await history.upsert_messages()

    first_etag = history._container.upsert_item.call_args_list[0].kwargs["etag"]
    second_etag = history._container.upsert_item.call_args_list[1].kwargs["etag"]
    assert first_etag == '"stale"'
    assert second_etag == '"fresh"'


# ---------------------------------------------------------------------------
# Warning logging on failures
# ---------------------------------------------------------------------------


async def test_aclear_logs_warning_on_failure(caplog: Any) -> None:
    """aclear logs a warning before re-raising."""
    import logging

    history = _make_async_history_helper()
    history._container.delete_item.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="boom"):
        await history.aclear()

    assert "Failed to delete session s1" in caplog.text


async def test_upsert_logs_warning_on_failure(caplog: Any) -> None:
    """upsert_messages logs a warning before re-raising."""
    import logging

    from azure.cosmos.exceptions import CosmosHttpResponseError

    history = _make_async_history_helper()
    history._container.upsert_item.side_effect = CosmosHttpResponseError(
        status_code=500, message="Server error"
    )

    with caplog.at_level(logging.WARNING), pytest.raises(CosmosHttpResponseError):
        await history.upsert_messages()

    assert "Failed to upsert messages for session s1" in caplog.text
