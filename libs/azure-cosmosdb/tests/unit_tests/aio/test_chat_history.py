"""Unit tests for AsyncCosmosDBChatMessageHistory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

# ---- init validation -------------------------------------------------------


def test_missing_credential_and_connection_string_raises() -> None:
    with patch("azure.cosmos.aio.CosmosClient"):
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
        "azure.cosmos.aio.CosmosClient.from_connection_string",
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
        "azure.cosmos.aio.CosmosClient",
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
    with patch("azure.cosmos.aio.CosmosClient", return_value=mock_client):
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
    with patch("azure.cosmos.aio.CosmosClient", return_value=mock_client):
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
    with patch("azure.cosmos.aio.CosmosClient", return_value=mock_client):
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
    with patch("azure.cosmos.aio.CosmosClient", return_value=mock_client):
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
