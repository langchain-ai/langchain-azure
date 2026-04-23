"""Unit tests for cosmos_client_kwargs propagation and error handling resilience."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

# ---------------------------------------------------------------------------
# Sync: CosmosDBSaverSync — cosmos_client_kwargs propagation
# ---------------------------------------------------------------------------

USER_AGENT_CHECKPOINT = "langchain-azure-cosmosdb-checkpoint"
USER_AGENT_CACHE = "langchain-azure-cosmosdb-lgcache"


class TestCosmosDBSaverSyncKwargs:
    """Verify cosmos_client_kwargs flows through to CosmosClient."""

    @patch(
        "langchain_azure_cosmosdb._langgraph_checkpoint_store.CosmosClient",
    )
    def test_kwargs_passed_with_key(self, mock_cosmos_cls: MagicMock) -> None:
        mock_client = mock_cosmos_cls.return_value
        mock_db = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = MagicMock()

        sentinel = object()
        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            CosmosDBSaverSync,
        )

        CosmosDBSaverSync(
            "db",
            "container",
            endpoint="https://fake.documents.azure.com:443/",
            key="fake_key",
            cosmos_client_kwargs={"retry_options": sentinel},
        )

        mock_cosmos_cls.assert_called_once_with(
            "https://fake.documents.azure.com:443/",
            "fake_key",
            user_agent=USER_AGENT_CHECKPOINT,
            retry_options=sentinel,
        )

    @patch(
        "langchain_azure_cosmosdb._langgraph_checkpoint_store.DefaultAzureCredential",
    )
    @patch(
        "langchain_azure_cosmosdb._langgraph_checkpoint_store.CosmosClient",
    )
    def test_kwargs_passed_with_credential(
        self, mock_cosmos_cls: MagicMock, mock_cred_cls: MagicMock
    ) -> None:
        mock_client = mock_cosmos_cls.return_value
        mock_db = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = MagicMock()

        sentinel = object()
        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            CosmosDBSaverSync,
        )

        CosmosDBSaverSync(
            "db",
            "container",
            endpoint="https://fake.documents.azure.com:443/",
            cosmos_client_kwargs={"retry_options": sentinel},
        )

        mock_cosmos_cls.assert_called_once_with(
            "https://fake.documents.azure.com:443/",
            credential=mock_cred_cls.return_value,
            user_agent=USER_AGENT_CHECKPOINT,
            retry_options=sentinel,
        )

    @patch(
        "langchain_azure_cosmosdb._langgraph_checkpoint_store.CosmosClient",
    )
    def test_no_kwargs_still_works(self, mock_cosmos_cls: MagicMock) -> None:
        mock_client = mock_cosmos_cls.return_value
        mock_db = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = MagicMock()

        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            CosmosDBSaverSync,
        )

        CosmosDBSaverSync(
            "db",
            "container",
            endpoint="https://fake.documents.azure.com:443/",
            key="fake_key",
        )

        mock_cosmos_cls.assert_called_once_with(
            "https://fake.documents.azure.com:443/",
            "fake_key",
            user_agent=USER_AGENT_CHECKPOINT,
        )


# ---------------------------------------------------------------------------
# Sync: CosmosDBCacheSync — cosmos_client_kwargs propagation
# ---------------------------------------------------------------------------


class TestCosmosDBCacheSyncKwargs:
    """Verify cosmos_client_kwargs flows through to CosmosClient."""

    @patch("langchain_azure_cosmosdb._langgraph_cache.CosmosClient")
    def test_kwargs_passed_with_key(self, mock_cosmos_cls: MagicMock) -> None:
        mock_client = mock_cosmos_cls.return_value
        mock_db = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = MagicMock()

        sentinel = object()
        from langchain_azure_cosmosdb._langgraph_cache import CosmosDBCacheSync

        CosmosDBCacheSync(
            "db",
            "container",
            endpoint="https://fake.documents.azure.com:443/",
            key="fake_key",
            cosmos_client_kwargs={"retry_options": sentinel},
        )

        mock_cosmos_cls.assert_called_once_with(
            "https://fake.documents.azure.com:443/",
            "fake_key",
            user_agent=USER_AGENT_CACHE,
            retry_options=sentinel,
        )


# ---------------------------------------------------------------------------
# Sync: CosmosDBStore — cosmos_client_kwargs propagation
# ---------------------------------------------------------------------------


class TestCosmosDBStoreKwargs:
    """Verify cosmos_client_kwargs flows through factory methods."""

    @patch("langchain_azure_cosmosdb._langgraph_store.CosmosClient")
    def test_from_conn_string_passes_kwargs(self, mock_cosmos_cls: MagicMock) -> None:
        sentinel = object()
        from langchain_azure_cosmosdb._langgraph_store import CosmosDBStore

        CosmosDBStore.from_conn_string(
            "AccountEndpoint=https://fake;AccountKey=key;",
            cosmos_client_kwargs={"retry_options": sentinel},
        )

        mock_cosmos_cls.from_connection_string.assert_called_once_with(
            "AccountEndpoint=https://fake;AccountKey=key;",
            user_agent="langchain-azure-cosmosdb-lgstore",
            retry_options=sentinel,
        )

    @patch("langchain_azure_cosmosdb._langgraph_store.DefaultAzureCredential")
    @patch("langchain_azure_cosmosdb._langgraph_store.CosmosClient")
    def test_from_endpoint_passes_kwargs(
        self, mock_cosmos_cls: MagicMock, mock_cred_cls: MagicMock
    ) -> None:
        sentinel = object()
        from langchain_azure_cosmosdb._langgraph_store import CosmosDBStore

        CosmosDBStore.from_endpoint(
            "https://fake.documents.azure.com:443/",
            cosmos_client_kwargs={"retry_options": sentinel},
        )

        mock_cosmos_cls.assert_called_once_with(
            "https://fake.documents.azure.com:443/",
            credential=mock_cred_cls.return_value,
            user_agent="langchain-azure-cosmosdb-lgstore",
            retry_options=sentinel,
        )


# ---------------------------------------------------------------------------
# Exception narrowing: _batch_put_ops point-read
# ---------------------------------------------------------------------------


class TestBatchPutOpsExceptionNarrowing:
    """Verify that non-404 errors on read_item propagate instead of being swallowed."""

    def test_429_propagates_on_point_read(self) -> None:
        """A 429 rate-limit error on read_item must NOT be swallowed."""
        from langchain_azure_cosmosdb._langgraph_store import CosmosDBStore

        mock_client = MagicMock()
        store = CosmosDBStore(conn=mock_client)
        store._container = MagicMock()

        # Simulate 429 on point-read
        error_429 = CosmosHttpResponseError(
            status_code=429, message="Rate limit exceeded"
        )
        store._container.read_item.side_effect = error_429

        from langgraph.store.base import PutOp

        op = PutOp(namespace=("test",), key="k1", value={"data": "value"})

        with pytest.raises(CosmosHttpResponseError) as exc_info:
            store._batch_put_ops([(0, op)])

        assert exc_info.value.status_code == 429

    def test_404_is_silently_handled(self) -> None:
        """A CosmosResourceNotFoundError on read_item should be handled gracefully."""
        from langchain_azure_cosmosdb._langgraph_store import CosmosDBStore

        mock_client = MagicMock()
        store = CosmosDBStore(conn=mock_client)
        store._container = MagicMock()

        # Simulate 404 on point-read
        store._container.read_item.side_effect = CosmosResourceNotFoundError()
        store._container.upsert_item.return_value = None

        from langgraph.store.base import PutOp

        op = PutOp(namespace=("test",), key="k1", value={"data": "value"})

        # Should not raise
        store._batch_put_ops([(0, op)])
        store._container.upsert_item.assert_called_once()


# ---------------------------------------------------------------------------
# Async exception narrowing
# ---------------------------------------------------------------------------


class TestAsyncBatchPutOpsExceptionNarrowing:
    """Verify that non-404 errors on async read_item propagate."""

    async def test_429_propagates_on_async_point_read(self) -> None:
        """A 429 rate-limit error on async read_item must NOT be swallowed."""
        from langchain_azure_cosmosdb.aio._langgraph_store import AsyncCosmosDBStore

        mock_client = MagicMock()
        store = AsyncCosmosDBStore(conn=mock_client)
        store._container = MagicMock()

        error_429 = CosmosHttpResponseError(
            status_code=429, message="Rate limit exceeded"
        )

        async def raise_429(*args: object, **kwargs: object) -> None:
            raise error_429

        store._container.read_item = raise_429
        store._container.upsert_item = MagicMock()

        from langgraph.store.base import PutOp

        op = PutOp(namespace=("test",), key="k1", value={"data": "value"})

        with pytest.raises(CosmosHttpResponseError) as exc_info:
            await store._abatch_put_ops([(0, op)])

        assert exc_info.value.status_code == 429

    async def test_404_is_silently_handled_async(self) -> None:
        """CosmosResourceNotFoundError on async read_item is handled gracefully."""
        from langchain_azure_cosmosdb.aio._langgraph_store import AsyncCosmosDBStore

        mock_client = MagicMock()
        store = AsyncCosmosDBStore(conn=mock_client)
        store._container = MagicMock()

        async def raise_not_found(*args: object, **kwargs: object) -> None:
            raise CosmosResourceNotFoundError()

        async def noop_upsert(*args: object, **kwargs: object) -> None:
            pass

        store._container.read_item = raise_not_found
        store._container.upsert_item = noop_upsert

        from langgraph.store.base import PutOp

        op = PutOp(namespace=("test",), key="k1", value={"data": "value"})

        # Should not raise
        await store._abatch_put_ops([(0, op)])


# ---------------------------------------------------------------------------
# Chat history logging on write failures
# ---------------------------------------------------------------------------


class TestChatHistoryWriteLogging:
    """Verify that write failures are logged before re-raising."""

    @patch("langchain_azure_cosmosdb._chat_history.logger")
    def test_upsert_messages_logs_on_failure(self, mock_logger: MagicMock) -> None:
        from langchain_azure_cosmosdb._chat_history import (
            CosmosDBChatMessageHistory,
        )

        history = CosmosDBChatMessageHistory.__new__(CosmosDBChatMessageHistory)
        history.session_id = "sess-1"
        history.user_id = "user-1"
        history.messages = []
        history._container = MagicMock()
        history._container.upsert_item.side_effect = CosmosHttpResponseError(
            status_code=500, message="Internal Server Error"
        )

        with pytest.raises(CosmosHttpResponseError):
            history.upsert_messages()

        mock_logger.warning.assert_called_once()
        assert "sess-1" in mock_logger.warning.call_args[0][1]

    @patch("langchain_azure_cosmosdb._chat_history.logger")
    def test_clear_logs_on_failure(self, mock_logger: MagicMock) -> None:
        from langchain_azure_cosmosdb._chat_history import (
            CosmosDBChatMessageHistory,
        )

        history = CosmosDBChatMessageHistory.__new__(CosmosDBChatMessageHistory)
        history.session_id = "sess-1"
        history.user_id = "user-1"
        history.messages = []
        history._container = MagicMock()
        history._container.delete_item.side_effect = CosmosHttpResponseError(
            status_code=429, message="Rate limit"
        )

        with pytest.raises(CosmosHttpResponseError):
            history.clear()

        mock_logger.warning.assert_called_once()
        assert "sess-1" in mock_logger.warning.call_args[0][1]


class TestAsyncChatHistoryWriteLogging:
    """Verify that async write failures are logged before re-raising."""

    @patch("langchain_azure_cosmosdb.aio._chat_history.logger")
    async def test_upsert_messages_logs_on_failure(
        self, mock_logger: MagicMock
    ) -> None:
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory.__new__(
            AsyncCosmosDBChatMessageHistory
        )
        history.session_id = "sess-1"
        history.user_id = "user-1"
        history.messages = []
        history._container = MagicMock()

        async def raise_error(*args: object, **kwargs: object) -> None:
            raise CosmosHttpResponseError(
                status_code=500, message="Internal Server Error"
            )

        history._container.upsert_item = raise_error

        with pytest.raises(CosmosHttpResponseError):
            await history.upsert_messages()

        mock_logger.warning.assert_called_once()

    @patch("langchain_azure_cosmosdb.aio._chat_history.logger")
    async def test_aclear_logs_on_failure(self, mock_logger: MagicMock) -> None:
        from langchain_azure_cosmosdb.aio._chat_history import (
            AsyncCosmosDBChatMessageHistory,
        )

        history = AsyncCosmosDBChatMessageHistory.__new__(
            AsyncCosmosDBChatMessageHistory
        )
        history.session_id = "sess-1"
        history.user_id = "user-1"
        history.messages = []
        history._container = MagicMock()

        async def raise_error(*args: object, **kwargs: object) -> None:
            raise CosmosHttpResponseError(status_code=429, message="Rate limit")

        history._container.delete_item = raise_error

        with pytest.raises(CosmosHttpResponseError):
            await history.aclear()

        mock_logger.warning.assert_called_once()
