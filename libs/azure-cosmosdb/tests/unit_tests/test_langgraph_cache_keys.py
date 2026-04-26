"""Unit tests for langgraph cache key helpers."""

from typing import Any

from langchain_azure_cosmosdb._langgraph_cache import _make_cache_key


def test_make_cache_key_basic() -> None:
    result = _make_cache_key("ns", "key")
    assert result == "cache$ns$key"


def test_make_cache_key_with_commas() -> None:
    result = _make_cache_key("a,b,c", "mykey")
    assert result == "cache$a,b,c$mykey"


def test_make_cache_key_empty_namespace() -> None:
    result = _make_cache_key("", "key")
    assert result == "cache$$key"


def test_make_cache_key_empty_key() -> None:
    result = _make_cache_key("ns", "")
    assert result == "cache$ns$"


# ---------------------------------------------------------------------------
# Sync cache context manager
# ---------------------------------------------------------------------------


class TestSyncCacheContextManager:
    def _make_cache(self) -> Any:
        from unittest.mock import MagicMock

        from langchain_azure_cosmosdb._langgraph_cache import CosmosDBCacheSync

        cache = CosmosDBCacheSync.__new__(CosmosDBCacheSync)
        cache.client = MagicMock()
        cache.container = MagicMock()
        return cache

    def test_close(self) -> None:
        cache = self._make_cache()
        cache.close()
        cache.client.close.assert_called_once()

    def test_context_manager(self) -> None:
        cache = self._make_cache()
        with cache as c:
            assert c is cache
        cache.client.close.assert_called_once()

    def test_clear_namespaces_passes_partition_key(self) -> None:
        cache = self._make_cache()
        cache.container.query_items.return_value = [{"id": "doc1"}]
        cache.clear(namespaces=[("a", "b")])
        call_kwargs = cache.container.query_items.call_args[1]
        assert call_kwargs["partition_key"] == "a|b"
        assert "enable_cross_partition_query" not in call_kwargs


# ---------------------------------------------------------------------------
# cosmos_client_kwargs propagation
# ---------------------------------------------------------------------------


def test_cache_cosmos_client_kwargs_forwarded() -> None:
    """CosmosDBCacheSync forwards cosmos_client_kwargs to CosmosClient."""
    from unittest.mock import MagicMock, patch

    with patch(
        "langchain_azure_cosmosdb._langgraph_cache.CosmosClient",
    ) as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = MagicMock()

        from langchain_azure_cosmosdb import CosmosDBCacheSync

        CosmosDBCacheSync(
            database_name="db",
            container_name="ctr",
            endpoint="https://fake.documents.azure.com:443/",
            key="fakekey",
            cosmos_client_kwargs={"retry_total": 3},
        )
        _, kwargs = mock_cls.call_args
        assert kwargs.get("retry_total") == 3


# ---------------------------------------------------------------------------
# Store: cosmos_client_kwargs + exception narrowing
# ---------------------------------------------------------------------------


def test_store_from_conn_string_passes_kwargs() -> None:
    """CosmosDBStore.from_conn_string forwards cosmos_client_kwargs."""
    from unittest.mock import MagicMock, patch

    with patch(
        "langchain_azure_cosmosdb._langgraph_store.CosmosClient",
    ) as mock_cls:
        mock_cls.from_connection_string.return_value = MagicMock()

        from langchain_azure_cosmosdb import CosmosDBStore

        CosmosDBStore.from_conn_string(
            conn_string="AccountEndpoint=https://fake;AccountKey=abc",
            cosmos_client_kwargs={"retry_total": 7},
        )
        _, kwargs = mock_cls.from_connection_string.call_args
        assert kwargs.get("retry_total") == 7


def test_store_from_endpoint_passes_kwargs() -> None:
    """CosmosDBStore.from_endpoint forwards cosmos_client_kwargs."""
    from unittest.mock import MagicMock, patch

    with patch(
        "langchain_azure_cosmosdb._langgraph_store.CosmosClient",
    ) as mock_cls:
        mock_cls.return_value = MagicMock()

        from langchain_azure_cosmosdb import CosmosDBStore

        CosmosDBStore.from_endpoint(
            endpoint="https://fake.documents.azure.com:443/",
            credential="fakekey",
            cosmos_client_kwargs={"retry_total": 9},
        )
        _, kwargs = mock_cls.call_args
        assert kwargs.get("retry_total") == 9


def test_store_put_swallows_not_found_only() -> None:
    """_batch_put_ops catches CosmosResourceNotFoundError, not others."""
    from unittest.mock import MagicMock

    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    from langchain_azure_cosmosdb._langgraph_store import CosmosDBStore
    from langgraph.store.base import PutOp

    store = CosmosDBStore.__new__(CosmosDBStore)
    store.index_config = None
    store.embeddings = None
    store._deserializer = None
    store.ttl_config = None
    store._container = MagicMock()

    # CosmosResourceNotFoundError is swallowed (new doc)
    store._container.read_item.side_effect = CosmosResourceNotFoundError(
        status_code=404, message="Not found"
    )
    store._container.upsert_item.return_value = None

    ops = [
        (
            0,
            PutOp(
                namespace=("test",),
                key="k1",
                value={"text": "hello"},
            ),
        ),
    ]
    store._batch_put_ops(ops)
    store._container.upsert_item.assert_called_once()

    # Other exceptions propagate
    store._container.reset_mock()
    store._container.read_item.side_effect = RuntimeError("unexpected")
    import pytest

    with pytest.raises(RuntimeError, match="unexpected"):
        store._batch_put_ops(ops)
