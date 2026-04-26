"""Unit tests for langgraph cache key helpers."""

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
    def _make_cache(self):
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
