"""Unit tests for AzureAIMemoryStore."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_memory_item(content: str) -> MagicMock:
    """Build a mock memory search item with the given content string."""
    mem = MagicMock()
    mem.memory_item.content = content
    return mem


def _make_search_result(memories: List[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.memories = memories
    return result


def _make_update_result() -> MagicMock:
    poller = MagicMock()
    poller.result.return_value = MagicMock()
    return poller


# ---------------------------------------------------------------------------
# Import helpers (suppress the experimental warning in tests)
# ---------------------------------------------------------------------------

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_azure_ai.stores.azure_ai_memory import (
        AzureAIMemoryStore,
        _KEY_VALUE_PATTERN,
    )


# ---------------------------------------------------------------------------
# _parse_memory_content
# ---------------------------------------------------------------------------


class TestParseMemoryContent:
    """Tests for the static _parse_memory_content helper."""

    def test_valid_format(self) -> None:
        content = "The value for key 'my_key' is: {\"foo\": 1}"
        result = AzureAIMemoryStore._parse_memory_content(content)
        assert result == ("my_key", {"foo": 1})

    def test_invalid_json(self) -> None:
        content = "The value for key 'bad' is: not_json"
        assert AzureAIMemoryStore._parse_memory_content(content) is None

    def test_unrecognised_format(self) -> None:
        content = "User prefers dark roast coffee"
        assert AzureAIMemoryStore._parse_memory_content(content) is None

    def test_non_dict_json(self) -> None:
        # JSON array is not a dict – should return None
        content = "The value for key 'k' is: [1, 2, 3]"
        assert AzureAIMemoryStore._parse_memory_content(content) is None

    def test_key_with_spaces(self) -> None:
        content = "The value for key 'my key name' is: {\"x\": true}"
        result = AzureAIMemoryStore._parse_memory_content(content)
        assert result == ("my key name", {"x": True})


# ---------------------------------------------------------------------------
# _make_message
# ---------------------------------------------------------------------------


class TestMakeMessage:
    """Tests for the static _make_message helper."""

    def test_structure(self) -> None:
        msg = AzureAIMemoryStore._make_message("k", {"a": 1})
        assert msg["role"] == "user"
        assert msg["type"] == "message"
        assert "The value for key 'k' is:" in msg["content"]
        assert '"a": 1' in msg["content"]

    def test_roundtrip(self) -> None:
        key = "prefs"
        value = {"theme": "dark", "lang": "en"}
        msg = AzureAIMemoryStore._make_message(key, value)
        parsed = AzureAIMemoryStore._parse_memory_content(msg["content"])
        assert parsed == (key, value)


# ---------------------------------------------------------------------------
# _namespace_to_scope
# ---------------------------------------------------------------------------


class TestNamespaceToScope:
    def test_single(self) -> None:
        assert AzureAIMemoryStore._namespace_to_scope(("users",)) == "users"

    def test_nested(self) -> None:
        assert (
            AzureAIMemoryStore._namespace_to_scope(("users", "alice"))
            == "users.alice"
        )

    def test_empty(self) -> None:
        assert AzureAIMemoryStore._namespace_to_scope(()) == ""


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for AzureAIMemoryStore.__init__."""

    def test_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="endpoint"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                AzureAIMemoryStore(memory_store_name="test")

    def test_endpoint_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT",
            "https://example.services.ai.azure.com/api/projects/p",
        )
        with (
            warnings.catch_warnings(),
            patch("langchain_azure_ai.stores.azure_ai_memory.AIProjectClient"),
            patch(
                "langchain_azure_ai.stores.azure_ai_memory.DefaultAzureCredential"
            ),
        ):
            warnings.simplefilter("ignore")
            store = AzureAIMemoryStore(memory_store_name="ms")
        assert store._memory_store_name == "ms"

    def test_explicit_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        with (
            warnings.catch_warnings(),
            patch("langchain_azure_ai.stores.azure_ai_memory.AIProjectClient"),
            patch(
                "langchain_azure_ai.stores.azure_ai_memory.DefaultAzureCredential"
            ),
        ):
            warnings.simplefilter("ignore")
            store = AzureAIMemoryStore(
                memory_store_name="ms",
                endpoint="https://example.services.ai.azure.com/api/projects/p",
            )
        assert store._memory_store_name == "ms"


# ---------------------------------------------------------------------------
# batch operations (mocked client)
# ---------------------------------------------------------------------------


def _make_store() -> AzureAIMemoryStore:
    """Return an AzureAIMemoryStore with a mocked AIProjectClient."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch(
            "langchain_azure_ai.stores.azure_ai_memory.AIProjectClient"
        ) as MockClient, patch(
            "langchain_azure_ai.stores.azure_ai_memory.DefaultAzureCredential"
        ):
            store = AzureAIMemoryStore(
                memory_store_name="my-store",
                endpoint="https://ex.services.ai.azure.com/api/projects/p",
            )
            store._client = MockClient.return_value
    return store


class TestBatchGet:
    """Tests for GetOp handling."""

    def test_get_returns_item_when_found(self) -> None:
        store = _make_store()
        content = "The value for key 'prefs' is: {\"theme\": \"dark\"}"
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([_make_memory_item(content)])
        )

        item = store.get(("users", "alice"), "prefs")
        assert item is not None
        assert item.key == "prefs"
        assert item.value == {"theme": "dark"}
        assert item.namespace == ("users", "alice")

    def test_get_returns_none_when_not_found(self) -> None:
        store = _make_store()
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        item = store.get(("users", "alice"), "missing")
        assert item is None

    def test_get_skips_unparseable_memories(self) -> None:
        store = _make_store()
        # A memory that doesn't match our pattern + the one that does.
        memories = [
            _make_memory_item("User likes coffee"),
            _make_memory_item(
                "The value for key 'k' is: {\"v\": 1}"
            ),
        ]
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        item = store.get(("ns",), "k")
        assert item is not None
        assert item.value == {"v": 1}

    def test_get_wrong_key_returns_none(self) -> None:
        store = _make_store()
        content = "The value for key 'other_key' is: {\"x\": 1}"
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([_make_memory_item(content)])
        )

        item = store.get(("ns",), "target_key")
        assert item is None

    def test_get_http_error_returns_none(self) -> None:
        from azure.core.exceptions import HttpResponseError

        store = _make_store()
        store._client.beta.memory_stores.search_memories.side_effect = (
            HttpResponseError("not found")
        )

        item = store.get(("ns",), "k")
        assert item is None


class TestBatchPut:
    """Tests for PutOp handling."""

    def test_put_calls_begin_update_memories(self) -> None:
        store = _make_store()
        poller = _make_update_result()
        store._client.beta.memory_stores.begin_update_memories.return_value = poller

        store.put(("users",), "name", {"val": "Alice"})

        store._client.beta.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = (
            store._client.beta.memory_stores.begin_update_memories.call_args
        )
        assert call_kwargs.kwargs["scope"] == "users"
        assert call_kwargs.kwargs["update_delay"] == 0
        items = call_kwargs.kwargs["items"]
        assert len(items) == 1
        assert "name" in items[0]["content"]

    def test_put_none_calls_delete_scope(self) -> None:
        store = _make_store()

        store.put(("users", "alice"), "prefs", None)

        store._client.beta.memory_stores.delete_scope.assert_called_once_with(
            name="my-store",
            scope="users.alice",
        )

    def test_put_none_suppresses_delete_error(self) -> None:
        store = _make_store()
        store._client.beta.memory_stores.delete_scope.side_effect = Exception(
            "service error"
        )

        # Should not raise
        store.put(("ns",), "k", None)

    def test_put_http_error_propagates(self) -> None:
        from azure.core.exceptions import HttpResponseError

        store = _make_store()
        store._client.beta.memory_stores.begin_update_memories.side_effect = (
            HttpResponseError("server error")
        )

        with pytest.raises(HttpResponseError):
            store.put(("ns",), "k", {"a": 1})


class TestBatchSearch:
    """Tests for SearchOp handling."""

    def test_search_returns_parsed_items(self) -> None:
        store = _make_store()
        memories = [
            _make_memory_item(
                "The value for key 'prefs' is: {\"theme\": \"dark\"}"
            ),
            _make_memory_item(
                "The value for key 'name' is: {\"val\": \"Alice\"}"
            ),
        ]
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        results = store.search(("users",), query="user data")
        assert len(results) == 2
        keys = {r.key for r in results}
        assert "prefs" in keys
        assert "name" in keys

    def test_search_skips_unparseable(self) -> None:
        store = _make_store()
        memories = [
            _make_memory_item("User likes dark coffee"),
            _make_memory_item(
                "The value for key 'k' is: {\"v\": 1}"
            ),
        ]
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        results = store.search(("ns",), query="k")
        assert len(results) == 1
        assert results[0].key == "k"

    def test_search_http_error_returns_empty(self) -> None:
        from azure.core.exceptions import HttpResponseError

        store = _make_store()
        store._client.beta.memory_stores.search_memories.side_effect = (
            HttpResponseError("error")
        )

        results = store.search(("ns",), query="q")
        assert results == []

    def test_search_uses_scope_from_namespace_prefix(self) -> None:
        store = _make_store()
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        store.search(("users", "alice"), query="q")

        call_kwargs = (
            store._client.beta.memory_stores.search_memories.call_args
        )
        assert call_kwargs.kwargs["scope"] == "users.alice"


class TestBatchListNamespaces:
    """Tests for ListNamespacesOp handling."""

    def test_list_namespaces_returns_empty(self) -> None:
        store = _make_store()
        result = store.list_namespaces()
        assert result == []


# ---------------------------------------------------------------------------
# Async abatch
# ---------------------------------------------------------------------------


class TestAbatch:
    """Tests for the async abatch method."""

    @pytest.mark.asyncio
    async def test_abatch_get(self) -> None:
        store = _make_store()
        content = "The value for key 'k' is: {\"a\": 1}"
        store._client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([_make_memory_item(content)])
        )

        item = await store.aget(("ns",), "k")
        assert item is not None
        assert item.value == {"a": 1}

    @pytest.mark.asyncio
    async def test_abatch_put(self) -> None:
        store = _make_store()
        poller = _make_update_result()
        store._client.beta.memory_stores.begin_update_memories.return_value = poller

        await store.aput(("ns",), "k", {"a": 1})

        store._client.beta.memory_stores.begin_update_memories.assert_called_once()
