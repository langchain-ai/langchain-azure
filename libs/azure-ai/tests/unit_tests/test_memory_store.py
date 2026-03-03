"""Unit tests for AzureAIMemoryStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_memory_item(
    memory_id: str,
    scope: str,
    content: str,
    updated_at: Optional[datetime] = None,
) -> MagicMock:
    """Build a mock memory item with the given content string."""
    mem = MagicMock()
    mem.memory_id = memory_id
    mem.scope = scope
    mem.content = content
    mem.updated_at = updated_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return mem


def _make_search_result(memories: List[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.memories = [_wrap_search_item(m) for m in memories]
    return result


def _wrap_search_item(memory_item: MagicMock) -> MagicMock:
    """Wrap a MemoryItem in a MemorySearchItem mock."""
    wrapper = MagicMock()
    wrapper.memory_item = memory_item
    return wrapper


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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AIProjectClient."""
    client = MagicMock()
    client.beta.memory_stores.search_memories.return_value = _make_search_result([])
    poller = MagicMock()
    poller.result.return_value = MagicMock()
    client.beta.memory_stores.begin_update_memories.return_value = poller
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> AzureAIMemoryStore:
    """Create an AzureAIMemoryStore with a patched AIProjectClient."""
    import azure.ai.projects as _azmod

    with patch.object(_azmod, "AIProjectClient", return_value=mock_client):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return AzureAIMemoryStore(
                endpoint="https://example.azure.com/api/projects/proj",
                memory_store_name="my-store",
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

    def test_legacy_json_format(self) -> None:
        """Legacy JSON format {"key": ..., "value": ...} is also accepted."""
        content = json.dumps({"key": "prefs", "value": {"theme": "dark"}})
        result = AzureAIMemoryStore._parse_memory_content(content)
        assert result == ("prefs", {"theme": "dark"})

    def test_legacy_json_format_invalid_value_type(self) -> None:
        """Legacy JSON with non-dict value returns None."""
        content = json.dumps({"key": "k", "value": [1, 2, 3]})
        assert AzureAIMemoryStore._parse_memory_content(content) is None

    def test_legacy_json_format_missing_key(self) -> None:
        """Legacy JSON missing 'key' field returns None."""
        content = json.dumps({"value": {"x": 1}})
        assert AzureAIMemoryStore._parse_memory_content(content) is None


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
        # Components are joined with "_" (not ".") since Azure AI scopes
        # only allow [A-Za-z0-9_-].
        assert (
            AzureAIMemoryStore._namespace_to_scope(("users", "alice"))
            == "users_alice"
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
        import azure.ai.projects as _azmod

        with (
            warnings.catch_warnings(),
            patch.object(_azmod, "AIProjectClient", return_value=MagicMock()),
        ):
            warnings.simplefilter("ignore")
            store = AzureAIMemoryStore(memory_store_name="ms")
        assert store._memory_store_name == "ms"

    def test_explicit_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        import azure.ai.projects as _azmod

        with (
            warnings.catch_warnings(),
            patch.object(_azmod, "AIProjectClient", return_value=MagicMock()),
        ):
            warnings.simplefilter("ignore")
            store = AzureAIMemoryStore(
                memory_store_name="ms",
                endpoint="https://example.services.ai.azure.com/api/projects/p",
            )
        assert store._memory_store_name == "ms"

    def test_api_version_and_client_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """api_version and client_kwargs are forwarded to AIProjectClient."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        import azure.ai.projects as _azmod

        with (
            warnings.catch_warnings(),
            patch.object(
                _azmod, "AIProjectClient", return_value=MagicMock()
            ) as mock_ctor,
        ):
            warnings.simplefilter("ignore")
            AzureAIMemoryStore(
                memory_store_name="ms",
                endpoint="https://example.azure.com/api/projects/p",
                api_version="2024-01-01",
                client_kwargs={"timeout": 30},
            )
        call_kwargs = mock_ctor.call_args.kwargs
        assert call_kwargs.get("api_version") == "2024-01-01"
        assert call_kwargs.get("timeout") == 30

    def test_user_agent_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The user_agent is set to 'langchain-azure-ai' by default."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        import azure.ai.projects as _azmod

        with (
            warnings.catch_warnings(),
            patch.object(
                _azmod, "AIProjectClient", return_value=MagicMock()
            ) as mock_ctor,
        ):
            warnings.simplefilter("ignore")
            AzureAIMemoryStore(
                memory_store_name="ms",
                endpoint="https://example.azure.com/api/projects/p",
            )
        call_kwargs = mock_ctor.call_args.kwargs
        assert call_kwargs.get("user_agent") == "langchain-azure-ai"

    def test_missing_sdk_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ImportError is raised with helpful message when SDK is missing."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        import sys

        # Temporarily remove the module from sys.modules to simulate
        # an environment where azure-ai-projects is not installed.
        saved = sys.modules.pop("azure.ai.projects", None)
        sys.modules["azure.ai.projects"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="azure-ai-projects"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    AzureAIMemoryStore(
                        memory_store_name="ms",
                        endpoint="https://example.azure.com/api/projects/p",
                    )
        finally:
            if saved is not None:
                sys.modules["azure.ai.projects"] = saved
            else:
                sys.modules.pop("azure.ai.projects", None)


# ---------------------------------------------------------------------------
# batch operations (mocked client)
# ---------------------------------------------------------------------------


class TestBatchGet:
    """Tests for GetOp handling."""

    def test_get_returns_item_when_found(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        content = "The value for key 'prefs' is: {\"theme\": \"dark\"}"
        mem = _make_memory_item("m1", "users_alice", content)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = store.get(("users", "alice"), "prefs")
        assert item is not None
        assert item.key == "prefs"
        assert item.value == {"theme": "dark"}
        assert item.namespace == ("users", "alice")

    def test_get_uses_updated_at_timestamp(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """Item timestamps come from memory.updated_at, not datetime.now()."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        content = "The value for key 'k' is: {\"v\": 1}"
        mem = _make_memory_item("m1", "ns", content, updated_at=ts)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = store.get(("ns",), "k")
        assert item is not None
        assert item.updated_at == ts
        assert item.created_at == ts

    def test_get_returns_none_when_not_found(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        item = store.get(("users", "alice"), "missing")
        assert item is None

    def test_get_skips_unparseable_memories(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        memories = [
            _make_memory_item("m1", "ns", "User likes coffee"),
            _make_memory_item(
                "m2", "ns", "The value for key 'k' is: {\"v\": 1}"
            ),
        ]
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        item = store.get(("ns",), "k")
        assert item is not None
        assert item.value == {"v": 1}

    def test_get_legacy_json_format(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """get() handles legacy JSON format memories returned by the service."""
        content = json.dumps({"key": "prefs", "value": {"theme": "dark"}})
        mem = _make_memory_item("m1", "users_alice", content)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = store.get(("users", "alice"), "prefs")
        assert item is not None
        assert item.value == {"theme": "dark"}

    def test_get_wrong_key_returns_none(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        content = "The value for key 'other_key' is: {\"x\": 1}"
        mem = _make_memory_item("m1", "ns", content)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = store.get(("ns",), "target_key")
        assert item is None

    def test_get_http_error_returns_none(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        from azure.core.exceptions import HttpResponseError

        mock_client.beta.memory_stores.search_memories.side_effect = (
            HttpResponseError("not found")
        )

        item = store.get(("ns",), "k")
        assert item is None

    def test_get_uses_underscore_scope_separator(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """Namespace tuple is joined with '_' to form the Azure scope string."""
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        store.get(("users", "alice"), "k")

        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args.kwargs
        assert call_kwargs["scope"] == "users_alice"


class TestBatchPut:
    """Tests for PutOp handling."""

    def test_put_calls_begin_update_memories(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        store.put(("users",), "name", {"val": "Alice"})

        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = (
            mock_client.beta.memory_stores.begin_update_memories.call_args
        )
        assert call_kwargs.kwargs["scope"] == "users"
        assert call_kwargs.kwargs["update_delay"] == 0
        items = call_kwargs.kwargs["items"]
        assert len(items) == 1
        assert "name" in items[0]["content"]

    def test_put_none_calls_delete_scope(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        store.put(("users", "alice"), "prefs", None)

        mock_client.beta.memory_stores.delete_scope.assert_called_once_with(
            name="my-store",
            scope="users_alice",
        )

    def test_put_none_suppresses_delete_error(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mock_client.beta.memory_stores.delete_scope.side_effect = Exception(
            "service error"
        )

        # Should not raise
        store.put(("ns",), "k", None)

    def test_put_http_error_propagates(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        from azure.core.exceptions import HttpResponseError

        mock_client.beta.memory_stores.begin_update_memories.side_effect = (
            HttpResponseError("server error")
        )

        with pytest.raises(HttpResponseError):
            store.put(("ns",), "k", {"a": 1})


class TestBatchSearch:
    """Tests for SearchOp handling."""

    def test_search_returns_parsed_items(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        memories = [
            _make_memory_item(
                "m1",
                "users",
                "The value for key 'prefs' is: {\"theme\": \"dark\"}",
            ),
            _make_memory_item(
                "m2",
                "users",
                "The value for key 'name' is: {\"val\": \"Alice\"}",
            ),
        ]
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        results = store.search(("users",), query="user data")
        assert len(results) == 2
        keys = {r.key for r in results}
        assert "prefs" in keys
        assert "name" in keys

    def test_search_skips_unparseable(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        memories = [
            _make_memory_item("m1", "ns", "User likes dark coffee"),
            _make_memory_item(
                "m2", "ns", "The value for key 'k' is: {\"v\": 1}"
            ),
        ]
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        results = store.search(("ns",), query="k")
        assert len(results) == 1
        assert results[0].key == "k"

    def test_search_http_error_returns_empty(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        from azure.core.exceptions import HttpResponseError

        mock_client.beta.memory_stores.search_memories.side_effect = (
            HttpResponseError("error")
        )

        results = store.search(("ns",), query="q")
        assert results == []

    def test_search_uses_scope_from_namespace_prefix(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        store.search(("users", "alice"), query="q")

        call_kwargs = (
            mock_client.beta.memory_stores.search_memories.call_args
        )
        assert call_kwargs.kwargs["scope"] == "users_alice"

    def test_search_uses_updated_at_timestamp(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchItem timestamps come from memory.updated_at."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        content = "The value for key 'k' is: {\"v\": 1}"
        mem = _make_memory_item("m1", "ns", content, updated_at=ts)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        results = store.search(("ns",), query="k")
        assert len(results) == 1
        assert results[0].updated_at == ts
        assert results[0].created_at == ts

    def test_search_legacy_json_format(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """search() handles legacy JSON format memories from the service."""
        content = json.dumps({"key": "prefs", "value": {"theme": "dark"}})
        mem = _make_memory_item("m1", "users", content)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        results = store.search(("users",), query="preferences")
        assert len(results) == 1
        assert results[0].key == "prefs"
        assert results[0].value == {"theme": "dark"}


class TestBatchListNamespaces:
    """Tests for ListNamespacesOp handling."""

    def test_list_namespaces_returns_empty(
        self, store: AzureAIMemoryStore
    ) -> None:
        result = store.list_namespaces()
        assert result == []


# ---------------------------------------------------------------------------
# Mixed operations
# ---------------------------------------------------------------------------


class TestBatchMixedOps:
    """Tests for batching multiple different ops together."""

    def test_mixed_ops_returns_correct_length(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """batch() returns one result per operation."""
        from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

        ops = [
            PutOp(namespace=("ns",), key="k1", value={"v": 1}),
            GetOp(namespace=("ns",), key="k1"),
            SearchOp(namespace_prefix=("ns",)),
            ListNamespacesOp(),
        ]
        results = store.batch(ops)
        assert len(results) == 4

    def test_unknown_op_raises(self, store: AzureAIMemoryStore) -> None:
        """batch() raises ValueError for unknown op types."""

        class UnknownOp:
            pass

        with pytest.raises(ValueError, match="Unknown operation type"):
            store.batch([UnknownOp()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Async abatch
# ---------------------------------------------------------------------------


class TestAbatch:
    """Tests for the async abatch method."""

    @pytest.mark.asyncio
    async def test_abatch_get(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        content = "The value for key 'k' is: {\"a\": 1}"
        mem = _make_memory_item("m1", "ns", content)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = await store.aget(("ns",), "k")
        assert item is not None
        assert item.value == {"a": 1}

    @pytest.mark.asyncio
    async def test_abatch_put(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        await store.aput(("ns",), "k", {"a": 1})

        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_abatch_search(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        content = json.dumps({"key": "k", "value": {"info": "test"}})
        mem = _make_memory_item("m1", "ns", content)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        from langgraph.store.base import SearchOp

        results = await store.abatch([SearchOp(namespace_prefix=("ns",), query="test")])
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_abatch_list_namespaces(
        self, store: AzureAIMemoryStore
    ) -> None:
        from langgraph.store.base import ListNamespacesOp

        results = await store.abatch([ListNamespacesOp()])
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_abatch_unknown_op_raises(
        self, store: AzureAIMemoryStore
    ) -> None:
        class UnknownOp:
            pass

        with pytest.raises(ValueError, match="Unknown operation type"):
            await store.abatch([UnknownOp()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Stores package import
# ---------------------------------------------------------------------------


class TestStoresPackageImport:
    """Tests for the stores package lazy-import mechanism."""

    def test_import_from_package(self) -> None:
        """AzureAIMemoryStore is importable from the stores package."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langchain_azure_ai.stores import AzureAIMemoryStore as Cls
        assert Cls is AzureAIMemoryStore

    def test_unknown_attribute_raises(self) -> None:
        """Accessing unknown names raises AttributeError."""
        import langchain_azure_ai.stores as stores_pkg

        with pytest.raises(AttributeError):
            _ = stores_pkg.NonExistentClass  # type: ignore[attr-defined]
