"""Unit tests for AzureAIMemoryStore."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, List, Optional
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_azure_ai.stores.memory.azure_ai_memory import AzureAIMemoryStore


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


@pytest.fixture(autouse=True)
def patch_memory_search_options() -> Any:
    """Patch MemorySearchOptions so tests work without azure-ai-projects>=2.0.0b4."""
    import sys

    models_mod = sys.modules.get("azure.ai.projects.models")
    if models_mod is not None and not hasattr(models_mod, "MemorySearchOptions"):
        mock_cls = MagicMock(return_value=MagicMock())
        models_mod.MemorySearchOptions = mock_cls  # type: ignore[attr-defined]
        yield
        try:
            del models_mod.MemorySearchOptions  # type: ignore[attr-defined]
        except AttributeError:
            pass
    else:
        yield


@pytest.fixture
def store(mock_client: MagicMock) -> AzureAIMemoryStore:
    """Create an AzureAIMemoryStore with a patched AIProjectClient."""
    import azure.ai.projects as _azmod

    with patch.object(_azmod, "AIProjectClient", return_value=mock_client):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return AzureAIMemoryStore(
                project_endpoint="https://example.azure.com/api/projects/proj",
                memory_store_name="my-store",
            )


# ---------------------------------------------------------------------------
# _make_message
# ---------------------------------------------------------------------------


class TestMakeMessage:
    """Tests for the static _make_message helper."""

    def test_structure(self) -> None:
        msg = AzureAIMemoryStore._make_message("hello world")
        assert msg["role"] == "user"
        assert msg["type"] == "message"
        assert msg["content"] == "hello world"

    def test_content_passed_through(self) -> None:
        msg = AzureAIMemoryStore._make_message("user prefers dark theme")
        assert msg["content"] == "user prefers dark theme"


# ---------------------------------------------------------------------------
# _namespace_to_scope
# ---------------------------------------------------------------------------


class TestNamespaceToScope:
    def test_single(self) -> None:
        assert AzureAIMemoryStore._namespace_to_scope(("users",)) == "users"

    def test_multi_element_raises(self) -> None:
        """Multi-element namespaces are not supported and raise ValueError."""
        with pytest.raises(ValueError, match="exactly one element"):
            AzureAIMemoryStore._namespace_to_scope(("users", "alice"))

    def test_empty_raises(self) -> None:
        """Empty namespaces raise ValueError."""
        with pytest.raises(ValueError, match="exactly one element"):
            AzureAIMemoryStore._namespace_to_scope(())


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for AzureAIMemoryStore.__init__."""

    def test_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="project_endpoint"):
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

    def test_explicit_project_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        import azure.ai.projects as _azmod

        with (
            warnings.catch_warnings(),
            patch.object(_azmod, "AIProjectClient", return_value=MagicMock()),
        ):
            warnings.simplefilter("ignore")
            store = AzureAIMemoryStore(
                memory_store_name="ms",
                project_endpoint="https://example.services.ai.azure.com/api/projects/p",
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
                project_endpoint="https://example.azure.com/api/projects/p",
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
                project_endpoint="https://example.azure.com/api/projects/p",
            )
        call_kwargs = mock_ctor.call_args.kwargs
        assert call_kwargs.get("user_agent") == "langchain-azure-ai"


# ---------------------------------------------------------------------------
# batch operations (mocked client)
# ---------------------------------------------------------------------------


class TestBatchGet:
    """Tests for GetOp handling."""

    def test_get_returns_item_when_found(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mem = _make_memory_item("m1", "user_alice", "prefers dark theme")
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = store.get(("user_alice",), "content")
        assert item is not None
        assert item.key == "content"
        assert item.value == {"content": "prefers dark theme"}
        assert item.namespace == ("user_alice",)

    def test_get_uses_updated_at_timestamp(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """Item timestamps come from memory.updated_at, not datetime.now()."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        mem = _make_memory_item("m1", "ns", "some content", updated_at=ts)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = store.get(("ns",), "content")
        assert item is not None
        assert item.updated_at == ts
        assert item.created_at == ts

    def test_get_returns_none_when_not_found(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        item = store.get(("user_alice",), "content")
        assert item is None

    def test_get_http_error_returns_none(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        from azure.core.exceptions import HttpResponseError

        mock_client.beta.memory_stores.search_memories.side_effect = HttpResponseError(
            "not found"
        )

        item = store.get(("ns",), "content")
        assert item is None

    def test_get_uses_key_as_query(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """The LangGraph key is used as the search query text."""
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        store.get(("ns",), "user_preferences")

        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args.kwargs
        assert call_kwargs["items"][0]["content"] == "user_preferences"

    def test_get_multi_element_namespace_raises(
        self, store: AzureAIMemoryStore
    ) -> None:
        """get() raises ValueError for multi-element namespaces."""
        with pytest.raises(ValueError, match="exactly one element"):
            store.get(("users", "alice"), "content")


class TestBatchPut:
    """Tests for PutOp handling."""

    def test_put_stores_content_value(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        store.put(("user_alice",), "content", {"content": "prefers dark theme"})

        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = mock_client.beta.memory_stores.begin_update_memories.call_args
        assert call_kwargs.kwargs["scope"] == "user_alice"
        assert call_kwargs.kwargs["update_delay"] == 0
        items = call_kwargs.kwargs["items"]
        assert len(items) == 1
        assert items[0]["content"] == "prefers dark theme"
        assert items[0]["role"] == "user"

    def test_put_missing_content_key_raises(self, store: AzureAIMemoryStore) -> None:
        """put() raises ValueError when value lacks a 'content' key."""
        with pytest.raises(ValueError, match="'content' key"):
            store.put(("ns",), "k", {"theme": "dark"})

    def test_put_none_calls_delete_scope(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        store.put(("user_alice",), "content", None)  # type: ignore[arg-type]

        mock_client.beta.memory_stores.delete_scope.assert_called_once_with(
            name="my-store",
            scope="user_alice",
        )

    def test_put_none_suppresses_delete_error(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mock_client.beta.memory_stores.delete_scope.side_effect = Exception(
            "service error"
        )

        # Should not raise
        store.put(("ns",), "content", None)  # type: ignore[arg-type]

    def test_put_http_error_propagates(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        from azure.core.exceptions import HttpResponseError

        mock_client.beta.memory_stores.begin_update_memories.side_effect = (
            HttpResponseError("server error")
        )

        with pytest.raises(HttpResponseError):
            store.put(("ns",), "content", {"content": "some text"})

    def test_put_multi_element_namespace_raises(
        self, store: AzureAIMemoryStore
    ) -> None:
        """put() raises ValueError for multi-element namespaces."""
        with pytest.raises(ValueError, match="exactly one element"):
            store.put(("users", "alice"), "content", {"content": "text"})


class TestBatchSearch:
    """Tests for SearchOp handling."""

    def test_search_returns_content_items(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        memories = [
            _make_memory_item("m1", "users", "prefers dark theme"),
            _make_memory_item("m2", "users", "speaks English"),
        ]
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        results = store.search(("users",), query="user data")
        assert len(results) == 2
        assert all(r.key == "content" for r in results)
        contents = {r.value["content"] for r in results}
        assert "prefers dark theme" in contents
        assert "speaks English" in contents

    def test_search_all_memories_returned(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """All memories (including AI-synthesised ones) are returned as-is."""
        memories = [
            _make_memory_item("m1", "ns", "User likes dark coffee"),
            _make_memory_item("m2", "ns", "prefers Python over Java"),
        ]
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )

        results = store.search(("ns",), query="preferences")
        assert len(results) == 2

    def test_search_http_error_returns_empty(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        from azure.core.exceptions import HttpResponseError

        mock_client.beta.memory_stores.search_memories.side_effect = HttpResponseError(
            "error"
        )

        results = store.search(("ns",), query="q")
        assert results == []

    def test_search_uses_scope_from_namespace_prefix(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )

        store.search(("users",), query="q")

        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args
        assert call_kwargs.kwargs["scope"] == "users"

    def test_search_uses_updated_at_timestamp(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchItem timestamps come from memory.updated_at."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        mem = _make_memory_item("m1", "ns", "some content", updated_at=ts)
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        results = store.search(("ns",), query="k")
        assert len(results) == 1
        assert results[0].updated_at == ts
        assert results[0].created_at == ts

    def test_search_multi_element_namespace_raises(
        self, store: AzureAIMemoryStore
    ) -> None:
        """search() raises ValueError for multi-element namespaces."""
        with pytest.raises(ValueError, match="exactly one element"):
            store.search(("users", "alice"), query="q")


class TestBatchListNamespaces:
    """Tests for ListNamespacesOp handling."""

    def test_list_namespaces_returns_empty(self, store: AzureAIMemoryStore) -> None:
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
            PutOp(namespace=("ns",), key="content", value={"content": "hello"}),
            GetOp(namespace=("ns",), key="content"),
            SearchOp(namespace_prefix=("ns",)),
            ListNamespacesOp(),
        ]
        results = store.batch(ops)  # type: ignore[arg-type]
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
        mem = _make_memory_item("m1", "ns", "some info")
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        item = await store.aget(("ns",), "content")
        assert item is not None
        assert item.value == {"content": "some info"}

    @pytest.mark.asyncio
    async def test_abatch_put(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        await store.aput(("ns",), "content", {"content": "hello"})

        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_abatch_search(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        mem = _make_memory_item("m1", "ns", "some info")
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        from langgraph.store.base import SearchOp

        results = await store.abatch([SearchOp(namespace_prefix=("ns",), query="test")])
        assert len(results[0]) == 1  # type: ignore[arg-type]
        assert results[0][0].value == {"content": "some info"}  # type: ignore[arg-type,index,union-attr]

    @pytest.mark.asyncio
    async def test_abatch_list_namespaces(self, store: AzureAIMemoryStore) -> None:
        from langgraph.store.base import ListNamespacesOp

        results = await store.abatch([ListNamespacesOp()])
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_abatch_unknown_op_raises(self, store: AzureAIMemoryStore) -> None:
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
            from langchain_azure_ai.stores.memory import AzureAIMemoryStore as Cls
        assert Cls is AzureAIMemoryStore

    def test_unknown_attribute_raises(self) -> None:
        """Accessing unknown names raises AttributeError."""
        import langchain_azure_ai.stores.memory as stores_pkg

        with pytest.raises(AttributeError):
            _ = stores_pkg.NonExistentClass  # type: ignore[attr-defined]
