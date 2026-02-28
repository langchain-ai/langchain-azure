"""Unit tests for AzureAIMemoryStore."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
)

from langchain_azure_ai.stores.azure_ai_memory import AzureAIMemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory_item(
    memory_id: str,
    scope: str,
    content: str,
    updated_at: datetime | None = None,
) -> MagicMock:
    """Create a mock MemoryItem."""
    item = MagicMock()
    item.memory_id = memory_id
    item.scope = scope
    item.content = content
    item.updated_at = updated_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return item


def _make_search_result(memories: list[MagicMock]) -> MagicMock:
    """Create a mock MemoryStoreSearchResult."""
    result = MagicMock()
    result.memories = [_wrap_search_item(m) for m in memories]
    return result


def _wrap_search_item(memory_item: MagicMock) -> MagicMock:
    """Wrap a MemoryItem in a MemorySearchItem mock."""
    wrapper = MagicMock()
    wrapper.memory_item = memory_item
    return wrapper


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AIProjectClient."""
    client = MagicMock()
    # Default: search returns empty result
    client.beta.memory_stores.search_memories.return_value = _make_search_result([])
    # Default: begin_update_memories returns a poller that completes immediately
    poller = MagicMock()
    poller.result.return_value = MagicMock()
    client.beta.memory_stores.begin_update_memories.return_value = poller
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> AzureAIMemoryStore:
    """Create an AzureAIMemoryStore with a mock client."""
    return AzureAIMemoryStore(
        project_client=mock_client,
        memory_store_name="test-store",
    )


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestAzureAIMemoryStoreInit:
    """Tests for AzureAIMemoryStore initialisation."""

    def test_init_stores_client_and_name(self, mock_client: MagicMock) -> None:
        """Test that the store keeps references to the client and store name."""
        store = AzureAIMemoryStore(
            project_client=mock_client,
            memory_store_name="my-store",
        )
        assert store._client is mock_client
        assert store._memory_store_name == "my-store"

    def test_init_raises_without_sdk(self, mock_client: MagicMock) -> None:
        """Test ImportError is raised when azure-ai-projects is missing."""
        with patch.dict("sys.modules", {"azure.ai.projects": None}):
            with pytest.raises(ImportError, match="azure-ai-projects"):
                AzureAIMemoryStore(
                    project_client=mock_client,
                    memory_store_name="store",
                )

    def test_init_raises_when_client_has_no_beta(self) -> None:
        """Test ValueError when the client lacks beta.memory_stores (V1 SDK)."""
        v1_client = MagicMock(spec=[])  # no attributes at all
        with pytest.raises(ValueError, match="memory stores API"):
            AzureAIMemoryStore(
                project_client=v1_client,
                memory_store_name="store",
            )

    def test_init_raises_when_client_beta_has_no_memory_stores(self) -> None:
        """Test ValueError is raised when beta exists but memory_stores does not."""
        client = MagicMock()
        client.beta = MagicMock(spec=[])  # beta exists but has no memory_stores
        with pytest.raises(ValueError, match="memory stores API"):
            AzureAIMemoryStore(
                project_client=client,
                memory_store_name="store",
            )


# ---------------------------------------------------------------------------
# Tests: namespace ↔ scope helpers
# ---------------------------------------------------------------------------


class TestNamespaceHelpers:
    """Tests for namespace/scope conversion helpers."""

    def test_namespace_to_scope(self, store: AzureAIMemoryStore) -> None:
        """Namespace tuple is joined with '_'."""
        assert store._namespace_to_scope(("users", "alice")) == "users_alice"

    def test_namespace_to_scope_single(self, store: AzureAIMemoryStore) -> None:
        """Single-element namespace produces no separator."""
        assert store._namespace_to_scope(("global",)) == "global"

    def test_scope_to_namespace(self, store: AzureAIMemoryStore) -> None:
        """Scope string is split back to a tuple."""
        assert store._scope_to_namespace("users_alice") == ("users", "alice")

    def test_round_trip(self, store: AzureAIMemoryStore) -> None:
        """Namespace → scope → namespace is a round-trip."""
        ns = ("a", "b", "c")
        assert store._scope_to_namespace(store._namespace_to_scope(ns)) == ns


# ---------------------------------------------------------------------------
# Tests: _make_message / _parse_memory_content
# ---------------------------------------------------------------------------


class TestMessageHelpers:
    """Tests for message serialisation helpers."""

    def test_make_message_format(self, store: AzureAIMemoryStore) -> None:
        """_make_message returns a natural-language message with embedded JSON value."""
        msg = store._make_message("key1", {"x": 1})
        assert msg["role"] == "user"
        assert msg["type"] == "message"
        assert msg["content"] == f"The value for key 'key1' is: {json.dumps({'x': 1})}"

    def test_parse_memory_content_valid(self, store: AzureAIMemoryStore) -> None:
        """Legacy JSON content is parsed correctly."""
        content = json.dumps({"key": "k", "value": {"a": 1}})
        key, value = store._parse_memory_content(content)
        assert key == "k"
        assert value == {"a": 1}

    def test_parse_memory_content_natural_language(
        self, store: AzureAIMemoryStore
    ) -> None:
        """Current natural-language format is parsed correctly."""
        content = f"The value for key 'k' is: {json.dumps({'a': 1})}"
        key, value = store._parse_memory_content(content)
        assert key == "k"
        assert value == {"a": 1}

    def test_parse_memory_content_invalid_json(self, store: AzureAIMemoryStore) -> None:
        """Non-JSON content returns (None, None)."""
        key, value = store._parse_memory_content("free-form text memory")
        assert key is None
        assert value is None

    def test_parse_memory_content_missing_keys(self, store: AzureAIMemoryStore) -> None:
        """JSON without 'key'/'value' fields returns (None, None)."""
        key, value = store._parse_memory_content(json.dumps({"other": "field"}))
        assert key is None
        assert value is None


# ---------------------------------------------------------------------------
# Tests: batch – PutOp
# ---------------------------------------------------------------------------


class TestBatchPutOp:
    """Tests for PutOp handling in batch()."""

    def test_put_calls_begin_update_memories(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """PutOp triggers begin_update_memories with correct args."""
        op = PutOp(
            namespace=("users", "bob"),
            key="prefs",
            value={"theme": "light"},
        )
        results = store.batch([op])
        assert results == [None]
        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = mock_client.beta.memory_stores.begin_update_memories.call_args
        assert call_kwargs.kwargs["name"] == "test-store"
        assert call_kwargs.kwargs["scope"] == "users_bob"
        assert call_kwargs.kwargs["update_delay"] == 0
        items = call_kwargs.kwargs["items"]
        assert len(items) == 1
        content = items[0]["content"]
        expected = f"The value for key 'prefs' is: {json.dumps({'theme': 'light'})}"
        assert content == expected

    def test_put_waits_for_poller(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """PutOp calls .result() on the returned poller."""
        poller = MagicMock()
        mock_client.beta.memory_stores.begin_update_memories.return_value = poller
        store.batch([PutOp(namespace=("ns",), key="k", value={"v": 1})])
        poller.result.assert_called_once()

    def test_put_none_value_calls_delete_scope(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """PutOp with value=None calls delete_scope instead of storing."""
        op = PutOp(namespace=("users", "alice"), key="prefs", value=None)
        results = store.batch([op])
        assert results == [None]
        mock_client.beta.memory_stores.delete_scope.assert_called_once_with(
            name="test-store",
            scope="users_alice",
        )
        mock_client.beta.memory_stores.begin_update_memories.assert_not_called()

    def test_put_none_delete_scope_exception_is_suppressed(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """delete_scope exceptions during a delete PutOp are suppressed."""
        mock_client.beta.memory_stores.delete_scope.side_effect = RuntimeError("boom")
        op = PutOp(namespace=("ns",), key="k", value=None)
        # Should not raise
        store.batch([op])


# ---------------------------------------------------------------------------
# Tests: batch – GetOp
# ---------------------------------------------------------------------------


class TestBatchGetOp:
    """Tests for GetOp handling in batch()."""

    def test_get_returns_none_when_no_match(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """GetOp returns None when no matching memory is found."""
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([])
        )
        results = store.batch([GetOp(namespace=("ns",), key="missing")])
        assert results == [None]

    def test_get_returns_item_on_match(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """GetOp returns an Item when a matching memory is found (current format)."""
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        memory = _make_memory_item(
            memory_id="m1",
            scope="users_alice",
            content=f"The value for key 'prefs' is: {json.dumps({'theme': 'dark'})}",
            updated_at=ts,
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = store.batch([GetOp(namespace=("users", "alice"), key="prefs")])
        assert len(results) == 1
        item = results[0]
        assert isinstance(item, Item)
        assert item.key == "prefs"
        assert item.value == {"theme": "dark"}
        assert item.namespace == ("users", "alice")
        assert item.updated_at == ts

    def test_get_returns_item_on_match_legacy_format(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """GetOp returns an Item when memory content uses the legacy JSON format."""
        memory = _make_memory_item(
            memory_id="m1",
            scope="users_alice",
            content=json.dumps({"key": "prefs", "value": {"theme": "dark"}}),
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = store.batch([GetOp(namespace=("users", "alice"), key="prefs")])
        item = results[0]
        assert isinstance(item, Item)
        assert item.key == "prefs"
        assert item.value == {"theme": "dark"}

    def test_get_skips_non_matching_key(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """GetOp returns None when memory content has a different key."""
        memory = _make_memory_item(
            memory_id="m1",
            scope="ns",
            content=f"The value for key 'other' is: {json.dumps({})}",
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = store.batch([GetOp(namespace=("ns",), key="wanted")])
        assert results == [None]

    def test_get_passes_correct_scope_and_query(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """GetOp passes the correct scope and a matching query to search_memories."""
        store.batch([GetOp(namespace=("a", "b"), key="x")])
        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args.kwargs
        assert call_kwargs["scope"] == "a_b"
        assert call_kwargs["items"][0]["content"] == "The value for key 'x'"


# ---------------------------------------------------------------------------
# Tests: batch – SearchOp
# ---------------------------------------------------------------------------


class TestBatchSearchOp:
    """Tests for SearchOp handling in batch()."""

    def test_search_returns_empty_when_no_results(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp returns an empty list when no memories are found."""
        results = store.batch([SearchOp(namespace_prefix=("ns",), query="hello")])
        assert results == [[]]

    def test_search_returns_search_items(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp converts memory results to SearchItem objects."""
        ts = datetime(2024, 3, 1, tzinfo=timezone.utc)
        memory = _make_memory_item(
            memory_id="m1",
            scope="ns",
            content=json.dumps({"key": "fact1", "value": {"info": "hello"}}),
            updated_at=ts,
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = store.batch([SearchOp(namespace_prefix=("ns",), query="hello")])
        assert len(results) == 1
        search_results = results[0]
        assert len(search_results) == 1
        si = search_results[0]
        assert isinstance(si, SearchItem)
        assert si.key == "fact1"
        assert si.value == {"info": "hello"}
        assert si.namespace == ("ns",)

    def test_search_filters_by_namespace_prefix(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp filters out memories whose namespace doesn't match the prefix."""
        m1 = _make_memory_item(
            "m1",
            "ns_sub",
            json.dumps({"key": "k1", "value": {"x": 1}}),
        )
        m2 = _make_memory_item(
            "m2",
            "other_path",
            json.dumps({"key": "k2", "value": {"x": 2}}),
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([m1, m2])
        )
        results = store.batch([SearchOp(namespace_prefix=("ns",))])
        assert len(results[0]) == 1
        assert results[0][0].key == "k1"

    def test_search_applies_filter(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp applies value-level filter to results."""
        m1 = _make_memory_item(
            "m1", "ns", json.dumps({"key": "k1", "value": {"color": "red"}})
        )
        m2 = _make_memory_item(
            "m2", "ns", json.dumps({"key": "k2", "value": {"color": "blue"}})
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([m1, m2])
        )
        results = store.batch(
            [SearchOp(namespace_prefix=("ns",), filter={"color": "red"})]
        )
        assert len(results[0]) == 1
        assert results[0][0].key == "k1"

    def test_search_raw_content_fallback(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp falls back to raw content when content is not JSON."""
        memory = _make_memory_item("m1", "ns", "user likes dark mode")
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = store.batch([SearchOp(namespace_prefix=("ns",))])
        assert len(results[0]) == 1
        si = results[0][0]
        assert si.value == {"content": "user likes dark mode"}
        assert si.key == "m1"

    def test_search_respects_limit_and_offset(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp respects limit and offset parameters."""
        memories = [
            _make_memory_item(f"m{i}", "ns", json.dumps({"key": f"k{i}", "value": {}}))
            for i in range(5)
        ]
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result(memories)
        )
        results = store.batch(
            [SearchOp(namespace_prefix=("ns",), limit=2, offset=1)]
        )
        assert len(results[0]) == 2
        assert results[0][0].key == "k1"
        assert results[0][1].key == "k2"


# ---------------------------------------------------------------------------
# Tests: batch – ListNamespacesOp
# ---------------------------------------------------------------------------


class TestBatchListNamespacesOp:
    """Tests for ListNamespacesOp handling in batch()."""

    def test_list_namespaces_returns_empty(self, store: AzureAIMemoryStore) -> None:
        """ListNamespacesOp always returns an empty list."""
        results = store.batch([ListNamespacesOp()])
        assert results == [[]]


# ---------------------------------------------------------------------------
# Tests: batch – mixed ops
# ---------------------------------------------------------------------------


class TestBatchMixedOps:
    """Tests for batching multiple different ops together."""

    def test_mixed_ops_returns_correct_length(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """batch() returns one result per operation."""
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
# Tests: abatch
# ---------------------------------------------------------------------------


class TestAbatch:
    """Tests for the async abatch() method."""

    @pytest.mark.asyncio
    async def test_abatch_put(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """abatch() handles PutOp correctly."""
        results = await store.abatch(
            [PutOp(namespace=("ns",), key="k", value={"x": 1})]
        )
        assert results == [None]
        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_abatch_get_returns_item(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """abatch() handles GetOp and returns an Item."""
        memory = _make_memory_item(
            "m1",
            "ns",
            json.dumps({"key": "k", "value": {"z": 99}}),
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = await store.abatch([GetOp(namespace=("ns",), key="k")])
        assert len(results) == 1
        assert results[0].value == {"z": 99}  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_abatch_search(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """abatch() handles SearchOp and returns SearchItems."""
        memory = _make_memory_item(
            "m1",
            "ns",
            json.dumps({"key": "k", "value": {"info": "test"}}),
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = await store.abatch(
            [SearchOp(namespace_prefix=("ns",), query="test")]
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_abatch_list_namespaces(
        self, store: AzureAIMemoryStore
    ) -> None:
        """abatch() handles ListNamespacesOp and returns empty list."""
        results = await store.abatch([ListNamespacesOp()])
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_abatch_unknown_op_raises(
        self, store: AzureAIMemoryStore
    ) -> None:
        """abatch() raises ValueError for unknown op types."""

        class UnknownOp:
            pass

        with pytest.raises(ValueError, match="Unknown operation type"):
            await store.abatch([UnknownOp()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Tests: lazy import from stores package
# ---------------------------------------------------------------------------


class TestStoresPackageImport:
    """Tests for the stores package lazy-import mechanism."""

    def test_import_from_package(self) -> None:
        """AzureAIMemoryStore is importable from the stores package."""
        from langchain_azure_ai.stores import AzureAIMemoryStore as Cls

        assert Cls is AzureAIMemoryStore

    def test_unknown_attribute_raises(self) -> None:
        """Accessing unknown names raises AttributeError."""
        import langchain_azure_ai.stores as stores_pkg

        with pytest.raises(AttributeError):
            _ = stores_pkg.NonExistentClass  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests: local in-process cache
# ---------------------------------------------------------------------------


class TestLocalCache:
    """Tests for the local in-process cache used by put/get/search/list_namespaces."""

    def test_cache_initialised_empty(self, store: AzureAIMemoryStore) -> None:
        """The local cache is empty on construction."""
        assert store._cache == {}

    def test_put_populates_cache(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """PutOp populates the local cache with the stored value."""
        store.batch(
            [PutOp(namespace=("users", "alice"), key="prefs", value={"x": 1})]
        )
        assert ("users", "alice") in store._cache
        assert store._cache[("users", "alice")]["prefs"]["value"] == {"x": 1}

    def test_put_preserves_created_at_on_update(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """Updating an existing key keeps the original created_at timestamp."""
        store.batch(
            [PutOp(namespace=("ns",), key="k", value={"v": 1})]
        )
        first_created = store._cache[("ns",)]["k"]["created_at"]
        store.batch(
            [PutOp(namespace=("ns",), key="k", value={"v": 2})]
        )
        assert store._cache[("ns",)]["k"]["created_at"] == first_created
        assert store._cache[("ns",)]["k"]["value"] == {"v": 2}

    def test_get_served_from_cache_no_api_call(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """GetOp hits the cache and does NOT call search_memories."""
        store.batch(
            [PutOp(namespace=("ns",), key="k", value={"a": 1})]
        )
        mock_client.beta.memory_stores.search_memories.reset_mock()
        results = store.batch([GetOp(namespace=("ns",), key="k")])
        mock_client.beta.memory_stores.search_memories.assert_not_called()
        assert isinstance(results[0], Item)
        assert results[0].key == "k"
        assert results[0].value == {"a": 1}

    def test_get_from_cache_after_put(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """put() then get() in the same session returns the stored value."""
        store.batch(
            [
                PutOp(
                    namespace=("users", "alice"),
                    key="preferences",
                    value={"theme": "dark", "coffee": "dark roast"},
                )
            ]
        )
        results = store.batch(
            [GetOp(namespace=("users", "alice"), key="preferences")]
        )
        item = results[0]
        assert isinstance(item, Item)
        assert item.key == "preferences"
        assert item.value == {"theme": "dark", "coffee": "dark roast"}
        assert item.namespace == ("users", "alice")

    def test_delete_removes_key_from_cache(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """PutOp with value=None removes the specific key from the cache."""
        store.batch(
            [
                PutOp(namespace=("ns",), key="k1", value={"x": 1}),
                PutOp(namespace=("ns",), key="k2", value={"x": 2}),
            ]
        )
        assert "k1" in store._cache[("ns",)]
        assert "k2" in store._cache[("ns",)]

        store.batch([PutOp(namespace=("ns",), key="k1", value=None)])
        assert "k1" not in store._cache.get(("ns",), {})
        assert "k2" in store._cache[("ns",)]

    def test_delete_last_key_removes_namespace_from_cache(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """Deleting the last key in a namespace removes the namespace entry."""
        store.batch([PutOp(namespace=("ns",), key="only", value={"x": 1})])
        store.batch([PutOp(namespace=("ns",), key="only", value=None)])
        assert ("ns",) not in store._cache

    def test_delete_with_remaining_keys_readds_them(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """Deleting one key re-adds remaining keys to Azure AI for semantic search."""
        mock_client.beta.memory_stores.begin_update_memories.reset_mock()
        store.batch(
            [
                PutOp(namespace=("ns",), key="k1", value={"x": 1}),
                PutOp(namespace=("ns",), key="k2", value={"x": 2}),
            ]
        )
        mock_client.beta.memory_stores.begin_update_memories.reset_mock()
        store.batch([PutOp(namespace=("ns",), key="k1", value=None)])
        # k2 should have been re-added to Azure AI after scope deletion
        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = (
            mock_client.beta.memory_stores.begin_update_memories.call_args.kwargs
        )
        assert "k2" in call_kwargs["items"][0]["content"]


# ---------------------------------------------------------------------------
# Tests: _matches_query
# ---------------------------------------------------------------------------


class TestMatchesQuery:
    """Tests for the _matches_query keyword-matching helper."""

    def test_empty_query_always_matches(self, store: AzureAIMemoryStore) -> None:
        """An empty query string matches every entry."""
        assert store._matches_query("key", {"x": 1}, "") is True

    def test_query_keyword_in_value_matches(self, store: AzureAIMemoryStore) -> None:
        """A keyword present in the serialised value matches."""
        assert store._matches_query(
            "preferences", {"coffee": "dark roast"}, "What coffee does Alice prefer?"
        ) is True

    def test_query_keyword_in_key_matches(self, store: AzureAIMemoryStore) -> None:
        """A keyword present in the key matches."""
        assert store._matches_query(
            "preferences", {"x": 1}, "user preferences"
        ) is True

    def test_query_no_match(self, store: AzureAIMemoryStore) -> None:
        """Keywords not present in key or value return False."""
        assert store._matches_query(
            "profile", {"name": "Alice"}, "weather forecast"
        ) is False

    def test_short_words_ignored(self, store: AzureAIMemoryStore) -> None:
        """Query words shorter than 3 characters are skipped (no match needed)."""
        # "is" (2 chars) is the only word; since no words >= 3 chars, returns True
        assert store._matches_query("k", {"v": 1}, "is") is True


# ---------------------------------------------------------------------------
# Tests: search with local cache
# ---------------------------------------------------------------------------


class TestSearchWithCache:
    """Tests for SearchOp when the local cache has entries."""

    def test_search_returns_cache_entry_matching_query(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp returns the cache entry whose value matches the query keyword."""
        store.batch(
            [
                PutOp(
                    namespace=("users", "alice"),
                    key="preferences",
                    value={"coffee": "dark roast", "theme": "dark"},
                ),
                PutOp(
                    namespace=("users", "alice"),
                    key="profile",
                    value={"location": "Seattle", "role": "engineer"},
                ),
            ]
        )
        results = store.batch(
            [
                SearchOp(
                    namespace_prefix=("users", "alice"),
                    query="What coffee does Alice prefer?",
                )
            ]
        )
        keys = [item.key for item in results[0]]
        # "coffee" keyword matches 'preferences' but not 'profile'
        assert "preferences" in keys

    def test_search_no_query_returns_all_cache_entries(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp without query returns all cache entries for the namespace."""
        store.batch(
            [
                PutOp(namespace=("ns",), key="a", value={"x": 1}),
                PutOp(namespace=("ns",), key="b", value={"x": 2}),
            ]
        )
        results = store.batch([SearchOp(namespace_prefix=("ns",))])
        keys = {item.key for item in results[0]}
        assert keys == {"a", "b"}

    def test_search_cache_deduplicates_with_ai_results(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """AI results for keys already in the cache are deduplicated."""
        store.batch(
            [PutOp(namespace=("ns",), key="k1", value={"color": "red"})]
        )
        # Azure AI also returns k1 – should NOT appear twice in the result.
        memory = _make_memory_item(
            "m1",
            "ns",
            json.dumps({"key": "k1", "value": {"color": "red"}}),
        )
        mock_client.beta.memory_stores.search_memories.return_value = (
            _make_search_result([memory])
        )
        results = store.batch([SearchOp(namespace_prefix=("ns",))])
        assert len(results[0]) == 1
        assert results[0][0].key == "k1"

    def test_search_cache_filter_applied(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """SearchOp filter is applied to cache entries."""
        store.batch(
            [
                PutOp(namespace=("ns",), key="a", value={"color": "red"}),
                PutOp(namespace=("ns",), key="b", value={"color": "blue"}),
            ]
        )
        results = store.batch(
            [SearchOp(namespace_prefix=("ns",), filter={"color": "red"})]
        )
        assert len(results[0]) == 1
        assert results[0][0].key == "a"


# ---------------------------------------------------------------------------
# Tests: list_namespaces with local cache
# ---------------------------------------------------------------------------


class TestListNamespacesWithCache:
    """Tests for ListNamespacesOp when the local cache has entries."""

    def test_list_namespaces_after_put(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """list_namespaces() returns namespaces populated via put()."""
        store.batch(
            [
                PutOp(
                    namespace=("users", "alice"),
                    key="prefs",
                    value={"x": 1},
                ),
                PutOp(
                    namespace=("users", "alice"),
                    key="profile",
                    value={"y": 2},
                ),
            ]
        )
        results = store.batch([ListNamespacesOp()])
        assert ("users", "alice") in results[0]

    def test_list_namespaces_multiple_namespaces(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """list_namespaces() returns all distinct namespaces."""
        store.batch(
            [
                PutOp(namespace=("users", "alice"), key="k", value={}),
                PutOp(namespace=("users", "bob"), key="k", value={}),
            ]
        )
        results = store.batch([ListNamespacesOp()])
        ns_set = set(results[0])
        assert ("users", "alice") in ns_set
        assert ("users", "bob") in ns_set

    def test_list_namespaces_prefix_condition(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """list_namespaces() with a prefix match_condition filters correctly."""
        store.batch(
            [
                PutOp(namespace=("users", "alice"), key="k", value={}),
                PutOp(namespace=("docs", "alice"), key="k", value={}),
            ]
        )
        op = ListNamespacesOp(
            match_conditions=(
                MatchCondition(match_type="prefix", path=("users",)),
            )
        )
        results = store.batch([op])
        assert results[0] == [("users", "alice")]

    def test_list_namespaces_suffix_condition(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """list_namespaces() with a suffix match_condition filters correctly."""
        store.batch(
            [
                PutOp(namespace=("users", "alice"), key="k", value={}),
                PutOp(namespace=("users", "bob"), key="k", value={}),
            ]
        )
        op = ListNamespacesOp(
            match_conditions=(
                MatchCondition(match_type="suffix", path=("alice",)),
            )
        )
        results = store.batch([op])
        assert results[0] == [("users", "alice")]

    def test_list_namespaces_max_depth(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """list_namespaces() with max_depth truncates and deduplicates."""
        store.batch(
            [
                PutOp(namespace=("users", "alice", "prefs"), key="k", value={}),
                PutOp(namespace=("users", "bob", "prefs"), key="k", value={}),
            ]
        )
        op = ListNamespacesOp(max_depth=1)
        results = store.batch([op])
        assert results[0] == [("users",)]

    def test_list_namespaces_limit_and_offset(
        self, store: AzureAIMemoryStore, mock_client: MagicMock
    ) -> None:
        """list_namespaces() respects limit and offset."""
        for i in range(5):
            store.batch(
                [PutOp(namespace=(f"ns{i}",), key="k", value={})]
            )
        results = store.batch([ListNamespacesOp(limit=2, offset=1)])
        assert len(results[0]) == 2
