"""Unit tests for AzureAIMemoryStore."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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

from langchain_azure_ai.stores.azure_ai_memory import (
    AzureAIMemoryStore,
    _build_message_items,
    _does_match,
    _memory_item_to_search_item,
    _memory_item_to_store_item,
    _namespace_to_scope,
    _scope_to_namespace,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_memory_item(
    memory_id: str = "mem-1",
    content: str = "The user likes Python",
    scope: str = "user/123",
    kind: str = "user_profile",
    updated_at: datetime | None = None,
) -> MagicMock:
    item = MagicMock()
    item.memory_id = memory_id
    item.content = content
    item.scope = scope
    item.kind = kind
    item.updated_at = updated_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    return item


def _make_search_result(
    memory_items: list[Any] | None = None,
    search_id: str = "search-1",
) -> MagicMock:
    result = MagicMock()
    result.search_id = search_id
    memories = []
    for mem in (memory_items or []):
        search_item = MagicMock()
        search_item.memory_item = mem
        memories.append(search_item)
    result.memories = memories
    return result


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestNamespaceConversion:
    def test_namespace_to_scope(self) -> None:
        assert _namespace_to_scope(("user", "123")) == "user/123"

    def test_namespace_to_scope_single(self) -> None:
        assert _namespace_to_scope(("user123",)) == "user123"

    def test_namespace_to_scope_custom_delimiter(self) -> None:
        assert _namespace_to_scope(("a", "b", "c"), ".") == "a.b.c"

    def test_scope_to_namespace(self) -> None:
        assert _scope_to_namespace("user/123") == ("user", "123")

    def test_scope_to_namespace_single(self) -> None:
        assert _scope_to_namespace("user123") == ("user123",)

    def test_scope_to_namespace_custom_delimiter(self) -> None:
        assert _scope_to_namespace("a.b.c", ".") == ("a", "b", "c")


class TestMemoryItemConversion:
    def test_to_store_item(self) -> None:
        mem = _make_memory_item()
        item = _memory_item_to_store_item(mem, ("user", "123"))
        assert isinstance(item, Item)
        assert item.key == "mem-1"
        assert item.namespace == ("user", "123")
        assert item.value["content"] == "The user likes Python"
        assert item.value["kind"] == "user_profile"

    def test_to_search_item(self) -> None:
        mem = _make_memory_item()
        item = _memory_item_to_search_item(mem, ("user", "123"), score=0.95)
        assert isinstance(item, SearchItem)
        assert item.key == "mem-1"
        assert item.score == 0.95

    def test_to_store_item_naive_datetime(self) -> None:
        mem = _make_memory_item(updated_at=datetime(2025, 6, 1))
        item = _memory_item_to_store_item(mem, ("user",))
        assert item.updated_at.tzinfo is not None


class TestBuildMessageItems:
    def test_with_content_string(self) -> None:
        items = _build_message_items({"content": "Hello"})
        assert len(items) == 1

    def test_with_items_key(self) -> None:
        pre_built = [MagicMock(), MagicMock()]
        items = _build_message_items({"items": pre_built})
        assert items == pre_built

    def test_empty_content(self) -> None:
        assert _build_message_items({"content": ""}) == []

    def test_no_content_or_items(self) -> None:
        assert _build_message_items({}) == []


class TestDoesMatch:
    def test_prefix_match(self) -> None:
        cond = MatchCondition(match_type="prefix", path=("user",))
        assert _does_match(cond, ("user", "123"))

    def test_prefix_no_match(self) -> None:
        cond = MatchCondition(match_type="prefix", path=("admin",))
        assert not _does_match(cond, ("user", "123"))

    def test_suffix_match(self) -> None:
        cond = MatchCondition(match_type="suffix", path=("123",))
        assert _does_match(cond, ("user", "123"))

    def test_wildcard_prefix(self) -> None:
        cond = MatchCondition(match_type="prefix", path=("*", "123"))
        assert _does_match(cond, ("user", "123"))

    def test_key_shorter_than_path(self) -> None:
        cond = MatchCondition(match_type="prefix", path=("a", "b", "c"))
        assert not _does_match(cond, ("a",))


# ---------------------------------------------------------------------------
# AzureAIMemoryStore tests
# ---------------------------------------------------------------------------

class TestAzureAIMemoryStoreInit:
    def test_init(self) -> None:
        client = MagicMock()
        store = AzureAIMemoryStore(
            project_client=client,
            memory_store_name="test-store",
        )
        assert store._memory_store_name == "test-store"
        assert store._client is client
        assert store._async_client is None


class TestAzureAIMemoryStoreBatch:
    def _make_store(self) -> tuple[AzureAIMemoryStore, MagicMock]:
        client = MagicMock()
        store = AzureAIMemoryStore(
            project_client=client,
            memory_store_name="test-store",
        )
        return store, client

    def test_get_op_found(self) -> None:
        store, client = self._make_store()
        mem = _make_memory_item(memory_id="key1", scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])

        results = store.batch([GetOp(("user",), "key1")])
        assert len(results) == 1
        assert isinstance(results[0], Item)
        assert results[0].key == "key1"

    def test_get_op_not_found(self) -> None:
        store, client = self._make_store()
        mem = _make_memory_item(memory_id="other", scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])

        results = store.batch([GetOp(("user",), "key1")])
        assert results[0] is None

    def test_get_op_exception(self) -> None:
        store, client = self._make_store()
        client.memory_stores.search_memories.side_effect = Exception("fail")

        results = store.batch([GetOp(("user",), "key1")])
        assert results[0] is None

    def test_search_op(self) -> None:
        store, client = self._make_store()
        mem = _make_memory_item(scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])

        results = store.batch([SearchOp(("user",), None, 10, 0, "hobbies")])
        assert len(results) == 1
        items = results[0]
        assert len(items) == 1
        assert isinstance(items[0], SearchItem)

    def test_search_op_with_filter(self) -> None:
        store, client = self._make_store()
        mem1 = _make_memory_item(memory_id="m1", kind="user_profile", scope="user")
        mem2 = _make_memory_item(memory_id="m2", kind="chat_summary", scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result(
            [mem1, mem2]
        )

        results = store.batch(
            [SearchOp(("user",), {"kind": "user_profile"}, 10, 0, "query")]
        )
        items = results[0]
        assert len(items) == 1
        assert items[0].value["kind"] == "user_profile"

    def test_search_op_with_pagination(self) -> None:
        store, client = self._make_store()
        mems = [
            _make_memory_item(memory_id=f"m{i}", scope="user") for i in range(5)
        ]
        client.memory_stores.search_memories.return_value = _make_search_result(mems)

        results = store.batch([SearchOp(("user",), None, 2, 1, "query")])
        items = results[0]
        assert len(items) == 2

    def test_search_op_exception(self) -> None:
        store, client = self._make_store()
        client.memory_stores.search_memories.side_effect = Exception("fail")

        results = store.batch([SearchOp(("user",), None, 10, 0, "query")])
        assert results[0] == []

    def test_put_op_update(self) -> None:
        store, client = self._make_store()
        poller = MagicMock()
        client.memory_stores.begin_update_memories.return_value = poller

        store.batch(
            [PutOp(("user",), "key1", {"content": "I like hiking"})]
        )

        client.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = client.memory_stores.begin_update_memories.call_args
        assert call_kwargs.kwargs["scope"] == "user"
        assert call_kwargs.kwargs["update_delay"] == 0
        poller.result.assert_called_once()

    def test_put_op_delete(self) -> None:
        store, client = self._make_store()

        store.batch([PutOp(("user",), "key1", None)])

        client.memory_stores.delete_scope.assert_called_once()
        call_kwargs = client.memory_stores.delete_scope.call_args
        assert call_kwargs.kwargs["scope"] == "user"

    def test_put_op_empty_content(self) -> None:
        store, client = self._make_store()

        store.batch([PutOp(("user",), "key1", {"content": ""})])

        client.memory_stores.begin_update_memories.assert_not_called()

    def test_put_op_with_items(self) -> None:
        store, client = self._make_store()
        poller = MagicMock()
        client.memory_stores.begin_update_memories.return_value = poller

        pre_built_items = [MagicMock()]
        store.batch(
            [PutOp(("user",), "key1", {"items": pre_built_items})]
        )

        client.memory_stores.begin_update_memories.assert_called_once()

    def test_list_namespaces_op(self) -> None:
        store, client = self._make_store()
        # Simulate some observed scopes by running a search first
        mem = _make_memory_item(scope="user/123")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])
        store.batch([SearchOp(("user", "123"), None, 10, 0, "query")])

        results = store.batch([ListNamespacesOp()])
        ns_list = results[0]
        assert ("user", "123") in ns_list

    def test_list_namespaces_with_prefix(self) -> None:
        store, _ = self._make_store()
        store._known_scopes = {"user/1", "user/2", "admin/1"}

        results = store.batch(
            [
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("user",)),
                    )
                )
            ]
        )
        ns_list = results[0]
        assert len(ns_list) == 2
        assert all(ns[0] == "user" for ns in ns_list)

    def test_list_namespaces_with_max_depth(self) -> None:
        store, _ = self._make_store()
        store._known_scopes = {"a/b/c", "a/b/d"}

        results = store.batch([ListNamespacesOp(max_depth=2)])
        ns_list = results[0]
        assert all(len(ns) <= 2 for ns in ns_list)

    def test_unknown_op_raises(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="Unknown operation type"):
            store.batch(["not-an-op"])  # type: ignore[list-item]


class TestAzureAIMemoryStoreAsync:
    def _make_store(self) -> tuple[AzureAIMemoryStore, MagicMock]:
        client = MagicMock()
        async_client = AsyncMock()
        store = AzureAIMemoryStore(
            project_client=client,
            async_project_client=async_client,
            memory_store_name="test-store",
        )
        return store, async_client

    @pytest.mark.asyncio
    async def test_abatch_requires_async_client(self) -> None:
        client = MagicMock()
        store = AzureAIMemoryStore(
            project_client=client,
            memory_store_name="test-store",
        )
        with pytest.raises(ValueError, match="async_project_client"):
            await store.abatch([GetOp(("user",), "key1")])

    @pytest.mark.asyncio
    async def test_aget(self) -> None:
        store, async_client = self._make_store()
        mem = _make_memory_item(memory_id="key1", scope="user")
        async_client.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        results = await store.abatch([GetOp(("user",), "key1")])
        assert results[0] is not None
        assert results[0].key == "key1"

    @pytest.mark.asyncio
    async def test_asearch(self) -> None:
        store, async_client = self._make_store()
        mem = _make_memory_item(scope="user")
        async_client.memory_stores.search_memories.return_value = (
            _make_search_result([mem])
        )

        results = await store.abatch(
            [SearchOp(("user",), None, 10, 0, "hobbies")]
        )
        items = results[0]
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_aput_update(self) -> None:
        store, async_client = self._make_store()
        poller = AsyncMock()
        async_client.memory_stores.begin_update_memories.return_value = poller

        await store.abatch(
            [PutOp(("user",), "key1", {"content": "I like hiking"})]
        )

        async_client.memory_stores.begin_update_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_aput_delete(self) -> None:
        store, async_client = self._make_store()

        await store.abatch([PutOp(("user",), "key1", None)])

        async_client.memory_stores.delete_scope.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_op_raises_async(self) -> None:
        store, _ = self._make_store()
        with pytest.raises(ValueError, match="Unknown operation type"):
            await store.abatch(["not-an-op"])  # type: ignore[list-item]


class TestAzureAIMemoryStoreHighLevel:
    """Test the high-level convenience methods inherited from BaseStore."""

    def _make_store(self) -> tuple[AzureAIMemoryStore, MagicMock]:
        client = MagicMock()
        store = AzureAIMemoryStore(
            project_client=client,
            memory_store_name="test-store",
        )
        return store, client

    def test_put_and_search(self) -> None:
        store, client = self._make_store()
        poller = MagicMock()
        client.memory_stores.begin_update_memories.return_value = poller

        store.put(("user",), "conv1", {"content": "I like hiking"})

        client.memory_stores.begin_update_memories.assert_called_once()

    def test_delete(self) -> None:
        store, client = self._make_store()

        store.delete(("user",), "key1")

        client.memory_stores.delete_scope.assert_called_once()

    def test_search(self) -> None:
        store, client = self._make_store()
        mem = _make_memory_item(scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])

        results = store.search(("user",), query="hobbies")
        assert len(results) == 1
        assert isinstance(results[0], SearchItem)

    def test_get(self) -> None:
        store, client = self._make_store()
        mem = _make_memory_item(memory_id="key1", scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])

        result = store.get(("user",), "key1")
        assert result is not None
        assert result.key == "key1"

    def test_list_namespaces(self) -> None:
        store, _ = self._make_store()
        store._known_scopes = {"user/1", "admin/1"}

        namespaces = store.list_namespaces(prefix=("user",))
        assert len(namespaces) == 1
        assert namespaces[0] == ("user", "1")

    def test_scope_tracking(self) -> None:
        store, client = self._make_store()
        mem = _make_memory_item(scope="user")
        client.memory_stores.search_memories.return_value = _make_search_result([mem])

        store.search(("user",), query="test")
        assert "user" in store._known_scopes

    def test_scope_removed_on_delete(self) -> None:
        store, client = self._make_store()
        store._known_scopes = {"user"}

        store.delete(("user",), "key1")
        assert "user" not in store._known_scopes
