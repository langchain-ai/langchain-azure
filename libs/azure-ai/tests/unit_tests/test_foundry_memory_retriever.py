"""Unit tests for FoundryMemoryRetriever."""

from unittest.mock import Mock

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from langchain_azure_ai.memory import (
    FoundryMemoryChatMessageHistory,
    FoundryMemoryRetriever,
)


class TestRetrieverConstruction:
    """Test retriever initialization."""

    def test_retriever_with_history_ref(self) -> None:
        """Test retriever construction with history reference."""
        mock_client = Mock()
        history = FoundryMemoryChatMessageHistory(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            session_id="session_1",
            base_history_factory=lambda _: InMemoryChatMessageHistory(),
        )

        retriever = FoundryMemoryRetriever(
            client=mock_client, history_ref=history, k=10
        )

        assert retriever.store_name == "test_store"
        assert retriever.scope == "user:test"
        assert retriever.session_id == "session_1"
        assert retriever.k == 10

    def test_retriever_without_history_ref(self) -> None:
        """Test retriever construction without history reference."""
        mock_client = Mock()

        retriever = FoundryMemoryRetriever(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            k=5,
        )

        assert retriever.store_name == "test_store"
        assert retriever.scope == "user:test"
        assert retriever.k == 5

    def test_retriever_requires_store_and_scope_without_history(self) -> None:
        """Test that retriever requires store and scope if no history ref."""
        mock_client = Mock()

        with pytest.raises(
            ValueError, match="Either provide history_ref or both store_name and scope"
        ):
            FoundryMemoryRetriever(client=mock_client, store_name="test_store")

        with pytest.raises(
            ValueError, match="Either provide history_ref or both store_name and scope"
        ):
            FoundryMemoryRetriever(client=mock_client, scope="user:test")


class TestRetrieverSearch:
    """Test retriever search functionality."""

    def test_search_returns_documents(self) -> None:
        """Test that search returns LangChain Documents."""
        mock_client = Mock()

        # Mock search result
        mock_memory_item = Mock()
        mock_memory_item.content = "User prefers dark roast coffee"
        mock_memory_item.kind = "user_profile"
        mock_memory_item.memory_id = "mem_123"
        mock_memory_item.scope = "user:test"

        mock_search_item = Mock()
        mock_search_item.memory_item = mock_memory_item

        mock_result = Mock()
        mock_result.memories = [mock_search_item]
        mock_result.search_id = "search_abc"

        mock_client.memory_stores.search_memories = Mock(return_value=mock_result)

        retriever = FoundryMemoryRetriever(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            k=5,
        )

        docs = retriever.invoke("coffee preference")

        assert len(docs) == 1
        assert docs[0].page_content == "User prefers dark roast coffee"
        assert docs[0].metadata["kind"] == "user_profile"
        assert docs[0].metadata["memory_id"] == "mem_123"
        assert docs[0].metadata["scope"] == "user:test"
        assert docs[0].metadata["source"] == "foundry_memory"

    def test_search_with_dict_result(self) -> None:
        """Test that search handles dict-based responses (not just objects)."""
        mock_client = Mock()

        # Mock search result as dict
        mock_result = {
            "memories": [
                {
                    "memory_item": {
                        "content": "User likes hiking",
                        "kind": "user_profile",
                        "memory_id": "mem_456",
                        "scope": "user:test",
                    }
                }
            ],
            "search_id": "search_xyz",
        }

        mock_client.memory_stores.search_memories = Mock(return_value=mock_result)

        retriever = FoundryMemoryRetriever(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            k=5,
        )

        docs = retriever.invoke("hiking preference")

        assert len(docs) == 1
        assert docs[0].page_content == "User likes hiking"
        assert docs[0].metadata["kind"] == "user_profile"

    def test_search_caches_search_id_in_incremental_mode(self) -> None:
        """Test that search_id is cached in incremental mode."""
        mock_client = Mock()

        mock_result = Mock()
        mock_result.memories = []
        mock_result.search_id = "search_123"

        mock_client.memory_stores.search_memories = Mock(return_value=mock_result)

        history = FoundryMemoryChatMessageHistory(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            session_id="session_1",
            base_history_factory=lambda _: InMemoryChatMessageHistory(),
        )

        retriever = FoundryMemoryRetriever(client=mock_client, history_ref=history, k=5)

        retriever.invoke("first query")

        # Check that search_id was cached
        assert retriever._previous_search_id == "search_123"

        # Second query should use previous_search_id
        retriever.invoke("follow-up query")

        # Verify previous_search_id was passed
        call_kwargs = mock_client.memory_stores.search_memories.call_args[1]
        assert call_kwargs["previous_search_id"] == "search_123"

    def test_search_does_not_cache_in_non_incremental_mode(self) -> None:
        """Test that search_id is NOT cached in non-incremental mode."""
        mock_client = Mock()

        mock_result = Mock()
        mock_result.memories = []
        mock_result.search_id = "search_456"

        mock_client.memory_stores.search_memories = Mock(return_value=mock_result)

        retriever = FoundryMemoryRetriever(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            k=5,
        )

        retriever.invoke("first query")

        # Check that search_id was NOT cached
        assert retriever._previous_search_id is None

        # Second query should NOT use previous_search_id
        retriever.invoke("second query")

        call_kwargs = mock_client.memory_stores.search_memories.call_args[1]
        assert call_kwargs["previous_search_id"] is None

    def test_search_with_history_context_incremental(self) -> None:
        """Test search includes context from last assistant message."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(return_value=Mock())

        mock_result = Mock()
        mock_result.memories = []
        mock_result.search_id = "search_abc"
        mock_client.memory_stores.search_memories = Mock(return_value=mock_result)

        history = FoundryMemoryChatMessageHistory(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            session_id="session_1",
            base_history_factory=lambda _: InMemoryChatMessageHistory(),
        )

        # Add conversation history
        history.add_message(HumanMessage(content="What's my favorite drink?"))
        history.add_message(AIMessage(content="You prefer coffee"))
        history.add_message(HumanMessage(content="What about food?"))

        retriever = FoundryMemoryRetriever(client=mock_client, history_ref=history, k=5)

        retriever.invoke("Tell me about my preferences")

        # Verify that search included context from last assistant message onward
        call_kwargs = mock_client.memory_stores.search_memories.call_args[1]
        items = call_kwargs["items"]

        # Should have at least: last AI msg, last human msg, current query
        assert len(items) >= 3

    def test_search_handles_partial_failures(self) -> None:
        """Test that search handles partial parsing failures gracefully."""
        mock_client = Mock()

        # Mock result with some malformed data
        mock_result = Mock()
        mock_result.memories = [None, Mock(memory_item=None)]

        mock_client.memory_stores.search_memories = Mock(return_value=mock_result)

        retriever = FoundryMemoryRetriever(
            client=mock_client,
            store_name="test_store",
            scope="user:test",
            k=5,
        )

        # Should not raise exception
        docs = retriever.invoke("test query")

        # Should return empty list rather than crashing
        assert docs == []
