# type: ignore
"""Integration tests for CosmosDBStore (sync)."""

from __future__ import annotations

import math
import os
import time

import pytest
from langchain_azure_cosmosdb import CosmosDBStore
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)

from tests.embed_test_utils import CharacterEmbeddings

pytestmark = pytest.mark.skipif(
    not os.getenv("COSMOSDB_CONN_STRING") and not os.getenv("COSMOSDB_ENDPOINT"),
    reason="COSMOSDB_CONN_STRING or COSMOSDB_ENDPOINT environment variable not set",
)

DEFAULT_DATABASE = os.getenv("COSMOSDB_TEST_DATABASE", "langgraph_test")


def _make_store(
    *,
    container_name: str,
    index: dict | None = None,
    ttl: dict | None = None,
) -> CosmosDBStore:
    """Create a CosmosDBStore using conn string or Entra ID endpoint."""
    conn_string = os.getenv("COSMOSDB_CONN_STRING")
    if conn_string:
        return CosmosDBStore.from_conn_string(
            conn_string,
            database_name=DEFAULT_DATABASE,
            container_name=container_name,
            index=index,
            ttl=ttl,
        )
    from azure.identity import AzureCliCredential

    endpoint = os.environ["COSMOSDB_ENDPOINT"]
    credential = AzureCliCredential(process_timeout=60)
    return CosmosDBStore.from_endpoint(
        endpoint,
        credential=credential,
        database_name=DEFAULT_DATABASE,
        container_name=container_name,
        index=index,
        ttl=ttl,
    )


TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@pytest.fixture(scope="function")
def store() -> CosmosDBStore:
    """Create a fresh CosmosDBStore for each test with TTL enabled."""
    container_name = "store_test"
    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    store = _make_store(container_name=container_name, ttl=ttl_config)
    store.setup()
    yield store


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=500)


@pytest.fixture(scope="function")
def vector_store(fake_embeddings: CharacterEmbeddings) -> CosmosDBStore:
    """Create a CosmosDBStore with vector search enabled."""
    container_name = "store_test_vec"
    index_config = {
        "dims": 500,
        "embed": fake_embeddings,
        "fields": ["text"],
    }
    store = _make_store(container_name=container_name, index=index_config)
    store.setup()
    yield store


class TestBatchOperations:
    """Tests for batch operations."""

    def test_batch_order(self, store: CosmosDBStore) -> None:
        """Test that operations in a batch return results in the correct order."""
        store.put(("test", "foo"), "key1", {"data": "value1"})

        ops = [
            GetOp(namespace=("test", "foo"), key="key1"),
            SearchOp(namespace_prefix=("test",), filter=None, limit=10, offset=0),
            ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
            GetOp(namespace=("test", "foo"), key="key2"),
        ]

        results = store.batch(ops)
        assert len(results) == 4

        assert isinstance(results[0], Item)
        assert results[0].key == "key1"
        assert isinstance(results[1], list)
        assert isinstance(results[2], list)
        assert results[3] is None

    def test_batch_get_ops(self, store: CosmosDBStore) -> None:
        """Test getting items via batch operations."""
        store.put(("test",), "key1", {"data": "value1"})
        store.put(("test",), "key2", {"data": "value2"})

        ops = [
            GetOp(namespace=("test",), key="key1"),
            GetOp(namespace=("test",), key="key2"),
            GetOp(namespace=("test",), key="key3"),
        ]
        results = store.batch(ops)

        assert results[0] is not None
        assert results[0].value == {"data": "value1"}
        assert results[1] is not None
        assert results[1].value == {"data": "value2"}
        assert results[2] is None

    def test_batch_put_ops(self, store: CosmosDBStore) -> None:
        """Test putting items via batch operations."""
        ops = [
            PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
            PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        ]
        store.batch(ops)

        item1 = store.get(("test",), "key1")
        assert item1 is not None
        assert item1.value == {"data": "value1"}

        item2 = store.get(("test",), "key2")
        assert item2 is not None
        assert item2.value == {"data": "value2"}

    def test_batch_search_ops(self, store: CosmosDBStore) -> None:
        """Test searching items via batch operations."""
        store.put(("test", "search"), "key1", {"data": "value1"})
        store.put(("test", "search"), "key2", {"data": "value2"})

        ops = [
            SearchOp(
                namespace_prefix=("test", "search"),
                filter=None,
                limit=10,
                offset=0,
            ),
        ]
        results = store.batch(ops)

        assert isinstance(results[0], list)
        assert len(results[0]) == 2


class TestBasicStoreOps:
    """Tests for basic put/get/delete/search/list operations."""

    def test_put_get(self, store: CosmosDBStore) -> None:
        """Test basic put and get."""
        store.put(("test",), "key1", {"data": "value1"})
        item = store.get(("test",), "key1")
        assert item is not None
        assert item.value == {"data": "value1"}
        assert item.key == "key1"
        assert item.namespace == ("test",)

    def test_put_update(self, store: CosmosDBStore) -> None:
        """Test updating an existing item."""
        store.put(("test",), "key1", {"data": "value1"})
        store.put(("test",), "key1", {"data": "updated"})
        item = store.get(("test",), "key1")
        assert item is not None
        assert item.value == {"data": "updated"}

    def test_put_delete(self, store: CosmosDBStore) -> None:
        """Test deleting an item by putting None."""
        store.put(("test",), "key1", {"data": "value1"})
        store.delete(("test",), "key1")
        item = store.get(("test",), "key1")
        assert item is None

    def test_delete_nonexistent(self, store: CosmosDBStore) -> None:
        """Test deleting a non-existent item doesn't raise."""
        store.delete(("test",), "nonexistent")

    def test_get_nonexistent(self, store: CosmosDBStore) -> None:
        """Test getting a non-existent item returns None."""
        item = store.get(("test",), "nonexistent")
        assert item is None

    def test_put_many_get_many(self, store: CosmosDBStore) -> None:
        """Test putting and getting multiple items."""
        store.put(("test",), "key1", {"data": "value1"})
        store.put(("test",), "key2", {"data": "value2"})
        store.put(("test",), "key3", {"data": "value3"})

        results = store.batch(
            [
                GetOp(namespace=("test",), key="key1"),
                GetOp(namespace=("test",), key="key2"),
                GetOp(namespace=("test",), key="key3"),
            ]
        )
        items = [r for r in results if r is not None]
        assert len(items) == 3
        values = [item.value if item else None for item in items]
        assert {"data": "value1"} in values
        assert {"data": "value2"} in values
        assert {"data": "value3"} in values

    def test_namespaced_isolation(self, store: CosmosDBStore) -> None:
        """Test that items in different namespaces are isolated."""
        store.put(("ns1",), "key1", {"data": "ns1_value"})
        store.put(("ns2",), "key1", {"data": "ns2_value"})

        item1 = store.get(("ns1",), "key1")
        item2 = store.get(("ns2",), "key1")
        assert item1 is not None
        assert item1.value == {"data": "ns1_value"}
        assert item2 is not None
        assert item2.value == {"data": "ns2_value"}


class TestSearch:
    """Tests for search operations."""

    def test_search_basic(self, store: CosmosDBStore) -> None:
        """Test basic search by namespace prefix."""
        store.put(("test", "search"), "key1", {"data": "value1"})
        store.put(("test", "search"), "key2", {"data": "value2"})
        store.put(("other",), "key3", {"data": "value3"})

        results = store.search(("test", "search"))
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"key1", "key2"}

    def test_search_with_filter(self, store: CosmosDBStore) -> None:
        """Test search with filter conditions."""
        store.put(("test",), "key1", {"status": "active", "score": 10})
        store.put(("test",), "key2", {"status": "inactive", "score": 20})
        store.put(("test",), "key3", {"status": "active", "score": 30})

        results = store.search(("test",), filter={"status": "active"})
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"key1", "key3"}

    def test_search_with_operator_filter(self, store: CosmosDBStore) -> None:
        """Test search with comparison operators."""
        store.put(("test",), "key1", {"score": 10})
        store.put(("test",), "key2", {"score": 20})
        store.put(("test",), "key3", {"score": 30})

        results = store.search(("test",), filter={"score": {"$gt": 15}})
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"key2", "key3"}

    def test_search_limit(self, store: CosmosDBStore) -> None:
        """Test search with limit."""
        for i in range(5):
            store.put(("test",), f"key{i}", {"data": f"value{i}"})

        results = store.search(("test",), limit=3)
        assert len(results) == 3

    def test_search_offset(self, store: CosmosDBStore) -> None:
        """Test search with offset."""
        for i in range(5):
            store.put(("test",), f"key{i}", {"data": f"value{i}"})

        all_results = store.search(("test",), limit=10)
        offset_results = store.search(("test",), limit=10, offset=2)
        assert len(offset_results) == len(all_results) - 2

    def test_search_empty(self, store: CosmosDBStore) -> None:
        """Test search returns empty when no matches."""
        results = store.search(("nonexistent",))
        assert results == []


class TestListNamespaces:
    """Tests for list_namespaces operations."""

    def test_list_namespaces(self, store: CosmosDBStore) -> None:
        """Test listing all namespaces."""
        store.put(("a", "b"), "key1", {"data": "1"})
        store.put(("a", "c"), "key2", {"data": "2"})
        store.put(("d",), "key3", {"data": "3"})

        namespaces = store.list_namespaces(prefix=("a",))
        assert len(namespaces) >= 2
        assert ("a", "b") in namespaces
        assert ("a", "c") in namespaces

    def test_list_namespaces_with_prefix(self, store: CosmosDBStore) -> None:
        """Test listing namespaces with prefix filter."""
        store.put(("users", "alice"), "key1", {"data": "1"})
        store.put(("users", "bob"), "key2", {"data": "2"})
        store.put(("docs",), "key3", {"data": "3"})

        namespaces = store.list_namespaces(prefix=("users",))
        assert ("users", "alice") in namespaces
        assert ("users", "bob") in namespaces
        assert ("docs",) not in namespaces

    def test_list_namespaces_with_suffix(self, store: CosmosDBStore) -> None:
        """Test listing namespaces with suffix filter."""
        store.put(("users", "alice"), "key1", {"data": "1"})
        store.put(("users", "bob"), "key2", {"data": "2"})
        store.put(("admins", "alice"), "key3", {"data": "3"})

        namespaces = store.list_namespaces(suffix=("alice",))
        assert ("users", "alice") in namespaces
        assert ("admins", "alice") in namespaces
        assert ("users", "bob") not in namespaces

    def test_list_namespaces_max_depth(self, store: CosmosDBStore) -> None:
        """Test listing namespaces with max_depth."""
        store.put(("a", "b", "c"), "key1", {"data": "1"})
        store.put(("a", "b", "d"), "key2", {"data": "2"})
        store.put(("a", "x"), "key3", {"data": "3"})

        namespaces = store.list_namespaces(prefix=("a",), max_depth=2)
        assert ("a", "b") in namespaces
        assert ("a", "x") in namespaces
        assert ("a", "b", "c") not in namespaces

    def test_list_namespaces_limit_offset(self, store: CosmosDBStore) -> None:
        """Test listing namespaces with limit and offset."""
        for i in range(5):
            store.put(("ns", f"sub{i}"), "key", {"data": "1"})

        all_ns = store.list_namespaces(prefix=("ns",), limit=10)
        limited = store.list_namespaces(prefix=("ns",), limit=2)
        offset = store.list_namespaces(prefix=("ns",), limit=10, offset=2)

        assert len(limited) == 2
        assert len(offset) == len(all_ns) - 2


class TestVectorStore:
    """Tests for vector search functionality."""

    def test_vector_store_init(self, vector_store: CosmosDBStore) -> None:
        """Test that vector store is initialized correctly."""
        assert vector_store.index_config is not None
        assert vector_store.embeddings is not None

    def test_vector_store_insert_and_search(self, vector_store: CosmosDBStore) -> None:
        """Test inserting items and performing vector search."""
        vector_store.put(("docs",), "doc1", {"text": "Python programming language"})
        vector_store.put(("docs",), "doc2", {"text": "JavaScript web development"})
        vector_store.put(("docs",), "doc3", {"text": "Python data science"})

        results = vector_store.search(("docs",), query="Python coding", limit=3)
        assert len(results) > 0
        for result in results:
            assert result.score is not None

    def test_vector_store_update(self, vector_store: CosmosDBStore) -> None:
        """Test updating items with embeddings."""
        vector_store.put(("docs",), "doc1", {"text": "original text"})
        vector_store.put(("docs",), "doc1", {"text": "updated text here"})

        item = vector_store.get(("docs",), "doc1")
        assert item is not None
        assert item.value == {"text": "updated text here"}

    def test_vector_store_with_filter(self, vector_store: CosmosDBStore) -> None:
        """Test vector search with filters."""
        vector_store.put(
            ("docs",), "doc1", {"text": "Python programming", "category": "lang"}
        )
        vector_store.put(
            ("docs",), "doc2", {"text": "JavaScript programming", "category": "lang"}
        )
        vector_store.put(
            ("docs",), "doc3", {"text": "Python data science", "category": "data"}
        )

        results = vector_store.search(
            ("docs",), query="Python", filter={"category": "lang"}, limit=5
        )
        assert len(results) > 0
        for r in results:
            assert r.value.get("category") == "lang"

    def test_vector_store_pagination(self, vector_store: CosmosDBStore) -> None:
        """Test vector search with pagination."""
        for i in range(10):
            vector_store.put(("docs",), f"doc{i}", {"text": f"document number {i}"})

        page1 = vector_store.search(("docs",), query="document", limit=3, offset=0)
        page2 = vector_store.search(("docs",), query="document", limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3

        page1_keys = {r.key for r in page1}
        page2_keys = {r.key for r in page2}
        assert page1_keys.isdisjoint(page2_keys)

    def test_vector_store_index_false(self, vector_store: CosmosDBStore) -> None:
        """Test that items with index=False don't get embeddings."""
        vector_store.put(("docs",), "doc1", {"text": "indexed document"}, index=True)
        vector_store.put(
            ("docs",), "doc2", {"text": "not indexed document"}, index=False
        )

        item1 = vector_store.get(("docs",), "doc1")
        item2 = vector_store.get(("docs",), "doc2")
        assert item1 is not None
        assert item2 is not None

    def test_vector_store_edge_cases(self, vector_store: CosmosDBStore) -> None:
        """Test edge cases for vector store."""
        vector_store.put(("docs",), "empty", {"text": ""})
        item = vector_store.get(("docs",), "empty")
        assert item is not None

        vector_store.put(("docs",), "no_text", {"other": "data"})
        item2 = vector_store.get(("docs",), "no_text")
        assert item2 is not None

    def test_embed_with_path(self, fake_embeddings: CharacterEmbeddings) -> None:
        """Test vector store with custom embedding fields."""
        container_name = "store_test_embed"
        index_config = {
            "dims": 500,
            "embed": fake_embeddings,
            "fields": ["title", "description"],
        }
        store = _make_store(container_name=container_name, index=index_config)
        store.setup()
        store.put(
            ("docs",),
            "doc1",
            {"title": "Python Guide", "description": "Learn Python"},
        )
        results = store.search(("docs",), query="Python", limit=5)
        assert len(results) > 0


class TestScores:
    """Tests for search result scores."""

    def test_search_scores_without_vector(self, store: CosmosDBStore) -> None:
        """Test that non-vector search returns items without scores."""
        store.put(("test",), "key1", {"data": "value1"})
        results = store.search(("test",))
        assert len(results) > 0
        for r in results:
            assert r.score is None

    def test_search_scores_with_vector(self, vector_store: CosmosDBStore) -> None:
        """Test that vector search returns numeric scores."""
        vector_store.put(("docs",), "doc1", {"text": "Python programming"})
        vector_store.put(("docs",), "doc2", {"text": "JavaScript coding"})

        results = vector_store.search(("docs",), query="Python", limit=5)
        assert len(results) > 0
        for r in results:
            assert r.score is not None
            assert isinstance(r.score, float)


class TestTTL:
    """Tests for TTL (Time-to-Live) functionality."""

    def test_ttl_on_put(self, store: CosmosDBStore) -> None:
        """Test that items get default TTL from config."""
        store.put(("test",), "key1", {"data": "value1"})
        item = store.get(("test",), "key1")
        assert item is not None
        assert item.value == {"data": "value1"}

    def test_ttl_custom(self, store: CosmosDBStore) -> None:
        """Test setting custom TTL on individual items."""
        store.put(("test",), "key1", {"data": "value1"}, ttl=5.0)
        item = store.get(("test",), "key1")
        assert item is not None

    def test_ttl_fields_in_document(self) -> None:
        """Test that TTL fields are set correctly in documents."""
        container_name = "store_test_ttl"
        ttl_config = {
            "default_ttl": 10.0,
            "refresh_on_read": True,
        }
        store = _make_store(container_name=container_name, ttl=ttl_config)
        store.setup()
        store.put(("test",), "key1", {"data": "value1"})

        docs = list(
            store.container.query_items(
                query="SELECT * FROM c WHERE c.key = @key",
                parameters=[{"name": "@key", "value": "key1"}],
                enable_cross_partition_query=True,
            )
        )
        assert len(docs) == 1
        doc = docs[0]
        assert "ttl" in doc
        assert doc["ttl"] == 600  # 10 minutes * 60 seconds
        assert doc["ttl_minutes"] == 10.0


class TestNonAscii:
    """Tests for non-ASCII characters in namespaces, keys, and values."""

    def test_non_ascii_namespace(self, store: CosmosDBStore) -> None:
        """Test that non-ASCII characters work in namespaces."""
        store.put(("用户", "数据"), "key1", {"name": "测试"})
        item = store.get(("用户", "数据"), "key1")
        assert item is not None
        assert item.value == {"name": "测试"}

    def test_non_ascii_key(self, store: CosmosDBStore) -> None:
        """Test that non-ASCII characters work in keys."""
        store.put(("test",), "schlüssel", {"data": "wert"})
        item = store.get(("test",), "schlüssel")
        assert item is not None
        assert item.value == {"data": "wert"}

    def test_non_ascii_value(self, store: CosmosDBStore) -> None:
        """Test that non-ASCII characters work in values."""
        value = {"name": "日本語テスト", "emoji": "🎉"}
        store.put(("test",), "key1", value)
        item = store.get(("test",), "key1")
        assert item is not None
        assert item.value == value


class TestBatchListNamespacesOps:
    """Tests for batch list namespaces operations."""

    def test_batch_list_namespaces_ops(self, store: CosmosDBStore) -> None:
        """Test batch list namespace operations with various conditions."""
        test_data = [
            (("test", "documents", "public"), "doc1", {"content": "public doc"}),
            (("test", "documents", "private"), "doc2", {"content": "private doc"}),
            (("test", "images", "public"), "img1", {"content": "public image"}),
            (("prod", "documents", "public"), "doc3", {"content": "prod doc"}),
        ]
        for namespace, key, value in test_data:
            store.put(namespace, key, value)

        ops = [
            ListNamespacesOp(
                match_conditions=None, max_depth=None, limit=100, offset=0
            ),
            ListNamespacesOp(match_conditions=None, max_depth=2, limit=100, offset=0),
            ListNamespacesOp(
                match_conditions=[MatchCondition("suffix", "public")],
                max_depth=None,
                limit=100,
                offset=0,
            ),
        ]

        results = store.batch(ops)
        assert len(results) == 3

        expected_ns = {ns for ns, _, _ in test_data}
        assert expected_ns.issubset(set(results[0]))
        assert all(len(ns) <= 2 for ns in results[1])
        assert all(ns[-1] == "public" for ns in results[2])


class TestBatchPutDelete:
    """Tests for batch put with None value (delete)."""

    def test_batch_put_with_none_deletes(self, store: CosmosDBStore) -> None:
        """Test that putting None value in a batch deletes the item."""
        ops = [
            PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
            PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
            PutOp(namespace=("test",), key="key3", value=None),
        ]

        results = store.batch(ops)
        assert len(results) == 3
        assert all(result is None for result in results)

        item1 = store.get(("test",), "key1")
        item2 = store.get(("test",), "key2")
        item3 = store.get(("test",), "key3")

        assert item1 and item1.value == {"data": "value1"}
        assert item2 and item2.value == {"data": "value2"}
        assert item3 is None


class TestVectorInsertAutoEmbedding:
    """Tests for auto-embedding of inserted items."""

    def test_vector_insert_with_auto_embedding(
        self, vector_store: CosmosDBStore
    ) -> None:
        """Test inserting items that get auto-embedded."""
        docs = [
            ("doc1", {"text": "short text"}),
            ("doc2", {"text": "longer text document"}),
            ("doc3", {"text": "longest text document here"}),
            ("doc4", {"description": "text in description field"}),
            ("doc5", {"content": "text in content field"}),
            ("doc6", {"body": "text in body field"}),
        ]

        for key, value in docs:
            vector_store.put(("test",), key, value)

        results = vector_store.search(("test",), query="long text")
        assert len(results) > 0

        doc_order = [r.key for r in results]
        assert "doc2" in doc_order
        assert "doc3" in doc_order


class TestVectorUpdateEmbedding:
    """Tests for updating items and their embeddings."""

    def test_vector_update_with_embedding(self, vector_store: CosmosDBStore) -> None:
        """Test that updating items properly updates their embeddings."""
        vector_store.put(("test",), "doc1", {"text": "zany zebra Xerxes"})
        vector_store.put(("test",), "doc2", {"text": "something about dogs"})
        vector_store.put(("test",), "doc3", {"text": "text about birds"})

        results_initial = vector_store.search(("test",), query="Zany Xerxes")
        assert len(results_initial) > 0
        assert results_initial[0].key == "doc1"
        initial_score = results_initial[0].score

        vector_store.put(("test",), "doc1", {"text": "new text about dogs"})

        results_after = vector_store.search(("test",), query="Zany Xerxes")
        after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
        assert after_score < initial_score

        results_new = vector_store.search(("test",), query="new text about dogs")
        for r in results_new:
            if r.key == "doc1":
                assert r.score > after_score

        # Don't index this one
        vector_store.put(
            ("test",), "doc4", {"text": "new text about dogs"}, index=False
        )
        results_new = vector_store.search(
            ("test",), query="new text about dogs", limit=3
        )
        assert not any(r.key == "doc4" for r in results_new)


class TestVectorSearchWithFilters:
    """Tests for combining vector search with filters."""

    def test_vector_search_with_filters(self, vector_store: CosmosDBStore) -> None:
        """Test combining vector search with comparison filters."""
        docs = [
            ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
            ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
            ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
            ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
        ]

        for key, value in docs:
            vector_store.put(("test",), key, value)

        results = vector_store.search(("test",), query="apple", filter={"color": "red"})
        assert len(results) == 2
        assert results[0].key == "doc1"

        results = vector_store.search(("test",), query="car", filter={"color": "red"})
        assert len(results) == 2
        assert results[0].key == "doc2"

        results = vector_store.search(
            ("test",),
            query="bbbbluuu",
            filter={"score": {"$gt": 3.2}},
        )
        assert len(results) == 3
        assert results[0].key == "doc4"

        # Multiple filters
        results = vector_store.search(
            ("test",),
            query="apple",
            filter={"score": {"$gte": 4.0}, "color": "green"},
        )
        assert len(results) == 1
        assert results[0].key == "doc3"


class TestSearchSorting:
    """Tests for search result sorting."""

    def test_search_sorting(self, fake_embeddings: CharacterEmbeddings) -> None:
        """Test that the best match is returned first."""
        container_name = "store_test_sort"
        index_config = {
            "dims": 500,
            "embed": fake_embeddings,
            "fields": ["key1"],
        }
        store = _make_store(container_name=container_name, index=index_config)
        store.setup()
        amatch = {"key1": "mmm"}
        store.put(("test", "M"), "M", amatch)
        N = 50
        for i in range(N):
            store.put(("test", "A"), f"A{i}", {"key1": "no"})
        for i in range(N):
            store.put(("test", "Z"), f"Z{i}", {"key1": "no"})

        results = store.search(("test",), query="mmm", limit=10)
        assert len(results) == 10
        assert len(set(r.key for r in results)) == 10
        assert results[0].key == "M"
        assert results[0].score > results[1].score


class TestScoresVerification:
    """Tests for score verification with manual cosine similarity."""

    @pytest.mark.parametrize("query", ["aaa", "bbb", "ccc", "abcd", "poisson"])
    def test_scores_match_cosine(
        self,
        fake_embeddings: CharacterEmbeddings,
        query: str,
    ) -> None:
        """Test that returned scores match manual cosine similarity."""
        container_name = "store_test_scores"
        index_config = {
            "dims": 500,
            "embed": fake_embeddings,
            "fields": ["key0"],
        }
        store = _make_store(container_name=container_name, index=index_config)
        store.setup()
        doc = {"key0": "aaa"}
        store.put(("test",), "doc", doc, index=["key0", "key1"])

        results = store.search((), query=query)
        vec0 = fake_embeddings.embed_query(doc["key0"])
        vec1 = fake_embeddings.embed_query(query)
        similarities = _cosine_similarity(vec1, [vec0])

        assert len(results) == 1
        assert results[0].score == pytest.approx(similarities[0], abs=1e-3)


class TestStoreTTLExpiry:
    """Tests for real TTL expiry with sleep."""

    def test_store_ttl_expiry(self, store: CosmosDBStore) -> None:
        """Test that items actually expire after their TTL.

        Cosmos DB emulator's native TTL enforcement has granularity of
        approximately 5-10 seconds. We test basic expiry with generous
        margins and also test the refresh mechanism by re-putting the item.
        """
        short_ttl_seconds = 10
        short_ttl_minutes = short_ttl_seconds / 60

        ns = ("ttl_test",)

        # Phase 1: verify item exists before TTL
        store.put(ns, key="item1", value={"foo": "bar"}, ttl=short_ttl_minutes)
        time.sleep(3)
        res = store.get(ns, key="item1")
        assert res is not None, "Item should exist well before TTL"

        # Phase 2: verify item expires after TTL
        time.sleep(short_ttl_seconds + 10)
        res = store.get(ns, key="item1")
        assert res is None, "Item should have expired after TTL"

        # Phase 3: re-put and verify refresh by re-putting extends lifetime
        store.put(ns, key="item2", value={"bar": "baz"}, ttl=short_ttl_minutes)
        time.sleep(3)
        store.put(ns, key="item2", value={"bar": "baz"}, ttl=short_ttl_minutes)
        time.sleep(short_ttl_seconds - 2)
        res = store.get(ns, key="item2")
        assert res is not None, "Re-put should have extended TTL"


class TestNonAsciiWithVectorSearch:
    """Tests for non-ASCII characters with vector search."""

    def test_non_ascii_vector_search(
        self, fake_embeddings: CharacterEmbeddings
    ) -> None:
        """Test non-ASCII characters with vector search."""
        container_name = "store_test_noascii"
        index_config = {
            "dims": 500,
            "embed": fake_embeddings,
            "fields": ["text"],
        }
        store = _make_store(container_name=container_name, index=index_config)
        store.setup()
        store.put(("user_123", "memories"), "1", {"text": "这是中文"})
        store.put(("user_123", "memories"), "2", {"text": "これは日本語です"})
        store.put(("user_123", "memories"), "3", {"text": "이건 한국어야"})
        store.put(("user_123", "memories"), "4", {"text": "Это русский"})
        store.put(("user_123", "memories"), "5", {"text": "यह रूसी है"})

        result1 = store.search(("user_123", "memories"), query="这是中文")
        result2 = store.search(("user_123", "memories"), query="これは日本語です")
        result3 = store.search(("user_123", "memories"), query="이건 한국어야")
        result4 = store.search(("user_123", "memories"), query="Это русский")
        result5 = store.search(("user_123", "memories"), query="यह रूसी है")

        assert result1[0].key == "1"
        assert result2[0].key == "2"
        assert result3[0].key == "3"
        assert result4[0].key == "4"
        assert result5[0].key == "5"


def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    """Compute cosine similarity between a vector X and a matrix Y."""
    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y, strict=False))
        norm1 = math.sqrt(sum(a * a for a in X))
        norm2 = math.sqrt(sum(a * a for a in y))
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)
    return similarities


# ─── Gap-coverage tests ─────────────────────────────────────────────────────


class TestTTLRefreshOnSearch:
    """Verify that TTL is refreshed when search returns items."""

    def test_search_refresh_extends_ttl(self) -> None:
        """Put an item with a short TTL, search (which should refresh),
        then verify the item survives past its original expiry.

        This catches the bug where ``c.ttl_minutes`` was missing from
        the search SELECT, causing the refresh to silently be skipped.
        """
        short_ttl_seconds = 10
        short_ttl_minutes = short_ttl_seconds / 60
        container_name = "store_test_ttlsearch"
        ttl_config = {
            "default_ttl": None,
            "refresh_on_read": True,
        }
        store = _make_store(container_name=container_name, ttl=ttl_config)
        store.setup()
        ns = ("ttl_search_test",)
        store.put(ns, key="item1", value={"data": "hello"}, ttl=short_ttl_minutes)

        # Wait a bit, then search — this should reset the TTL timer
        time.sleep(3)
        results = store.search(ns, refresh_ttl=True)
        assert len(results) == 1
        assert results[0].key == "item1"

        # Verify the raw document still has ttl_minutes
        raw_docs = list(
            store.container.query_items(
                query="SELECT c.ttl, c.ttl_minutes FROM c WHERE c.key = 'item1'",
                partition_key="ttl_search_test",
            )
        )
        assert len(raw_docs) == 1
        assert (
            raw_docs[0].get("ttl_minutes") is not None
        ), "ttl_minutes should be present in the document"
        assert (
            raw_docs[0].get("ttl") == short_ttl_seconds
        ), "ttl should have been refreshed to original value"


class TestTTLRefreshOnGet:
    """Verify that TTL is refreshed when get returns items."""

    def test_get_refresh_extends_ttl(self) -> None:
        """Put an item with a short TTL, get (which should refresh),
        then verify the TTL was re-applied.
        """
        short_ttl_seconds = 10
        short_ttl_minutes = short_ttl_seconds / 60
        container_name = "store_test_ttlget"
        ttl_config = {
            "default_ttl": None,
            "refresh_on_read": True,
        }
        store = _make_store(container_name=container_name, ttl=ttl_config)
        store.setup()
        ns = ("ttl_get_test",)
        store.put(ns, key="item1", value={"data": "hello"}, ttl=short_ttl_minutes)

        time.sleep(3)
        result = store.get(ns, key="item1", refresh_ttl=True)
        assert result is not None

        # Verify TTL was refreshed in the raw doc
        raw_docs = list(
            store.container.query_items(
                query="SELECT c.ttl, c.ttl_minutes FROM c WHERE c.key = 'item1'",
                partition_key="ttl_get_test",
            )
        )
        assert len(raw_docs) == 1
        assert raw_docs[0]["ttl"] == short_ttl_seconds


class TestSweeperMethods:
    """Tests for sweep_ttl / start_ttl_sweeper / stop_ttl_sweeper API."""

    def test_sweep_ttl_returns_int(self, store: CosmosDBStore) -> None:
        """sweep_ttl should return 0 (Cosmos handles TTL natively)."""
        result = store.sweep_ttl()
        assert result == 0
        assert isinstance(result, int)

    def test_start_ttl_sweeper_returns_future(self, store: CosmosDBStore) -> None:
        """start_ttl_sweeper should return a resolved Future."""
        import concurrent.futures

        future = store.start_ttl_sweeper()
        assert isinstance(future, concurrent.futures.Future)
        assert future.done()
        assert future.result() is None

    def test_stop_ttl_sweeper_returns_true(self, store: CosmosDBStore) -> None:
        """stop_ttl_sweeper should return True."""
        result = store.stop_ttl_sweeper()
        assert result is True
