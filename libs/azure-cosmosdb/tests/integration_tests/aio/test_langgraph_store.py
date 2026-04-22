# type: ignore
"""Integration tests for AsyncCosmosDBStore."""

from __future__ import annotations

import asyncio
import math
import os

import pytest
from langchain_azure_cosmosdb.aio import AsyncCosmosDBStore
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)

from tests.embed_test_utils import AsyncCharacterEmbeddings, CharacterEmbeddings

pytestmark = [
    pytest.mark.skipif(
        not os.getenv("COSMOSDB_CONN_STRING") and not os.getenv("COSMOSDB_ENDPOINT"),
        reason="COSMOSDB_CONN_STRING or COSMOSDB_ENDPOINT environment variable not set",
    ),
]

DEFAULT_DATABASE = os.getenv("COSMOSDB_TEST_DATABASE", "langgraph_test")


def _open_store(
    *,
    container_name: str,
    index: dict | None = None,
    ttl: dict | None = None,
):
    """Return an async context manager that creates an AsyncCosmosDBStore."""
    conn_string = os.getenv("COSMOSDB_CONN_STRING")
    if conn_string:
        return AsyncCosmosDBStore.from_conn_string(
            conn_string,
            database_name=DEFAULT_DATABASE,
            container_name=container_name,
            index=index,
            ttl=ttl,
        )
    endpoint = os.environ["COSMOSDB_ENDPOINT"]
    from azure.identity.aio import AzureCliCredential

    credential = AzureCliCredential(process_timeout=60)
    return AsyncCosmosDBStore.from_endpoint(
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
async def store():
    """Create a fresh AsyncCosmosDBStore for each test with TTL enabled."""
    container_name = "astore_test"
    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    async with _open_store(container_name=container_name, ttl=ttl_config) as store:
        await store.setup()
        yield store


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=500)


@pytest.fixture(scope="function")
async def vector_store(fake_embeddings: CharacterEmbeddings):
    """Create an AsyncCosmosDBStore with vector search enabled."""
    container_name = "astore_test_vec"
    index_config = {
        "dims": 500,
        "embed": fake_embeddings,
        "fields": ["text"],
    }
    async with _open_store(container_name=container_name, index=index_config) as store:
        await store.setup()
        yield store


class TestAsyncBatchOperations:
    """Tests for async batch operations."""

    async def test_batch_order(self, store: AsyncCosmosDBStore) -> None:
        """Test that async batch returns results in the correct order."""
        await store.aput(("test", "foo"), "key1", {"data": "value1"})

        ops = [
            GetOp(namespace=("test", "foo"), key="key1"),
            SearchOp(namespace_prefix=("test",), filter=None, limit=10, offset=0),
            ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
            GetOp(namespace=("test", "foo"), key="key2"),
        ]

        results = await store.abatch(ops)
        assert len(results) == 4

        assert isinstance(results[0], Item)
        assert results[0].key == "key1"
        assert isinstance(results[1], list)
        assert isinstance(results[2], list)
        assert results[3] is None

    async def test_batch_get_ops(self, store: AsyncCosmosDBStore) -> None:
        """Test getting items via async batch operations."""
        await store.aput(("test",), "key1", {"data": "value1"})
        await store.aput(("test",), "key2", {"data": "value2"})

        ops = [
            GetOp(namespace=("test",), key="key1"),
            GetOp(namespace=("test",), key="key2"),
            GetOp(namespace=("test",), key="key3"),
        ]
        results = await store.abatch(ops)

        assert results[0] is not None
        assert results[0].value == {"data": "value1"}
        assert results[1] is not None
        assert results[1].value == {"data": "value2"}
        assert results[2] is None

    async def test_batch_put_ops(self, store: AsyncCosmosDBStore) -> None:
        """Test putting items via async batch operations."""
        ops = [
            PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
            PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        ]
        await store.abatch(ops)

        item1 = await store.aget(("test",), "key1")
        assert item1 is not None
        assert item1.value == {"data": "value1"}

        item2 = await store.aget(("test",), "key2")
        assert item2 is not None
        assert item2.value == {"data": "value2"}


class TestAsyncBasicStoreOps:
    """Tests for basic async put/get/delete/search/list operations."""

    async def test_put_get(self, store: AsyncCosmosDBStore) -> None:
        """Test basic async put and get."""
        await store.aput(("test",), "key1", {"data": "value1"})
        item = await store.aget(("test",), "key1")
        assert item is not None
        assert item.value == {"data": "value1"}
        assert item.key == "key1"
        assert item.namespace == ("test",)

    async def test_put_update(self, store: AsyncCosmosDBStore) -> None:
        """Test updating an existing item async."""
        await store.aput(("test",), "key1", {"data": "value1"})
        await store.aput(("test",), "key1", {"data": "updated"})
        item = await store.aget(("test",), "key1")
        assert item is not None
        assert item.value == {"data": "updated"}

    async def test_put_delete(self, store: AsyncCosmosDBStore) -> None:
        """Test async deleting an item."""
        await store.aput(("test",), "key1", {"data": "value1"})
        await store.adelete(("test",), "key1")
        item = await store.aget(("test",), "key1")
        assert item is None

    async def test_delete_nonexistent(self, store: AsyncCosmosDBStore) -> None:
        """Test async deleting a non-existent item doesn't raise."""
        await store.adelete(("test",), "nonexistent")

    async def test_get_nonexistent(self, store: AsyncCosmosDBStore) -> None:
        """Test async getting a non-existent item returns None."""
        item = await store.aget(("test",), "nonexistent")
        assert item is None

    async def test_namespaced_isolation(self, store: AsyncCosmosDBStore) -> None:
        """Test that items in different namespaces are isolated."""
        await store.aput(("ns1",), "key1", {"data": "ns1_value"})
        await store.aput(("ns2",), "key1", {"data": "ns2_value"})

        item1 = await store.aget(("ns1",), "key1")
        item2 = await store.aget(("ns2",), "key1")
        assert item1 is not None
        assert item1.value == {"data": "ns1_value"}
        assert item2 is not None
        assert item2.value == {"data": "ns2_value"}


class TestAsyncSearch:
    """Tests for async search operations."""

    async def test_search_basic(self, store: AsyncCosmosDBStore) -> None:
        """Test basic async search by namespace prefix."""
        await store.aput(("test", "search"), "key1", {"data": "value1"})
        await store.aput(("test", "search"), "key2", {"data": "value2"})
        await store.aput(("other",), "key3", {"data": "value3"})

        results = await store.asearch(("test", "search"))
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"key1", "key2"}

    async def test_search_with_filter(self, store: AsyncCosmosDBStore) -> None:
        """Test async search with filter conditions."""
        await store.aput(("test",), "key1", {"status": "active", "score": 10})
        await store.aput(("test",), "key2", {"status": "inactive", "score": 20})
        await store.aput(("test",), "key3", {"status": "active", "score": 30})

        results = await store.asearch(("test",), filter={"status": "active"})
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"key1", "key3"}

    async def test_search_limit(self, store: AsyncCosmosDBStore) -> None:
        """Test async search with limit."""
        for i in range(5):
            await store.aput(("test",), f"key{i}", {"data": f"value{i}"})

        results = await store.asearch(("test",), limit=3)
        assert len(results) == 3

    async def test_search_empty(self, store: AsyncCosmosDBStore) -> None:
        """Test async search returns empty when no matches."""
        results = await store.asearch(("nonexistent",))
        assert results == []


class TestAsyncListNamespaces:
    """Tests for async list_namespaces operations."""

    async def test_list_namespaces(self, store: AsyncCosmosDBStore) -> None:
        """Test async listing all namespaces."""
        await store.aput(("a", "b"), "key1", {"data": "1"})
        await store.aput(("a", "c"), "key2", {"data": "2"})
        await store.aput(("d",), "key3", {"data": "3"})

        namespaces = await store.alist_namespaces(prefix=("a",))
        assert len(namespaces) >= 2
        assert ("a", "b") in namespaces
        assert ("a", "c") in namespaces

    async def test_list_namespaces_max_depth(self, store: AsyncCosmosDBStore) -> None:
        """Test async listing namespaces with max_depth."""
        await store.aput(("a", "b", "c"), "key1", {"data": "1"})
        await store.aput(("a", "b", "d"), "key2", {"data": "2"})
        await store.aput(("a", "x"), "key3", {"data": "3"})

        namespaces = await store.alist_namespaces(prefix=("a",), max_depth=2)
        assert ("a", "b") in namespaces
        assert ("a", "x") in namespaces
        assert ("a", "b", "c") not in namespaces


class TestAsyncVectorStore:
    """Tests for async vector search functionality."""

    async def test_vector_search(self, vector_store: AsyncCosmosDBStore) -> None:
        """Test async vector search."""
        await vector_store.aput(
            ("docs",), "doc1", {"text": "Python programming language"}
        )
        await vector_store.aput(
            ("docs",), "doc2", {"text": "JavaScript web development"}
        )
        await vector_store.aput(("docs",), "doc3", {"text": "Python data science"})

        results = await vector_store.asearch(("docs",), query="Python coding", limit=3)
        assert len(results) > 0
        for result in results:
            assert result.score is not None

    async def test_vector_search_with_filter(
        self, vector_store: AsyncCosmosDBStore
    ) -> None:
        """Test async vector search with filters."""
        await vector_store.aput(
            ("docs",),
            "doc1",
            {"text": "Python programming", "category": "lang"},
        )
        await vector_store.aput(
            ("docs",),
            "doc2",
            {"text": "JavaScript programming", "category": "lang"},
        )
        await vector_store.aput(
            ("docs",),
            "doc3",
            {"text": "Python data science", "category": "data"},
        )

        results = await vector_store.asearch(
            ("docs",), query="Python", filter={"category": "lang"}, limit=5
        )
        assert len(results) > 0
        for r in results:
            assert r.value.get("category") == "lang"


class TestAsyncNonAscii:
    """Tests for non-ASCII characters with async store."""

    async def test_non_ascii_namespace(self, store: AsyncCosmosDBStore) -> None:
        """Test non-ASCII characters in namespaces."""
        await store.aput(("用户", "数据"), "key1", {"name": "测试"})
        item = await store.aget(("用户", "数据"), "key1")
        assert item is not None
        assert item.value == {"name": "测试"}

    async def test_non_ascii_key(self, store: AsyncCosmosDBStore) -> None:
        """Test non-ASCII characters in keys."""
        await store.aput(("test",), "schlüssel", {"data": "wert"})
        item = await store.aget(("test",), "schlüssel")
        assert item is not None
        assert item.value == {"data": "wert"}

    async def test_non_ascii_value(self, store: AsyncCosmosDBStore) -> None:
        """Test non-ASCII characters in values."""
        value = {"name": "日本語テスト", "emoji": "🎉"}
        await store.aput(("test",), "key1", value)
        item = await store.aget(("test",), "key1")
        assert item is not None
        assert item.value == value


class TestAsyncBatchListNamespaces:
    """Tests for async batch list namespaces operations."""

    async def test_batch_list_namespaces_ops(self, store: AsyncCosmosDBStore) -> None:
        """Test async batch list namespace operations."""
        test_data = [
            (
                ("test", "documents", "public"),
                "doc1",
                {"content": "public doc"},
            ),
            (
                ("test", "documents", "private"),
                "doc2",
                {"content": "private doc"},
            ),
            (
                ("test", "images", "public"),
                "img1",
                {"content": "public image"},
            ),
            (
                ("prod", "documents", "public"),
                "doc3",
                {"content": "prod doc"},
            ),
        ]
        for namespace, key, value in test_data:
            await store.aput(namespace, key, value)

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

        results = await store.abatch(ops)
        assert len(results) == 3

        expected_ns = {ns for ns, _, _ in test_data}
        assert expected_ns.issubset(set(results[0]))
        assert all(len(ns) <= 2 for ns in results[1])
        assert all(ns[-1] == "public" for ns in results[2])


class TestAsyncScoresVerification:
    """Tests for async score verification."""

    @pytest.mark.parametrize("query", ["aaa", "bbb", "ccc", "abcd", "poisson"])
    async def test_scores_match_cosine(
        self,
        fake_embeddings: CharacterEmbeddings,
        query: str,
    ) -> None:
        """Test that async returned scores match manual cosine similarity."""
        container_name = "astore_test_scores"
        index_config = {
            "dims": 500,
            "embed": fake_embeddings,
            "fields": ["key0"],
        }
        async with _open_store(
            container_name=container_name, index=index_config
        ) as store:
            await store.setup()
            doc = {"key0": "aaa"}
            await store.aput(("test",), "doc", doc, index=["key0"])

            results = await store.asearch((), query=query)
            vec0 = fake_embeddings.embed_query(doc["key0"])
            vec1 = fake_embeddings.embed_query(query)
            similarities = _cosine_similarity(vec1, [vec0])

            assert len(results) == 1
            assert results[0].score == pytest.approx(similarities[0], abs=1e-3)


class TestAsyncNonAsciiWithVectorSearch:
    """Tests for non-ASCII characters with async vector search."""

    async def test_non_ascii_vector_search(
        self, fake_embeddings: CharacterEmbeddings
    ) -> None:
        """Test non-ASCII characters with async vector search."""
        container_name = "astore_test_noascii"
        index_config = {
            "dims": 500,
            "embed": fake_embeddings,
            "fields": ["text"],
        }
        async with _open_store(
            container_name=container_name, index=index_config
        ) as store:
            await store.setup()
            await store.aput(("user_123", "memories"), "1", {"text": "这是中文"})
            await store.aput(
                ("user_123", "memories"), "2", {"text": "これは日本語です"}
            )
            await store.aput(("user_123", "memories"), "3", {"text": "이건 한국어야"})

            result1 = await store.asearch(("user_123", "memories"), query="这是中文")
            result2 = await store.asearch(
                ("user_123", "memories"), query="これは日本語です"
            )
            result3 = await store.asearch(
                ("user_123", "memories"), query="이건 한국어야"
            )

            assert result1[0].key == "1"
            assert result2[0].key == "2"
            assert result3[0].key == "3"


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


class TestAsyncTTLRefreshOnSearch:
    """Verify that TTL is refreshed when async search returns items."""

    async def test_search_refresh_extends_ttl(self) -> None:
        """Put an item with a short TTL, search (which should refresh),
        then verify the raw document's TTL was re-applied.
        """
        short_ttl_seconds = 10
        short_ttl_minutes = short_ttl_seconds / 60
        container_name = "astore_test_ttlsearch"
        ttl_config = {
            "default_ttl": None,
            "refresh_on_read": True,
        }
        async with _open_store(container_name=container_name, ttl=ttl_config) as store:
            await store.setup()
            ns = ("ttl_search_test",)
            await store.aput(
                ns, key="item1", value={"data": "hello"}, ttl=short_ttl_minutes
            )

            await asyncio.sleep(3)
            results = await store.asearch(ns, refresh_ttl=True)
            assert len(results) == 1
            assert results[0].key == "item1"

            # Verify ttl_minutes is present and TTL was refreshed
            raw_docs = []
            async for doc in store.container.query_items(
                query="SELECT c.ttl, c.ttl_minutes FROM c WHERE c.key = 'item1'",
                partition_key="ttl_search_test",
            ):
                raw_docs.append(doc)
            assert len(raw_docs) == 1
            assert raw_docs[0].get("ttl_minutes") is not None
            assert raw_docs[0].get("ttl") == short_ttl_seconds


class TestAsyncSweeperMethods:
    """Tests for async sweep_ttl / start_ttl_sweeper / stop_ttl_sweeper."""

    async def test_sweep_ttl_returns_int(self, store: AsyncCosmosDBStore) -> None:
        result = await store.sweep_ttl()
        assert result == 0
        assert isinstance(result, int)

    async def test_start_ttl_sweeper_returns_task(
        self, store: AsyncCosmosDBStore
    ) -> None:
        task = await store.start_ttl_sweeper()
        assert isinstance(task, asyncio.Task)
        await task  # should complete immediately

    async def test_stop_ttl_sweeper_returns_true(
        self, store: AsyncCosmosDBStore
    ) -> None:
        result = await store.stop_ttl_sweeper()
        assert result is True


class TestAsyncContextManager:
    """Tests for __aenter__ / __aexit__ on direct instantiation."""

    async def test_direct_instantiation_context_manager(self) -> None:
        """AsyncCosmosDBStore should work as an async context manager
        even without using from_conn_string.
        """
        from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

        conn_string = os.getenv("COSMOSDB_CONN_STRING")
        if conn_string:
            client = AsyncCosmosClient.from_connection_string(conn_string)
        else:
            from azure.identity.aio import AzureCliCredential

            credential = AzureCliCredential(process_timeout=60)
            client = AsyncCosmosClient(
                os.environ["COSMOSDB_ENDPOINT"], credential=credential
            )
        try:
            store = AsyncCosmosDBStore(
                conn=client,
                database_name=DEFAULT_DATABASE,
                container_name="astore_test_ctx",
            )
            async with store as s:
                assert s is store
        finally:
            await client.close()


class TestAsyncEmbeddingsPath:
    """Verify the async store calls async embed methods, not sync ones."""

    async def test_aembed_called_on_put_and_search(self) -> None:
        """AsyncCharacterEmbeddings tracks sync vs async calls.
        After put + search the async path count should be > 0 and
        the sync path count should be 0.
        """
        container_name = "astore_test_aemb"
        emb = AsyncCharacterEmbeddings(dims=500)
        index_config = {
            "dims": 500,
            "embed": emb,
            "fields": ["text"],
        }
        async with _open_store(
            container_name=container_name, index=index_config
        ) as store:
            await store.setup()
            # Put triggers embedding generation via aembed_documents
            await store.aput(("emb_test",), "doc1", {"text": "hello world"})
            assert (
                emb.aembed_calls >= 1
            ), "aembed_documents should have been called for put"
            assert (
                emb.sync_embed_calls == 0
            ), "sync embed methods should NOT have been called"

            prev_async = emb.aembed_calls
            # Search triggers query embedding via aembed_query
            await store.asearch(("emb_test",), query="hello")
            assert (
                emb.aembed_calls > prev_async
            ), "aembed_query should have been called for search"
            assert (
                emb.sync_embed_calls == 0
            ), "sync embed methods should still NOT have been called"
