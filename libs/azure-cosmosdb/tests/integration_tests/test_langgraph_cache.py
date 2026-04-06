# type: ignore
import os
from collections.abc import Iterator

import pytest
from langchain_azure_cosmosdb import CosmosDBCacheSync

pytestmark = pytest.mark.skipif(
    not os.getenv("COSMOSDB_ENDPOINT"),
    reason="COSMOSDB_ENDPOINT environment variable not set",
)


@pytest.fixture
def cache() -> Iterator[CosmosDBCacheSync]:
    database_name = os.getenv("COSMOSDB_TEST_DATABASE", "test_langgraph")
    container_name = "test_cache"
    c = CosmosDBCacheSync(
        database_name=database_name,
        container_name=container_name,
    )
    yield c


def test_set_and_get(cache: CosmosDBCacheSync) -> None:
    ns = ("test", "ns")
    key = "test_key_1"
    cache.set({(ns, key): ("hello world", None)})

    result = cache.get([(ns, key)])
    assert (ns, key) in result
    assert result[(ns, key)] == "hello world"


def test_get_missing_key(cache: CosmosDBCacheSync) -> None:
    ns = ("test", "missing")
    result = cache.get([(ns, "nonexistent")])
    assert result == {}


def test_set_with_ttl(cache: CosmosDBCacheSync) -> None:
    ns = ("test", "ttl")
    key = "ttl_key"
    cache.set({(ns, key): ("ttl_value", 3600)})

    result = cache.get([(ns, key)])
    assert (ns, key) in result
    assert result[(ns, key)] == "ttl_value"


def test_clear_namespace(cache: CosmosDBCacheSync) -> None:
    ns = ("test", "clearns")
    cache.set({(ns, "k1"): ("v1", None), (ns, "k2"): ("v2", None)})
    cache.clear(namespaces=[ns])
    result = cache.get([(ns, "k1"), (ns, "k2")])
    assert result == {}


def test_clear_all(cache: CosmosDBCacheSync) -> None:
    ns1 = ("test", "all1")
    ns2 = ("test", "all2")
    cache.set({(ns1, "k1"): ("v1", None), (ns2, "k2"): ("v2", None)})
    cache.clear()
    result = cache.get([(ns1, "k1"), (ns2, "k2")])
    assert result == {}
