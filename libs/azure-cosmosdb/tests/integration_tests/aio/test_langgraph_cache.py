# type: ignore
"""Integration tests for async CosmosDBCache (LangGraph)."""

import os

import pytest

HOST = os.environ.get("COSMOSDB_ENDPOINT", "")
KEY = os.environ.get("COSMOSDB_KEY", "")

pytestmark = pytest.mark.skipif(
    not HOST or not KEY,
    reason="COSMOSDB_ENDPOINT/COSMOSDB_KEY not set",
)

DATABASE_NAME = os.environ.get("COSMOSDB_TEST_DATABASE", "test_langgraph_async")
CONTAINER_NAME = "test_async_cache"


async def test_async_set_and_get() -> None:
    from langchain_azure_cosmosdb.aio import CosmosDBCache

    async with CosmosDBCache.from_conn_info(
        endpoint=HOST,
        key=KEY,
        database_name=DATABASE_NAME,
        container_name=CONTAINER_NAME,
    ) as cache:
        ns = ("test", "async_ns")
        key = "test_key_1"
        await cache.aset({(ns, key): ("hello world", None)})

        result = await cache.aget([(ns, key)])
        assert (ns, key) in result
        assert result[(ns, key)] == "hello world"

        await cache.aclear()


async def test_async_get_missing_key() -> None:
    from langchain_azure_cosmosdb.aio import CosmosDBCache

    async with CosmosDBCache.from_conn_info(
        endpoint=HOST,
        key=KEY,
        database_name=DATABASE_NAME,
        container_name=CONTAINER_NAME,
    ) as cache:
        ns = ("test", "async_missing")
        result = await cache.aget([(ns, "nonexistent")])
        assert result == {}


async def test_async_set_with_ttl() -> None:
    from langchain_azure_cosmosdb.aio import CosmosDBCache

    async with CosmosDBCache.from_conn_info(
        endpoint=HOST,
        key=KEY,
        database_name=DATABASE_NAME,
        container_name=CONTAINER_NAME,
    ) as cache:
        ns = ("test", "async_ttl")
        key = "ttl_key"
        await cache.aset({(ns, key): ("ttl_value", 3600)})

        result = await cache.aget([(ns, key)])
        assert (ns, key) in result
        assert result[(ns, key)] == "ttl_value"

        await cache.aclear()


async def test_async_clear_namespace() -> None:
    from langchain_azure_cosmosdb.aio import CosmosDBCache

    async with CosmosDBCache.from_conn_info(
        endpoint=HOST,
        key=KEY,
        database_name=DATABASE_NAME,
        container_name=CONTAINER_NAME,
    ) as cache:
        ns = ("test", "async_clearns")
        await cache.aset({(ns, "k1"): ("v1", None), (ns, "k2"): ("v2", None)})
        await cache.aclear(namespaces=[ns])
        result = await cache.aget([(ns, "k1"), (ns, "k2")])
        assert result == {}


async def test_async_clear_all() -> None:
    from langchain_azure_cosmosdb.aio import CosmosDBCache

    async with CosmosDBCache.from_conn_info(
        endpoint=HOST,
        key=KEY,
        database_name=DATABASE_NAME,
        container_name=CONTAINER_NAME,
    ) as cache:
        ns1 = ("test", "async_all1")
        ns2 = ("test", "async_all2")
        await cache.aset({(ns1, "k1"): ("v1", None), (ns2, "k2"): ("v2", None)})
        await cache.aclear()
        result = await cache.aget([(ns1, "k1"), (ns2, "k2")])
        assert result == {}
