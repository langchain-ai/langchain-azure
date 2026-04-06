"""Unit tests for the async CosmosDB LangGraph cache."""

import base64
import datetime
from unittest.mock import AsyncMock, MagicMock

from langchain_azure_cosmosdb._langgraph_cache import _make_cache_key

# ---- helpers ---------------------------------------------------------------


def _make_serde() -> MagicMock:
    serde = MagicMock()
    serde.loads_typed = MagicMock(side_effect=lambda pair: f"decoded:{pair[1]!r}")
    serde.dumps_typed = MagicMock(return_value=("json", b'{"val": 1}'))
    return serde


def _make_cache(container: AsyncMock | None = None) -> "CosmosDBCache":  # type: ignore[name-defined]  # noqa: F821
    from langchain_azure_cosmosdb.aio._langgraph_cache import CosmosDBCache

    if container is None:
        container = AsyncMock()
    serde = _make_serde()
    cache = CosmosDBCache(container=container, serde=serde)
    return cache


# ---- aget (mock-based) ----------------------------------------------------


async def test_aget_returns_deserialized_value() -> None:
    container = AsyncMock()
    raw_value = b'{"val": 1}'
    encoded_val = base64.b64encode(raw_value).decode("utf-8")
    container.read_item = AsyncMock(
        return_value={
            "encoding": "json",
            "val": encoded_val,
            "expiry": None,
        }
    )

    cache = _make_cache(container)

    ns = ("my_ns",)
    key = "key1"
    result = await cache.aget([(ns, key)])

    expected_doc_id = _make_cache_key("my_ns", "key1")
    container.read_item.assert_awaited_once_with(
        item=expected_doc_id, partition_key="my_ns"
    )
    assert (ns, key) in result


async def test_aget_empty_keys_returns_empty() -> None:
    cache = _make_cache()
    result = await cache.aget([])
    assert result == {}


async def test_aget_expired_item_is_deleted() -> None:
    container = AsyncMock()
    past_ts = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    container.read_item = AsyncMock(
        return_value={
            "encoding": "json",
            "val": base64.b64encode(b"x").decode(),
            "expiry": past_ts,
        }
    )
    container.delete_item = AsyncMock()

    cache = _make_cache(container)
    result = await cache.aget([(("ns",), "k")])

    assert result == {}
    container.delete_item.assert_awaited_once()


# ---- aset (mock-based) ----------------------------------------------------


async def test_aset_calls_upsert_item() -> None:
    container = AsyncMock()
    cache = _make_cache(container)

    ns = ("ns1",)
    key = "k1"
    value = {"data": "test"}

    await cache.aset({(ns, key): (value, None)})

    container.upsert_item.assert_awaited_once()
    upserted = container.upsert_item.call_args[0][0]
    assert upserted["id"] == _make_cache_key("ns1", "k1")
    assert upserted["ns"] == "ns1"
    assert upserted["key"] == "k1"
    assert upserted["expiry"] is None
    assert "val" in upserted
    assert "encoding" in upserted


async def test_aset_with_ttl_sets_expiry() -> None:
    container = AsyncMock()
    cache = _make_cache(container)

    await cache.aset({(("ns",), "k"): ("v", 60)})

    upserted = container.upsert_item.call_args[0][0]
    assert upserted["expiry"] is not None
    assert isinstance(upserted["expiry"], float)


# ---- sync bridge methods exist --------------------------------------------


def test_sync_bridge_methods_exist() -> None:
    from langchain_azure_cosmosdb.aio._langgraph_cache import CosmosDBCache

    assert hasattr(CosmosDBCache, "get")
    assert hasattr(CosmosDBCache, "set")
    assert hasattr(CosmosDBCache, "clear")
    assert callable(getattr(CosmosDBCache, "get"))
    assert callable(getattr(CosmosDBCache, "set"))
    assert callable(getattr(CosmosDBCache, "clear"))
