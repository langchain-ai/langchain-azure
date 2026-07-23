"""Shared helpers for delta-channel-history parity unit tests.

Provides an in-memory fake CosmosDB container (sync and async variants) plus
utilities for seeding synthetic parent chains, so the Cosmos saver overrides of
``get_delta_channel_history`` / ``aget_delta_channel_history`` can be compared
against the ``BaseCheckpointSaver`` default implementation.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langchain_azure_cosmosdb._langgraph_checkpoint_store import _CosmosSerializer

# Channel seeded at the chain root -> its history should carry a ``seed``.
DELTA_A = "delta_a"
# Channel never seeded -> the walk reaches the root, so ``seed`` is omitted.
DELTA_B = "delta_b"
CHANNELS = [DELTA_A, DELTA_B]

# Ancestor-chain depths exercised by the parity tests.
DEPTHS = [1, 10, 30, 60, 100]


def make_serde() -> _CosmosSerializer:
    """Return the same serializer the savers use by default."""
    return _CosmosSerializer(JsonPlusSerializer())


def make_checkpoint(checkpoint_id: str, channel_values: dict[str, Any]) -> Checkpoint:
    """Build a minimal valid ``Checkpoint`` with the given channel values."""
    return Checkpoint(
        v=1,
        id=checkpoint_id,
        ts="2024-01-01T00:00:00.000000+00:00",
        channel_values=channel_values,
        channel_versions={},
        versions_seen={},
        updated_channels=None,
    )


def run_query(
    store: dict[str, dict[str, Any]],
    query: str,
    parameters: list[dict[str, Any]] | None,
    partition_key: str | None,
) -> list[dict[str, Any]]:
    """Emulate ``container.query_items`` against an in-memory doc store.

    Recognizes the specific query shapes issued by the savers:

    * ``STARTSWITH(c.partition_key, @prefix)`` -> cross-partition writes scan.
    * ``... AND c.id=@checkpoint_key`` -> single checkpoint lookup.
    * ``SELECT TOP 1 c.id ... ORDER BY c.id DESC`` -> latest checkpoint id.
    * ``c.partition_key=@partition_key`` -> partition scan.
    """
    params = {p["name"]: p["value"] for p in (parameters or [])}

    if "STARTSWITH" in query:
        prefix = params["@prefix"]
        return [
            doc
            for doc in store.values()
            if doc.get("partition_key", "").startswith(prefix)
        ]

    pk = params.get("@partition_key")
    results = [doc for doc in store.values() if doc.get("partition_key") == pk]

    if "@checkpoint_key" in params:
        results = [doc for doc in results if doc.get("id") == params["@checkpoint_key"]]

    if "ORDER BY c.id DESC" in query:
        results = sorted(results, key=lambda doc: doc["id"], reverse=True)

    if "TOP 1" in query:
        return [{"id": doc["id"]} for doc in results[:1]]

    return results


class FakeSyncContainer:
    """In-memory synchronous stand-in for a CosmosDB container proxy."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    def upsert_item(self, data: dict[str, Any]) -> dict[str, Any]:
        self.store[data["id"]] = dict(data)
        return data

    def create_item(self, data: dict[str, Any]) -> dict[str, Any]:
        self.store[data["id"]] = dict(data)
        return data

    def query_items(
        self,
        *,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | None = None,
    ) -> list[dict[str, Any]]:
        return run_query(self.store, query, parameters, partition_key)


class _AsyncResults:
    """Async-iterable wrapper over a list of query results."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def __aiter__(self) -> "_AsyncResults":
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeAsyncContainer:
    """In-memory asynchronous stand-in for a CosmosDB container proxy."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    async def upsert_item(self, data: dict[str, Any]) -> dict[str, Any]:
        self.store[data["id"]] = dict(data)
        return data

    async def create_item(self, data: dict[str, Any]) -> dict[str, Any]:
        self.store[data["id"]] = dict(data)
        return data

    def query_items(
        self,
        *,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | None = None,
    ) -> _AsyncResults:
        return _AsyncResults(run_query(self.store, query, parameters, partition_key))
