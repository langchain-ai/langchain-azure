"""Tests that checkpoint saver methods push sorting, filtering, and limiting
to CosmosDB server-side queries instead of fetching all items and processing
in Python.

These tests mock the CosmosDB container to capture the SQL queries issued and
verify that ORDER BY, TOP, and WHERE clauses are used appropriately.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
    CosmosDBSaverSync,
    _CosmosSerializer,
    _make_checkpoint_key,
    _make_checkpoint_writes_key,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64(val: str) -> str:
    return base64.b64encode(val.encode()).decode()


def _make_serde() -> _CosmosSerializer:
    return _CosmosSerializer(JsonPlusSerializer())


def _make_fake_checkpoint_item(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str, serde: _CosmosSerializer
) -> dict[str, Any]:
    """Build a checkpoint document as CosmosDB would store it."""
    key = _make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
    partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")
    cp_type, cp_data = serde.dumps_typed(
        {
            "v": 1,
            "id": checkpoint_id,
            "ts": checkpoint_id,
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
    )
    md_data = serde.dumps_typed({"source": "input", "step": 1})
    return {
        "id": key,
        "partition_key": partition_key,
        "thread_id": thread_id,
        "checkpoint": cp_data,
        "type": cp_type,
        "metadata": md_data,
        "parent_checkpoint_id": "",
    }


def _make_fake_write_item(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int,
    channel: str,
    serde: _CosmosSerializer,
) -> dict[str, Any]:
    key = _make_checkpoint_writes_key(
        thread_id, checkpoint_ns, checkpoint_id, task_id, idx
    )
    partition_key = _make_checkpoint_writes_key(
        thread_id, checkpoint_ns, checkpoint_id, "", None
    )
    type_, value = serde.dumps_typed(f"value-{idx}")
    return {
        "id": key,
        "partition_key": partition_key,
        "thread_id": thread_id,
        "channel": channel,
        "type": type_,
        "value": value,
    }


def _build_saver_with_mock_container() -> tuple[CosmosDBSaverSync, MagicMock]:
    """Construct a CosmosDBSaverSync with a mocked CosmosDB container."""
    with patch.object(CosmosDBSaverSync, "__init__", lambda self: None):
        saver = CosmosDBSaverSync.__new__(CosmosDBSaverSync)
    saver.serde = JsonPlusSerializer()
    saver.cosmos_serde = _CosmosSerializer(saver.serde)
    container = MagicMock()
    saver.container = container
    # Also set client/database to avoid AttributeError in any code path
    saver.client = MagicMock()
    saver.database = MagicMock()
    return saver, container


# ===================================================================
# _get_checkpoint_key: should use TOP 1 + ORDER BY DESC
# ===================================================================


class TestGetCheckpointKeyServerSide:
    def test_with_explicit_id_skips_query(self) -> None:
        saver, container = _build_saver_with_mock_container()
        result = saver._get_checkpoint_key(container, "t1", "ns", "cp-123")
        assert result == _make_checkpoint_key("t1", "ns", "cp-123")
        container.query_items.assert_not_called()

    def test_without_id_uses_top_1_order_by(self) -> None:
        """When no checkpoint_id is given, the query should use
        ORDER BY c.id DESC and TOP 1 to let CosmosDB return only
        the latest checkpoint instead of fetching all."""
        saver, container = _build_saver_with_mock_container()
        latest_key = _make_checkpoint_key("t1", "ns", "cp-003")
        container.query_items.return_value = [{"id": latest_key}]

        result = saver._get_checkpoint_key(container, "t1", "ns", None)

        assert result == latest_key
        call_args = container.query_items.call_args
        query = call_args.kwargs.get("query") or call_args[1].get("query")
        assert "ORDER BY" in query.upper(), f"Expected ORDER BY in query, got: {query}"
        assert (
            "TOP" in query.upper() or "OFFSET" in query.upper()
        ), f"Expected TOP or OFFSET 0 LIMIT 1 in query, got: {query}"

    def test_without_id_returns_none_when_empty(self) -> None:
        saver, container = _build_saver_with_mock_container()
        container.query_items.return_value = []
        result = saver._get_checkpoint_key(container, "t1", "ns", None)
        assert result is None


# ===================================================================
# list: should push ORDER BY, before-filter, and limit to CosmosDB
# ===================================================================


class TestListServerSide:
    def test_list_uses_order_by(self) -> None:
        """list() should push ORDER BY to CosmosDB, not sort in Python."""
        saver, container = _build_saver_with_mock_container()
        serde = saver.cosmos_serde
        items = [
            _make_fake_checkpoint_item("t1", "", f"cp-{i:03d}", serde) for i in range(3)
        ]
        # First call returns checkpoint items; subsequent calls return no writes
        container.query_items.side_effect = [items] + [[] for _ in range(len(items))]

        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        list(saver.list(config))

        call_args = container.query_items.call_args_list[0]
        query = call_args.kwargs.get("query") or call_args[1].get("query")
        assert "ORDER BY" in query.upper(), f"Expected ORDER BY in query, got: {query}"

    def test_list_pushes_before_filter_to_query(self) -> None:
        """When 'before' is specified, it should be in the WHERE clause."""
        saver, container = _build_saver_with_mock_container()
        serde = saver.cosmos_serde
        items = [
            _make_fake_checkpoint_item("t1", "", "cp-001", serde),
        ]
        # First call returns checkpoint items; second returns no writes
        container.query_items.side_effect = [items, []]

        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        before: RunnableConfig = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-002",
            }
        }
        list(saver.list(config, before=before))

        call_args = container.query_items.call_args_list[0]
        query = call_args.kwargs.get("query") or call_args[1].get("query")
        # The before checkpoint_id should appear in the query as a filter
        params = call_args.kwargs.get("parameters") or call_args[1].get("parameters")
        param_values = [p["value"] for p in params]
        assert "cp-002" in query or "cp-002" in str(
            param_values
        ), f"Expected before_id in query or params. Query: {query}, Params: {params}"

    def test_list_pushes_limit_to_query(self) -> None:
        """When 'limit' is specified, the query should use TOP or LIMIT."""
        saver, container = _build_saver_with_mock_container()
        serde = saver.cosmos_serde
        items = [_make_fake_checkpoint_item("t1", "", "cp-001", serde)]
        # First call returns checkpoint items; second returns no writes
        container.query_items.side_effect = [items, []]

        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        list(saver.list(config, limit=5))

        call_args = container.query_items.call_args_list[0]
        query = call_args.kwargs.get("query") or call_args[1].get("query")
        assert (
            "TOP" in query.upper() or "LIMIT" in query.upper()
        ), f"Expected TOP or LIMIT in query, got: {query}"

    def test_list_returns_results_in_descending_order(self) -> None:
        """Results should come back newest first regardless of storage order."""
        saver, container = _build_saver_with_mock_container()
        serde = saver.cosmos_serde
        # Simulate CosmosDB returning items already sorted DESC by ORDER BY
        items = [
            _make_fake_checkpoint_item("t1", "", f"cp-{i:03d}", serde)
            for i in [3, 2, 1]
        ]
        # First call returns checkpoint items; subsequent calls return no writes
        container.query_items.side_effect = [items] + [[] for _ in range(len(items))]

        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        results = list(saver.list(config))

        ids = [r.config["configurable"]["checkpoint_id"] for r in results]
        assert ids == sorted(
            ids, reverse=True
        ), f"Expected descending order, got: {ids}"


# ===================================================================
# _load_pending_writes: should use ORDER BY for idx
# ===================================================================


class TestLoadPendingWritesServerSide:
    def test_uses_order_by_for_idx(self) -> None:
        """_load_pending_writes should ORDER BY in the query, not sort
        in Python after fetching all writes."""
        saver, container = _build_saver_with_mock_container()
        serde = saver.cosmos_serde
        writes = [
            _make_fake_write_item("t1", "", "cp-001", "task1", i, f"ch{i}", serde)
            for i in range(3)
        ]
        container.query_items.return_value = writes

        saver._load_pending_writes("t1", "", "cp-001")

        call_args = container.query_items.call_args
        query = call_args.kwargs.get("query") or call_args[1].get("query")
        assert "ORDER BY" in query.upper(), f"Expected ORDER BY in query, got: {query}"

    def test_returns_writes_in_idx_order(self) -> None:
        """Writes must come back sorted by idx. Since we now rely on
        server-side ORDER BY, the mock simulates correct server ordering."""
        saver, container = _build_saver_with_mock_container()
        serde = saver.cosmos_serde
        # Server returns writes already sorted by id ASC (via ORDER BY)
        writes = [
            _make_fake_write_item("t1", "", "cp-001", "task1", i, f"ch{i}", serde)
            for i in [0, 1, 2]
        ]
        container.query_items.return_value = writes

        result = saver._load_pending_writes("t1", "", "cp-001")

        channels = [r[1] for r in result]
        assert channels == ["ch0", "ch1", "ch2"]
