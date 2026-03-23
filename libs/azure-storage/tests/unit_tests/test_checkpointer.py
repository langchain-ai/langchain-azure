# mypy: disable-error-code="arg-type,attr-defined,index,union-attr"
"""Unit tests for AzureTableStorageSaver with mocked Azure Table clients."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_azure_storage._table_utils import (
    chunk_data,
    make_checkpoint_row_key,
    make_writes_row_key,
)
from langchain_azure_storage.checkpointer import AzureTableStorageSaver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PATCH_TARGET = "azure.data.tables.TableServiceClient"


def _make_saver() -> AzureTableStorageSaver:
    """Create an ``AzureTableStorageSaver`` with fully mocked Azure clients."""
    with patch(_PATCH_TARGET) as mock_cls:
        mock_service = MagicMock()
        mock_cls.from_connection_string.return_value = mock_service
        mock_service.get_table_client.side_effect = lambda name: MagicMock(
            name=f"table_{name}"
        )
        saver = AzureTableStorageSaver(conn_str="fake-conn-str")
    return saver


def _make_checkpoint(
    checkpoint_id: str = "1ef4f797-8335-6428-8001-8a1503f9b875",
) -> dict[str, Any]:
    return {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": checkpoint_id,
        "channel_values": {"my_key": "meow", "node": "node"},
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3,
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
            "node": {"start:node": 2},
        },
        "pending_sends": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaverInit:
    def test_conn_string(self) -> None:
        saver = _make_saver()
        assert saver._checkpoint_table is not None
        assert saver._writes_table is not None

    def test_missing_params_raises(self) -> None:
        with patch(_PATCH_TARGET):
            with pytest.raises(ValueError, match="Provide either"):
                AzureTableStorageSaver()

    def test_from_conn_string_context_manager(self) -> None:
        with patch(_PATCH_TARGET) as mock_cls:
            mock_service = MagicMock()
            mock_cls.from_connection_string.return_value = mock_service
            mock_service.get_table_client.return_value = MagicMock()

            with AzureTableStorageSaver.from_conn_string("fake") as saver:
                assert saver is not None
            mock_service.close.assert_called_once()


class TestSetup:
    def test_creates_tables(self) -> None:
        saver = _make_saver()
        saver.setup()
        calls = saver._service_client.create_table_if_not_exists.call_args_list
        assert len(calls) == 2
        table_names = {
            c.args[0] if c.args else c.kwargs.get("table_name") for c in calls
        }
        assert "checkpoints" in table_names or any(
            "checkpoints" in str(c) for c in calls
        )


class TestPut:
    def test_put_stores_entity(self) -> None:
        saver = _make_saver()
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "prev-id",
            }
        }
        checkpoint = _make_checkpoint()
        metadata = {"source": "input", "step": 1, "writes": {}, "parents": {}}

        result = saver.put(config, checkpoint, metadata, {})

        saver._checkpoint_table.upsert_entity.assert_called_once()
        entity = saver._checkpoint_table.upsert_entity.call_args[0][0]
        assert entity["PartitionKey"] == "thread-1"
        assert "type" in entity
        assert "chunk_count" in entity
        assert entity["parent_checkpoint_id"] == "prev-id"
        assert result["configurable"]["thread_id"] == "thread-1"
        assert result["configurable"]["checkpoint_id"] == checkpoint["id"]

    def test_put_returns_correct_config(self) -> None:
        saver = _make_saver()
        config = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "ns1",
            }
        }
        checkpoint = _make_checkpoint("my-id")
        result = saver.put(config, checkpoint, {}, {})
        assert result == {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "my-id",
            }
        }


class TestPutWrites:
    def test_put_writes_stores_entities(self) -> None:
        saver = _make_saver()
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        }
        writes = [("channel_a", "value_a"), ("channel_b", "value_b")]

        saver.put_writes(config, writes, task_id="task-1")

        assert saver._writes_table.upsert_entity.call_count == 2
        first_entity = saver._writes_table.upsert_entity.call_args_list[0][0][0]
        assert first_entity["PartitionKey"] == "thread-1"
        assert first_entity["channel"] == "channel_a"
        assert first_entity["task_id"] == "task-1"


class TestGetTuple:
    def test_get_with_checkpoint_id(self) -> None:
        saver = _make_saver()
        checkpoint = _make_checkpoint()
        type_, data = saver.serde.dumps_typed(checkpoint)
        chunks = chunk_data(data)

        entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_checkpoint_row_key("", checkpoint["id"]),
            "type": type_,
            "metadata": json.dumps({"source": "input", "step": 1}),
            "parent_checkpoint_id": "",
            **chunks,
        }
        saver._checkpoint_table.get_entity.return_value = entity
        saver._writes_table.query_entities.return_value = []

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }
        result = saver.get_tuple(config)

        assert result is not None
        assert result.config["configurable"]["thread_id"] == "thread-1"
        assert result.config["configurable"]["checkpoint_id"] == checkpoint["id"]
        assert result.checkpoint["id"] == checkpoint["id"]

    def test_get_latest(self) -> None:
        saver = _make_saver()
        checkpoint = _make_checkpoint()
        type_, data = saver.serde.dumps_typed(checkpoint)
        chunks = chunk_data(data)

        entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_checkpoint_row_key("", checkpoint["id"]),
            "type": type_,
            "metadata": "{}",
            "parent_checkpoint_id": "",
            **chunks,
        }
        saver._checkpoint_table.query_entities.return_value = [entity]
        saver._writes_table.query_entities.return_value = []

        config = {"configurable": {"thread_id": "thread-1"}}
        result = saver.get_tuple(config)

        assert result is not None
        assert result.checkpoint["id"] == checkpoint["id"]

    def test_get_returns_none_on_miss(self) -> None:
        saver = _make_saver()
        saver._checkpoint_table.get_entity.side_effect = Exception("Not found")

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "nonexistent",
            }
        }
        result = saver.get_tuple(config)
        assert result is None

    def test_get_with_pending_writes(self) -> None:
        saver = _make_saver()
        checkpoint = _make_checkpoint()
        type_, data = saver.serde.dumps_typed(checkpoint)
        chunks = chunk_data(data)

        entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_checkpoint_row_key("", checkpoint["id"]),
            "type": type_,
            "metadata": "{}",
            "parent_checkpoint_id": "",
            **chunks,
        }
        saver._checkpoint_table.get_entity.return_value = entity

        # Create a pending-write entity
        write_type, write_data = saver.serde.dumps_typed("hello")
        write_chunks = chunk_data(write_data)
        write_entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_writes_row_key("", checkpoint["id"], "task-1", 0),
            "task_id": "task-1",
            "channel": "my_channel",
            "type": write_type,
            **write_chunks,
        }
        saver._writes_table.query_entities.return_value = [write_entity]

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }
        result = saver.get_tuple(config)

        assert result is not None
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0][0] == "task-1"
        assert result.pending_writes[0][1] == "my_channel"
        assert result.pending_writes[0][2] == "hello"


class TestList:
    def test_list_returns_checkpoints(self) -> None:
        saver = _make_saver()
        checkpoint = _make_checkpoint()
        type_, data = saver.serde.dumps_typed(checkpoint)
        chunks = chunk_data(data)

        entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_checkpoint_row_key("", checkpoint["id"]),
            "type": type_,
            "metadata": json.dumps({"source": "loop", "step": 1}),
            "parent_checkpoint_id": "",
            **chunks,
        }
        saver._checkpoint_table.query_entities.return_value = [entity]
        saver._writes_table.query_entities.return_value = []

        config = {"configurable": {"thread_id": "thread-1"}}
        results = list(saver.list(config))

        assert len(results) == 1
        assert results[0].checkpoint["id"] == checkpoint["id"]

    def test_list_with_limit(self) -> None:
        saver = _make_saver()
        entities = []
        for i in range(5):
            cp = _make_checkpoint(f"id-{i:03d}")
            t, d = saver.serde.dumps_typed(cp)
            ch = chunk_data(d)
            entities.append(
                {
                    "PartitionKey": "thread-1",
                    "RowKey": make_checkpoint_row_key("", cp["id"]),
                    "type": t,
                    "metadata": "{}",
                    "parent_checkpoint_id": "",
                    **ch,
                }
            )
        saver._checkpoint_table.query_entities.return_value = entities
        saver._writes_table.query_entities.return_value = []

        config = {"configurable": {"thread_id": "thread-1"}}
        results = list(saver.list(config, limit=2))
        assert len(results) == 2

    def test_list_with_metadata_filter(self) -> None:
        saver = _make_saver()

        # Two checkpoints with different metadata
        cp1 = _make_checkpoint("id-001")
        t1, d1 = saver.serde.dumps_typed(cp1)
        ch1 = chunk_data(d1)
        meta1 = {"source": "input", "step": 0}

        cp2 = _make_checkpoint("id-002")
        t2, d2 = saver.serde.dumps_typed(cp2)
        ch2 = chunk_data(d2)
        meta2 = {"source": "loop", "step": 1}

        entities = [
            {
                "PartitionKey": "thread-1",
                "RowKey": make_checkpoint_row_key("", "id-001"),
                "type": t1,
                "metadata": json.dumps(meta1),
                "parent_checkpoint_id": "",
                **ch1,
            },
            {
                "PartitionKey": "thread-1",
                "RowKey": make_checkpoint_row_key("", "id-002"),
                "type": t2,
                "metadata": json.dumps(meta2),
                "parent_checkpoint_id": "",
                **ch2,
            },
        ]
        saver._checkpoint_table.query_entities.return_value = entities
        saver._writes_table.query_entities.return_value = []

        config = {"configurable": {"thread_id": "thread-1"}}
        results = list(saver.list(config, filter={"source": "loop"}))
        assert len(results) == 1
        assert results[0].metadata["source"] == "loop"


class TestDeleteThread:
    def test_deletes_all_entities(self) -> None:
        saver = _make_saver()

        cp_entities = [
            {"PartitionKey": "thread-1", "RowKey": "rk-1"},
            {"PartitionKey": "thread-1", "RowKey": "rk-2"},
        ]
        wr_entities = [
            {"PartitionKey": "thread-1", "RowKey": "wrk-1"},
        ]

        # First call returns checkpoint entities, second returns writes
        saver._checkpoint_table.query_entities.return_value = cp_entities
        saver._writes_table.query_entities.return_value = wr_entities

        saver.delete_thread("thread-1")

        assert saver._checkpoint_table.delete_entity.call_count == 2
        assert saver._writes_table.delete_entity.call_count == 1


class TestParentConfig:
    def test_parent_config_set_when_parent_exists(self) -> None:
        saver = _make_saver()
        checkpoint = _make_checkpoint()
        type_, data = saver.serde.dumps_typed(checkpoint)
        chunks = chunk_data(data)

        entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_checkpoint_row_key("", checkpoint["id"]),
            "type": type_,
            "metadata": "{}",
            "parent_checkpoint_id": "parent-id-123",
            **chunks,
        }
        saver._checkpoint_table.get_entity.return_value = entity
        saver._writes_table.query_entities.return_value = []

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }
        result = saver.get_tuple(config)

        assert result is not None
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "parent-id-123"

    def test_parent_config_none_when_no_parent(self) -> None:
        saver = _make_saver()
        checkpoint = _make_checkpoint()
        type_, data = saver.serde.dumps_typed(checkpoint)
        chunks = chunk_data(data)

        entity = {
            "PartitionKey": "thread-1",
            "RowKey": make_checkpoint_row_key("", checkpoint["id"]),
            "type": type_,
            "metadata": "{}",
            "parent_checkpoint_id": "",
            **chunks,
        }
        saver._checkpoint_table.get_entity.return_value = entity
        saver._writes_table.query_entities.return_value = []

        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }
        result = saver.get_tuple(config)

        assert result is not None
        assert result.parent_config is None
