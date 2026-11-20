# type: ignore
"""Integration tests for the Azure Table Storage LangGraph checkpointer.

These run against a live storage account or the Azurite emulator; see
``conftest.py`` for the environment variables that select the target.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Optional

import pytest
from azure.data.tables import TableServiceClient
from azure.data.tables.aio import TableServiceClient as AsyncTableServiceClient
from azure.identity import DefaultAzureCredential

from langchain_azure_storage.checkpoint import AzureTableStorageSaver

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("AZURE_STORAGE_TABLE_ENDPOINT")
        or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    ),
    reason=(
        "Set AZURE_STORAGE_TABLE_ENDPOINT (live account) or "
        "AZURE_STORAGE_CONNECTION_STRING (e.g. Azurite) to run these tests."
    ),
)


def _table_name() -> str:
    return f"testcp{uuid.uuid4().hex[:20]}"


@pytest.fixture
def sync_saver(
    table_endpoint: Optional[str], connection_string: Optional[str]
) -> Iterator[AzureTableStorageSaver]:
    table_name = _table_name()
    if table_endpoint:
        saver = AzureTableStorageSaver(
            table_endpoint, table_name, credential=DefaultAzureCredential()
        )
    else:
        assert connection_string is not None
        saver = AzureTableStorageSaver.from_connection_string(
            connection_string, table_name
        )
    yield saver
    saver.close()
    _delete_table_sync(table_name, table_endpoint, connection_string)


@pytest.fixture
async def async_saver(
    table_endpoint: Optional[str], connection_string: Optional[str]
) -> AsyncIterator[AzureTableStorageSaver]:
    table_name = _table_name()
    if table_endpoint:
        from azure.identity.aio import (
            DefaultAzureCredential as AsyncDefaultAzureCredential,
        )

        saver = AzureTableStorageSaver(
            table_endpoint, table_name, credential=AsyncDefaultAzureCredential()
        )
    else:
        assert connection_string is not None
        saver = AzureTableStorageSaver.from_connection_string(
            connection_string, table_name
        )
    yield saver
    await saver.aclose()
    await _delete_table_async(table_name, table_endpoint, connection_string)


def _delete_table_sync(
    table_name: str, table_endpoint: Optional[str], connection_string: Optional[str]
) -> None:
    if table_endpoint:
        client = TableServiceClient(table_endpoint, credential=DefaultAzureCredential())
    else:
        assert connection_string is not None
        client = TableServiceClient.from_connection_string(connection_string)
    with client:
        client.delete_table(table_name)


async def _delete_table_async(
    table_name: str, table_endpoint: Optional[str], connection_string: Optional[str]
) -> None:
    if table_endpoint:
        from azure.identity.aio import (
            DefaultAzureCredential as AsyncDefaultAzureCredential,
        )

        client = AsyncTableServiceClient(
            table_endpoint, credential=AsyncDefaultAzureCredential()
        )
    else:
        assert connection_string is not None
        client = AsyncTableServiceClient.from_connection_string(connection_string)
    async with client:
        await client.delete_table(table_name)


def _checkpoint(cp_id: str, **channel_values: object) -> dict:
    return {
        "v": 1,
        "id": cp_id,
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": channel_values,
        "channel_versions": dict.fromkeys(channel_values, 1),
        "versions_seen": {},
        "updated_channels": None,
    }


def test_sync_put_and_get_tuple(sync_saver: AzureTableStorageSaver) -> None:
    tid = f"sync_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cpid}
    }

    result_config = sync_saver.put(
        config, _checkpoint(cpid, test_key="test_value"), {"source": "input"}, {}
    )
    assert result_config["configurable"]["thread_id"] == tid

    retrieved = sync_saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == cpid
    assert retrieved.checkpoint["channel_values"] == {"test_key": "test_value"}
    assert retrieved.metadata["source"] == "input"


def test_sync_list_orders_most_recent_first(sync_saver: AzureTableStorageSaver) -> None:
    tid = f"sync_list_{uuid.uuid4().hex[:8]}"
    cp_ids = [f"cp_{i:03d}_{uuid.uuid4().hex[:8]}" for i in range(3)]
    for cp_id in cp_ids:
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        sync_saver.put(config, _checkpoint(cp_id), {}, {})

    list_config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoints = list(sync_saver.list(list_config))
    assert [c.checkpoint["id"] for c in checkpoints] == list(reversed(cp_ids))


def test_sync_put_writes_and_get_tuple(sync_saver: AzureTableStorageSaver) -> None:
    tid = f"sync_writes_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cpid}
    }
    sync_saver.put(config, _checkpoint(cpid), {}, {})
    sync_saver.put_writes(config, [("channel_a", "value_a")], "task-1")

    retrieved = sync_saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.pending_writes == [("task-1", "channel_a", "value_a")]


def test_sync_large_checkpoint_round_trips(sync_saver: AzureTableStorageSaver) -> None:
    tid = f"sync_large_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cpid}
    }
    big_value = "z" * (150 * 1024)  # forces multi-chunk storage

    sync_saver.put(config, _checkpoint(cpid, data=big_value), {}, {})
    retrieved = sync_saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["channel_values"]["data"] == big_value


async def test_async_put_and_get_tuple(async_saver: AzureTableStorageSaver) -> None:
    tid = f"async_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cpid}
    }

    result_config = await async_saver.aput(
        config, _checkpoint(cpid, test_key="async_value"), {"source": "input"}, {}
    )
    assert result_config["configurable"]["thread_id"] == tid

    retrieved = await async_saver.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == cpid
    assert retrieved.checkpoint["channel_values"] == {"test_key": "async_value"}


async def test_async_list_orders_most_recent_first(
    async_saver: AzureTableStorageSaver,
) -> None:
    tid = f"async_list_{uuid.uuid4().hex[:8]}"
    cp_ids = [f"cp_{i:03d}_{uuid.uuid4().hex[:8]}" for i in range(2)]
    for cp_id in cp_ids:
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        await async_saver.aput(config, _checkpoint(cp_id), {}, {})

    list_config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoints = [c async for c in async_saver.alist(list_config)]
    assert [c.checkpoint["id"] for c in checkpoints] == list(reversed(cp_ids))


async def test_async_put_writes_and_get_tuple(
    async_saver: AzureTableStorageSaver,
) -> None:
    tid = f"async_writes_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cpid}
    }
    await async_saver.aput(config, _checkpoint(cpid), {}, {})
    await async_saver.aput_writes(config, [("channel_a", "value_a")], "task-1")

    retrieved = await async_saver.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.pending_writes == [("task-1", "channel_a", "value_a")]


def test_sync_list_without_namespace_spans_namespaces_and_threads(
    sync_saver: AzureTableStorageSaver,
) -> None:
    tid = f"sync_ns_{uuid.uuid4().hex[:8]}"
    other = f"{tid}x"  # shares the thread-id prefix; must not leak into `tid`
    for thread, ns, cp_id in [
        (tid, "", "cp_001"),
        (tid, "sub|node:1", "cp_002"),
        (other, "", "cp_003"),
    ]:
        config = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
        sync_saver.put(config, _checkpoint(cp_id), {}, {})

    one_thread = list(sync_saver.list({"configurable": {"thread_id": tid}}))
    assert [c.checkpoint["id"] for c in one_thread] == ["cp_002", "cp_001"]
    assert [c.config["configurable"]["checkpoint_ns"] for c in one_thread] == [
        "sub|node:1",
        "",
    ]

    everything = list(sync_saver.list(None))
    assert [c.checkpoint["id"] for c in everything] == ["cp_003", "cp_002", "cp_001"]


def test_sync_list_checkpoint_id_and_non_ascii_ids(
    sync_saver: AzureTableStorageSaver,
) -> None:
    tid = f"sync_ids_{uuid.uuid4().hex[:8]}"
    cp_ids = ["cp_001", "~cp", "é-cp"]
    for cp_id in cp_ids:
        config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
        sync_saver.put(config, _checkpoint(cp_id), {}, {})
    write_config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": "é-cp"}
    }
    sync_saver.put_writes(write_config, [("channel_a", "value_a")], "~task")

    latest = sync_saver.get_tuple({"configurable": {"thread_id": tid}})
    assert latest is not None
    assert latest.checkpoint["id"] == "é-cp"
    assert latest.pending_writes == [("~task", "channel_a", "value_a")]

    listed = list(sync_saver.list({"configurable": {"thread_id": tid}}))
    assert [c.checkpoint["id"] for c in listed] == ["é-cp", "~cp", "cp_001"]

    only_one = list(
        sync_saver.list(
            {"configurable": {"thread_id": tid, "checkpoint_id": "~cp"}},
        )
    )
    assert [c.checkpoint["id"] for c in only_one] == ["~cp"]


def test_sync_put_persists_config_metadata(sync_saver: AzureTableStorageSaver) -> None:
    tid = f"sync_meta_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": ""},
        "metadata": {"user_id": "u1"},
    }
    sync_saver.put(config, _checkpoint("cp_001"), {"source": "input"}, {})

    listed = list(sync_saver.list(None, filter={"user_id": "u1"}))
    assert [c.checkpoint["id"] for c in listed] == ["cp_001"]
    assert listed[0].metadata == {"source": "input", "user_id": "u1"}


def test_sync_put_writes_mixed_batch_upserts_only_special_channel(
    sync_saver: AzureTableStorageSaver,
) -> None:
    tid = f"sync_mixed_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": "cp_1"}
    }
    sync_saver.put(config, _checkpoint("cp_1"), {}, {})
    sync_saver.put_writes(config, [("channel_a", "first"), ("__resume__", "old")], "t")
    sync_saver.put_writes(config, [("channel_a", "second"), ("__resume__", "new")], "t")

    retrieved = sync_saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.pending_writes == [
        ("t", "__resume__", "new"),
        ("t", "channel_a", "first"),
    ]


async def test_async_list_without_namespace_spans_namespaces_and_threads(
    async_saver: AzureTableStorageSaver,
) -> None:
    tid = f"async_ns_{uuid.uuid4().hex[:8]}"
    other = f"{tid}x"
    for thread, ns, cp_id in [
        (tid, "", "cp_001"),
        (tid, "sub|node:1", "cp_002"),
        (other, "", "cp_003"),
    ]:
        config = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
        await async_saver.aput(config, _checkpoint(cp_id), {}, {})
    await async_saver.aput_writes(
        {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "sub|node:1",
                "checkpoint_id": "cp_002",
            }
        },
        [("channel_a", "value_a")],
        "~task",
    )

    one_thread = [
        c async for c in async_saver.alist({"configurable": {"thread_id": tid}})
    ]
    assert [c.checkpoint["id"] for c in one_thread] == ["cp_002", "cp_001"]
    assert one_thread[0].pending_writes == [("~task", "channel_a", "value_a")]

    everything = [c async for c in async_saver.alist(None)]
    assert [c.checkpoint["id"] for c in everything] == ["cp_003", "cp_002", "cp_001"]
