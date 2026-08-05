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
