# type: ignore
import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest

from langchain_azure_cosmosdb import CosmosDBSaver, CosmosDBSaverSync

pytestmark = pytest.mark.skipif(
    not os.getenv("COSMOSDB_ENDPOINT"),
    reason="COSMOSDB_ENDPOINT environment variable not set",
)


@pytest.fixture
def sync_saver() -> Iterator[CosmosDBSaverSync]:
    database_name = os.getenv("COSMOSDB_TEST_DATABASE", "test_langgraph")
    container_name = os.getenv("COSMOSDB_TEST_CONTAINER", "test_checkpoints")
    saver = CosmosDBSaverSync(
        database_name=database_name,
        container_name=container_name,
    )
    yield saver


@pytest.fixture
async def async_saver() -> AsyncIterator[CosmosDBSaver]:
    endpoint = os.getenv("COSMOSDB_ENDPOINT")
    key = os.getenv("COSMOSDB_KEY")
    database_name = os.getenv("COSMOSDB_TEST_DATABASE", "test_langgraph")
    container_name = os.getenv("COSMOSDB_TEST_CONTAINER", "test_checkpoints")
    async with CosmosDBSaver.from_conn_info(
        endpoint=endpoint,
        key=key,
        database_name=database_name,
        container_name=container_name,
    ) as saver:
        yield saver


def test_sync_init(sync_saver: CosmosDBSaverSync) -> None:
    assert sync_saver is not None
    assert sync_saver.container is not None


def test_sync_put_and_get(sync_saver: CosmosDBSaverSync) -> None:
    tid = f"sync_pg_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {
            "thread_id": tid,
            "checkpoint_ns": "",
            "checkpoint_id": cpid,
        }
    }
    checkpoint = {
        "v": 1,
        "id": cpid,
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": {"test_key": "test_value"},
        "channel_versions": {"test_key": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    result_config = sync_saver.put(config, checkpoint, {"source": "test"}, {})
    assert result_config["configurable"]["thread_id"] == tid

    retrieved = sync_saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == cpid
    assert retrieved.metadata["source"] == "test"


def test_sync_list(sync_saver: CosmosDBSaverSync) -> None:
    tid = f"sync_list_{uuid.uuid4().hex[:8]}"
    for i in range(3):
        cpid = f"cp_list_{tid}_{i}"
        config = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "",
                "checkpoint_id": cpid,
            }
        }
        checkpoint = {
            "v": 1,
            "id": cpid,
            "ts": f"2024-01-01T00:00:0{i}.000000+00:00",
            "channel_values": {"step": i},
            "channel_versions": {"step": i},
            "versions_seen": {},
            "pending_sends": [],
        }
        sync_saver.put(config, checkpoint, {"step": i}, {})

    list_config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoints = list(sync_saver.list(list_config))
    assert len(checkpoints) >= 3


async def test_async_put_and_get(async_saver: CosmosDBSaver) -> None:
    tid = f"async_pg_{uuid.uuid4().hex[:8]}"
    cpid = f"cp_{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {
            "thread_id": tid,
            "checkpoint_ns": "",
            "checkpoint_id": cpid,
        }
    }
    checkpoint = {
        "v": 1,
        "id": cpid,
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": {"async_key": "async_value"},
        "channel_versions": {"async_key": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    result_config = await async_saver.aput(config, checkpoint, {"source": "async"}, {})
    assert result_config["configurable"]["thread_id"] == tid

    retrieved = await async_saver.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == cpid


async def test_async_list(async_saver: CosmosDBSaver) -> None:
    tid = f"async_list_{uuid.uuid4().hex[:8]}"
    for i in range(2):
        cpid = f"cp_alist_{tid}_{i}"
        config = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "",
                "checkpoint_id": cpid,
            }
        }
        checkpoint = {
            "v": 1,
            "id": cpid,
            "ts": f"2024-01-01T00:00:0{i}.000000+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        await async_saver.aput(config, checkpoint, {}, {})

    list_config = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    checkpoints = []
    async for checkpoint_tuple in async_saver.alist(list_config):
        checkpoints.append(checkpoint_tuple)
    assert len(checkpoints) >= 2


def _seed_delta_chain_sync(
    saver: CosmosDBSaverSync, tid: str, depth: int, delta_ch: str
) -> str:
    parent_id = None
    head_id = ""
    for level in range(depth + 1):
        cpid = f"cp_delta_{tid}_{level:04d}"
        channel_values = {delta_ch: f"seed-{level}"} if level == 0 else {}
        config = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "",
                **({"checkpoint_id": parent_id} if parent_id else {}),
            }
        }
        checkpoint = {
            "v": 1,
            "id": cpid,
            "ts": "2024-01-01T00:00:00.000000+00:00",
            "channel_values": channel_values,
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        saver.put(config, checkpoint, {}, {})
        write_config = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "",
                "checkpoint_id": cpid,
            }
        }
        saver.put_writes(write_config, [(delta_ch, f"delta-{level}")], f"task-{level}")
        parent_id = cpid
        head_id = cpid
    return head_id


@pytest.mark.parametrize("depth", [1, 10, 30])
def test_sync_delta_history_matches_base(
    sync_saver: CosmosDBSaverSync, depth: int
) -> None:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    tid = f"sync_delta_{uuid.uuid4().hex[:8]}"
    delta_ch = "my_delta"
    head_id = _seed_delta_chain_sync(sync_saver, tid, depth, delta_ch)
    config = {
        "configurable": {
            "thread_id": tid,
            "checkpoint_ns": "",
            "checkpoint_id": head_id,
        }
    }
    base = BaseCheckpointSaver.get_delta_channel_history(
        sync_saver, config=config, channels=[delta_ch]
    )
    override = sync_saver.get_delta_channel_history(config=config, channels=[delta_ch])
    assert override == base
    assert len(override[delta_ch]["writes"]) == depth


async def _seed_delta_chain_async(
    saver: CosmosDBSaver, tid: str, depth: int, delta_ch: str
) -> str:
    parent_id = None
    head_id = ""
    for level in range(depth + 1):
        cpid = f"cp_delta_{tid}_{level:04d}"
        channel_values = {delta_ch: f"seed-{level}"} if level == 0 else {}
        config = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "",
                **({"checkpoint_id": parent_id} if parent_id else {}),
            }
        }
        checkpoint = {
            "v": 1,
            "id": cpid,
            "ts": "2024-01-01T00:00:00.000000+00:00",
            "channel_values": channel_values,
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        await saver.aput(config, checkpoint, {}, {})
        write_config = {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": "",
                "checkpoint_id": cpid,
            }
        }
        await saver.aput_writes(
            write_config, [(delta_ch, f"delta-{level}")], f"task-{level}"
        )
        parent_id = cpid
        head_id = cpid
    return head_id


@pytest.mark.parametrize("depth", [1, 10, 30])
async def test_async_delta_history_matches_base(
    async_saver: CosmosDBSaver, depth: int
) -> None:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    tid = f"async_delta_{uuid.uuid4().hex[:8]}"
    delta_ch = "my_delta"
    head_id = await _seed_delta_chain_async(async_saver, tid, depth, delta_ch)
    config = {
        "configurable": {
            "thread_id": tid,
            "checkpoint_ns": "",
            "checkpoint_id": head_id,
        }
    }
    base = await BaseCheckpointSaver.aget_delta_channel_history(
        async_saver, config=config, channels=[delta_ch]
    )
    override = await async_saver.aget_delta_channel_history(
        config=config, channels=[delta_ch]
    )
    assert override == base
    assert len(override[delta_ch]["writes"]) == depth
