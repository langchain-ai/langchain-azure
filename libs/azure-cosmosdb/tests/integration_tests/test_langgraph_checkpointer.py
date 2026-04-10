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
