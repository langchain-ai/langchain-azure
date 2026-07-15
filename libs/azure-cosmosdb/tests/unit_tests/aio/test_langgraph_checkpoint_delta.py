"""Parity unit tests for ``CosmosDBSaver.aget_delta_channel_history``.

Asserts the fast-path override returns output identical to the inherited
``BaseCheckpointSaver.aget_delta_channel_history`` default across a range of
ancestor-chain depths, using an in-memory async fake container.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from langchain_azure_cosmosdb import CosmosDBSaver
from tests.unit_tests._delta_common import (
    CHANNELS,
    DELTA_A,
    DELTA_B,
    DEPTHS,
    FakeAsyncContainer,
    make_checkpoint,
    make_serde,
)

THREAD_ID = "thread-async"
NS = ""


def _build_saver() -> CosmosDBSaver:
    saver = object.__new__(CosmosDBSaver)
    BaseCheckpointSaver.__init__(saver)
    saver.container = FakeAsyncContainer()
    saver.cosmos_serde = make_serde()
    saver._loop = None
    return saver


async def _seed_chain(saver: CosmosDBSaver, depth: int) -> str:
    parent_id: str | None = None
    head_id = ""
    for level in range(depth + 1):
        cp_id = f"cp{level:04d}"
        channel_values = {DELTA_A: f"seed-a-{level}"} if level == 0 else {}
        config: RunnableConfig = {
            "configurable": {
                "thread_id": THREAD_ID,
                "checkpoint_ns": NS,
                **({"checkpoint_id": parent_id} if parent_id else {}),
            }
        }
        await saver.aput(config, make_checkpoint(cp_id, channel_values), {}, {})

        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": THREAD_ID,
                "checkpoint_ns": NS,
                "checkpoint_id": cp_id,
            }
        }
        await saver.aput_writes(
            write_config,
            [(DELTA_A, f"a-{level}"), (DELTA_B, f"b-{level}")],
            f"task-{level}",
        )
        parent_id = cp_id
        head_id = cp_id
    return head_id


@pytest.mark.parametrize("depth", DEPTHS)
async def test_override_matches_base_across_depths(depth: int) -> None:
    saver = _build_saver()
    head_id = await _seed_chain(saver, depth)
    config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": NS,
            "checkpoint_id": head_id,
        }
    }

    base = await BaseCheckpointSaver.aget_delta_channel_history(
        saver, config=config, channels=CHANNELS
    )
    override = await saver.aget_delta_channel_history(config=config, channels=CHANNELS)

    assert override == base
    assert "seed" in override[DELTA_A]
    assert "seed" not in override[DELTA_B]
    assert len(override[DELTA_A]["writes"]) == depth
    assert len(override[DELTA_B]["writes"]) == depth
    values = [w[2] for w in override[DELTA_A]["writes"]]
    assert values == [f"a-{i}" for i in range(depth)]


async def test_override_matches_base_latest_checkpoint() -> None:
    saver = _build_saver()
    await _seed_chain(saver, 10)
    config: RunnableConfig = {
        "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": NS}
    }

    base = await BaseCheckpointSaver.aget_delta_channel_history(
        saver, config=config, channels=CHANNELS
    )
    override = await saver.aget_delta_channel_history(config=config, channels=CHANNELS)
    assert override == base


async def test_empty_channels_returns_empty() -> None:
    saver = _build_saver()
    await _seed_chain(saver, 5)
    config: RunnableConfig = {
        "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": NS}
    }
    assert await saver.aget_delta_channel_history(config=config, channels=[]) == {}


async def test_missing_thread_matches_base() -> None:
    saver = _build_saver()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "does-not-exist",
            "checkpoint_ns": NS,
            "checkpoint_id": "cp0000",
        }
    }
    base = await BaseCheckpointSaver.aget_delta_channel_history(
        saver, config=config, channels=CHANNELS
    )
    override = await saver.aget_delta_channel_history(config=config, channels=CHANNELS)
    assert override == base
    assert override == {DELTA_A: {"writes": []}, DELTA_B: {"writes": []}}
