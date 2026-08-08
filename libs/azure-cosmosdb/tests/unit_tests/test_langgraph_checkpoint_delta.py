"""Parity unit tests for ``CosmosDBSaverSync.get_delta_channel_history``.

Asserts the fast-path override returns output identical to the inherited
``BaseCheckpointSaver.get_delta_channel_history`` default across a range of
ancestor-chain depths, using an in-memory fake container.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from langchain_azure_cosmosdb import CosmosDBSaverSync
from tests.unit_tests._delta_common import (
    CHANNELS,
    DELTA_A,
    DELTA_B,
    DEPTHS,
    FakeSyncContainer,
    make_checkpoint,
    make_serde,
)

THREAD_ID = "thread-sync"
NS = ""


def _build_saver() -> CosmosDBSaverSync:
    saver = object.__new__(CosmosDBSaverSync)
    BaseCheckpointSaver.__init__(saver)
    saver.container = FakeSyncContainer()
    saver.cosmos_serde = make_serde()
    return saver


def _seed_chain(saver: CosmosDBSaverSync, depth: int) -> str:
    """Seed a parent chain of ``depth`` ancestors; return the head checkpoint id."""
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
        saver.put(config, make_checkpoint(cp_id, channel_values), {}, {})

        write_config: RunnableConfig = {
            "configurable": {
                "thread_id": THREAD_ID,
                "checkpoint_ns": NS,
                "checkpoint_id": cp_id,
            }
        }
        saver.put_writes(
            write_config,
            [(DELTA_A, f"a-{level}"), (DELTA_B, f"b-{level}")],
            f"task-{level}",
        )
        parent_id = cp_id
        head_id = cp_id
    return head_id


@pytest.mark.parametrize("depth", DEPTHS)
def test_override_matches_base_across_depths(depth: int) -> None:
    saver = _build_saver()
    head_id = _seed_chain(saver, depth)
    config: RunnableConfig = {
        "configurable": {
            "thread_id": THREAD_ID,
            "checkpoint_ns": NS,
            "checkpoint_id": head_id,
        }
    }

    base = BaseCheckpointSaver.get_delta_channel_history(
        saver, config=config, channels=CHANNELS
    )
    override = saver.get_delta_channel_history(config=config, channels=CHANNELS)

    assert override == base
    # Seeded channel carries a seed; unseeded channel omits it.
    assert "seed" in override[DELTA_A]
    assert "seed" not in override[DELTA_B]
    # Head checkpoint's own writes are excluded -> exactly ``depth`` deltas.
    assert len(override[DELTA_A]["writes"]) == depth
    assert len(override[DELTA_B]["writes"]) == depth
    # Writes are ordered oldest -> newest.
    values = [w[2] for w in override[DELTA_A]["writes"]]
    assert values == [f"a-{i}" for i in range(depth)]


def test_override_matches_base_latest_checkpoint() -> None:
    """Resolving the target via 'latest' (no checkpoint_id) also matches base."""
    saver = _build_saver()
    _seed_chain(saver, 10)
    config: RunnableConfig = {
        "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": NS}
    }

    base = BaseCheckpointSaver.get_delta_channel_history(
        saver, config=config, channels=CHANNELS
    )
    override = saver.get_delta_channel_history(config=config, channels=CHANNELS)
    assert override == base


def test_empty_channels_returns_empty() -> None:
    saver = _build_saver()
    _seed_chain(saver, 5)
    config: RunnableConfig = {
        "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": NS}
    }
    assert saver.get_delta_channel_history(config=config, channels=[]) == {}


def test_missing_thread_matches_base() -> None:
    saver = _build_saver()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "does-not-exist",
            "checkpoint_ns": NS,
            "checkpoint_id": "cp0000",
        }
    }
    base = BaseCheckpointSaver.get_delta_channel_history(
        saver, config=config, channels=CHANNELS
    )
    override = saver.get_delta_channel_history(config=config, channels=CHANNELS)
    assert override == base
    assert override == {DELTA_A: {"writes": []}, DELTA_B: {"writes": []}}
