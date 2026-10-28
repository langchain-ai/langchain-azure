# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Response-owned checkpoint adaptation shared by hosting paths."""

from __future__ import annotations

import asyncio
import json
from copy import copy
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver

from langchain_azure_ai.agents.hosting._responses import CheckpointRef
from langchain_azure_ai.agents.hosting._responses.task_storage_manager import (
    METADATA_RESPONSE_EXECUTION,
    ResponseCheckpointSaver,
)


def _config(checkpoint_id: str | None = None, namespace: str = "") -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": "thread",
            "checkpoint_ns": namespace,
            **({"checkpoint_id": checkpoint_id} if checkpoint_id else {}),
        }
    }


def _saver(
    backend: Any,
    *,
    recovery_ref: CheckpointRef | None = None,
    require_ownership: bool = False,
) -> ResponseCheckpointSaver:
    return ResponseCheckpointSaver(
        backend,
        thread_id="thread",
        response_key="response",
        parent_response_id="parent",
        recovery_ref=recovery_ref,
        require_ownership=require_ownership,
    )


async def test_progress_waits_for_a_successful_graph_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemorySaver()
    entered, release = asyncio.Event(), asyncio.Event()
    save = backend.aput

    async def delayed(*args: Any, **kwargs: Any) -> RunnableConfig:
        entered.set()
        await release.wait()
        return await save(*args, **kwargs)

    monkeypatch.setattr(backend, "aput", delayed)
    saver = _saver(backend)
    checkpoint = empty_checkpoint()
    reference = CheckpointRef("thread", checkpoint["id"])
    waiting = asyncio.create_task(saver.wait_for_checkpoint(reference))
    writing = asyncio.create_task(saver.aput(_config(), checkpoint, {"step": 0}, {}))
    await entered.wait()
    assert not waiting.done()
    release.set()
    saved_config = await writing
    assert await waiting
    stored = await backend.aget_tuple(saved_config)
    assert stored is not None
    stamp = stored.metadata.get(METADATA_RESPONSE_EXECUTION)
    assert isinstance(stamp, str)
    assert json.loads(stamp) == [
        "response",
        "parent",
    ]


@pytest.mark.parametrize("operation", ["checkpoint", "writes"])
async def test_failed_writes_unblock_progress_without_publishing(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    backend = InMemorySaver()
    saver = _saver(backend)
    checkpoint = empty_checkpoint()
    reference = CheckpointRef("thread", checkpoint["id"])
    failure = RuntimeError("storage unavailable")
    method = "aput" if operation == "checkpoint" else "aput_writes"
    monkeypatch.setattr(backend, method, AsyncMock(side_effect=failure))
    waiting = asyncio.create_task(saver.wait_for_checkpoint(reference))
    with pytest.raises(RuntimeError, match="storage unavailable"):
        if operation == "checkpoint":
            await saver.aput(_config(), checkpoint, {"step": 0}, {})
        else:
            await saver.aput_writes(_config(checkpoint["id"]), [], "task")
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await asyncio.wait_for(waiting, 1)


@pytest.mark.parametrize("field", ["missing", "thread", "namespace", "config_id", "id"])
async def test_exact_reads_reject_missing_or_mismatched_checkpoints(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    backend = InMemorySaver()
    checkpoint = empty_checkpoint()
    config = await backend.aput(_config(), checkpoint, {"step": 0}, {})
    stored = await backend.aget_tuple(config)
    assert stored is not None
    if field == "missing":
        stored = None
    elif field == "id":
        stored.checkpoint["id"] = "different"
    else:
        key = {
            "thread": "thread_id",
            "namespace": "checkpoint_ns",
            "config_id": "checkpoint_id",
        }[field]
        stored.config["configurable"][key] = "different"
    monkeypatch.setattr(backend, "aget_tuple", AsyncMock(return_value=stored))
    with pytest.raises(RuntimeError, match="checkpoint"):
        await _saver(backend).aget_tuple(_config(checkpoint["id"]))


async def test_native_recovery_keeps_its_head_and_pending_writes() -> None:
    backend = InMemorySaver()
    original = _saver(backend)
    checkpoint = empty_checkpoint()
    config = await original.aput(_config(), checkpoint, {"step": 0}, {})
    await original.aput_writes(config, [("messages", ["finished"])], "task", "node")
    sibling = empty_checkpoint()
    await backend.aput(config, sibling, {"step": 1}, {})
    reference = CheckpointRef("thread", checkpoint["id"])
    saver = _saver(backend, recovery_ref=reference, require_ownership=True)
    resume = saver.resume_config(config)
    assert "checkpoint_id" not in resume["configurable"]
    restored = await saver.aget_tuple(resume)
    assert restored is not None
    assert restored.checkpoint["id"] == checkpoint["id"]
    assert restored.pending_writes == [("task", "messages", ["finished"])]
    assert await saver.wait_for_checkpoint(reference)

    next_checkpoint = empty_checkpoint()
    await saver.aput(config, next_checkpoint, {"step": 1}, {})
    advanced = await saver.aget_tuple(resume)
    assert advanced is not None
    assert advanced.checkpoint["id"] == next_checkpoint["id"]
    assert (await backend.aget_tuple(_config(sibling["id"]))) is not None


@pytest.mark.parametrize("stamp", [None, '["other","parent"]'])
@pytest.mark.parametrize("require_ownership", [False, True])
async def test_recovery_checks_ownership_with_legacy_compatibility(
    stamp: str | None, require_ownership: bool
) -> None:
    backend = InMemorySaver()
    checkpoint = empty_checkpoint()
    metadata: dict[str, Any] = {"step": 0}
    if stamp is not None:
        metadata[METADATA_RESPONSE_EXECUTION] = stamp
    config = await backend.aput(
        _config(), checkpoint, cast(CheckpointMetadata, metadata), {}
    )
    saver = _saver(
        backend,
        recovery_ref=CheckpointRef("thread", checkpoint["id"]),
        require_ownership=require_ownership,
    )
    if stamp is None and not require_ownership:
        assert await saver.aget_tuple(saver.resume_config(config)) is not None
    else:
        with pytest.raises(RuntimeError, match="ownership"):
            await saver.aget_tuple(saver.resume_config(config))


async def test_inherited_checkpoint_is_not_current_response_progress() -> None:
    backend = InMemorySaver()
    checkpoint = empty_checkpoint()
    config = await backend.aput(_config(), checkpoint, {"step": 0}, {})
    saver = _saver(backend)
    assert await saver.aget_tuple(config) is not None
    assert not await saver.wait_for_checkpoint(
        CheckpointRef("thread", checkpoint["id"])
    )
    stored = await backend.aget_tuple(config)
    assert stored is not None
    assert METADATA_RESPONSE_EXECUTION not in stored.metadata


async def test_saver_clones_share_confirmation_and_preserve_private_subgraphs() -> None:
    backend = InMemorySaver()
    saver = _saver(backend)
    clone = copy(saver).with_allowlist({(__name__, "Payload")})
    checkpoint = empty_checkpoint()
    await clone.aput(_config(), checkpoint, {"step": 0}, {})
    assert await saver.wait_for_checkpoint(CheckpointRef("thread", checkpoint["id"]))
    private = empty_checkpoint()
    private_config = clone.put(
        _config(namespace="private:node"), private, {"step": 0}, {}
    )
    stored: CheckpointTuple | None = backend.get_tuple(private_config)
    assert stored is not None
    assert METADATA_RESPONSE_EXECUTION not in stored.metadata
