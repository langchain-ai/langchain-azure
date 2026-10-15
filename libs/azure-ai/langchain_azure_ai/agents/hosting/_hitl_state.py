# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pause generations shared by the Responses and Invocations hosts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Interrupt

from ._converters import pending_interrupts_from_snapshot
from ._responses import (
    CheckpointRef,
    ConversationChainStoreProtocol,
    HostingRunnableConfig,
    consume_pending_interrupt_ids,
    load_pending_interrupt_index,
    store_pending_interrupt_ids,
)

LEGACY_PARTIAL_RESUME_ERROR = (
    "This legacy or interrupted checkpoint has no reliable active interrupt "
    "index. Start a new conversation and resubmit the required original input "
    "without reusing the old conversation, agent_session_id, previous_response_id, "
    "previous_invocation_id, call IDs, or approval IDs."
)


@dataclass(frozen=True)
class PendingInterruptState:
    """Current obligations and their published pause generation."""

    pending: tuple[Interrupt, ...] = ()
    checkpoint_ref: CheckpointRef | None = None
    generation: str | None = None


async def read_pending_interrupt_state(
    graph: Runnable[Any, Any],
    config: RunnableConfig,
    store: ConversationChainStoreProtocol,
) -> PendingInterruptState | None:
    """Read active pauses without interpreting or hashing resume values.

    Args:
        graph: The hosted runnable.
        config: The exact execution configuration.
        store: The hosting pause-record store.

    Returns:
        The current pause, or ``None`` when a new conversation is required.
    """
    get_state = getattr(graph, "aget_state", None)
    if get_state is None:
        return PendingInterruptState()
    snapshot = await get_state(config)
    if snapshot is None:
        return PendingInterruptState()
    ref = HostingRunnableConfig(snapshot.config).checkpoint_ref
    if not any(task.interrupts for task in snapshot.tasks):
        return PendingInterruptState(checkpoint_ref=ref)
    if ref is None:
        raise ValueError("Pending interrupts have no checkpoint reference.")
    index = await load_pending_interrupt_index(store, ref)
    if index is not None:
        if index.interrupt_ids is None:
            return None
        pending = pending_interrupts_from_snapshot(
            snapshot, active_ids=index.interrupt_ids
        )
        if {interrupt.id for interrupt in pending} != index.interrupt_ids:
            raise ValueError("The active interrupt index does not match the graph.")
        return PendingInterruptState(pending, ref, index.generation)

    checkpointer = getattr(graph, "checkpointer", None)
    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise RuntimeError("Pending interrupts have no checkpoint saver.")
    checkpoint = await checkpointer.aget_tuple(snapshot.config)
    if checkpoint is None:
        raise RuntimeError("The pending interrupt checkpoint is missing.")
    if any(
        channel == "__resume__" for _, channel, _ in checkpoint.pending_writes or ()
    ):
        return None
    return PendingInterruptState(pending_interrupts_from_snapshot(snapshot), ref)


async def begin_interrupt_run(
    graph: Runnable[Any, Any],
    config: RunnableConfig,
    store: ConversationChainStoreProtocol,
) -> None:
    """Consume the old pause before the graph can write to its checkpoint.

    Args:
        graph: The hosted runnable.
        config: The exact execution configuration.
        store: The hosting pause-record store.
    """
    get_state = getattr(graph, "aget_state", None)
    if get_state is None:
        return
    snapshot = await get_state(config)
    if snapshot is not None:
        ref = HostingRunnableConfig(snapshot.config).checkpoint_ref
        if ref is not None:
            await consume_pending_interrupt_ids(store, ref)


async def publish_pending_interrupts(
    graph: Runnable[Any, Any],
    config: RunnableConfig,
    store: ConversationChainStoreProtocol,
    pending: Sequence[Interrupt],
    *,
    checkpoint_ref: CheckpointRef | None = None,
) -> None:
    """Publish exact stream-observed pauses only after successful execution.

    Args:
        graph: The hosted runnable.
        config: The execution configuration.
        store: The hosting pause-record store.
        pending: Interrupts emitted by the completed graph stream.
        checkpoint_ref: The last checkpoint emitted by the graph, when known.
    """
    if checkpoint_ref is None:
        get_state = getattr(graph, "aget_state", None)
        if get_state is None:
            if pending:
                raise RuntimeError("Pending interrupts have no checkpoint reader.")
            return
        latest_config: RunnableConfig = {
            **config,
            "configurable": dict(config.get("configurable", {})),
        }
        latest_config["configurable"].pop("checkpoint_id", None)
        snapshot = await get_state(latest_config)
        if snapshot is not None:
            checkpoint_ref = HostingRunnableConfig(snapshot.config).checkpoint_ref
    if checkpoint_ref is None:
        if pending:
            raise RuntimeError("Pending interrupts have no checkpoint reference.")
        return
    await store_pending_interrupt_ids(
        store, checkpoint_ref, (interrupt.id for interrupt in pending)
    )
