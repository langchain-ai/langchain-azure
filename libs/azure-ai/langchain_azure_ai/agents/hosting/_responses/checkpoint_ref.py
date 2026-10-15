# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""LangGraph checkpoint references and active interrupt indexes."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_chain_store import ConversationChainStoreProtocol

CONVERSATION_CHECKPOINT_KEY = "langgraph_checkpoint"
_CHECKPOINT_ID = "checkpoint_id"
_THREAD_ID = "thread_id"
_PENDING_INTERRUPTS_PREFIX = "langgraph_pending_interrupts:"
_MIN_RECORD_CAPACITY = 512


@dataclass(frozen=True)
class CheckpointRef:
    """A LangGraph thread and checkpoint reference."""

    thread_id: str
    checkpoint_id: str

    @classmethod
    def from_dict(cls, data: dict[str, str] | None) -> "CheckpointRef | None":
        """Parse a checkpoint reference from a dictionary."""
        if data is None:
            return None
        thread_id = data.get(_THREAD_ID)
        if not isinstance(thread_id, str) or not thread_id:
            return None
        checkpoint_id = data.get(_CHECKPOINT_ID)
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return cls(thread_id, checkpoint_id)

    def to_dict(self) -> dict[str, str]:
        """Return the checkpoint reference as a dictionary."""
        return {
            _THREAD_ID: self.thread_id,
            _CHECKPOINT_ID: self.checkpoint_id,
        }


@dataclass(frozen=True)
class PendingInterruptIndex:
    """One published pause generation, or a consumed generation."""

    generation: str
    interrupt_ids: frozenset[str] | None


async def store_pending_interrupt_ids(
    store: ConversationChainStoreProtocol,
    ref: CheckpointRef,
    interrupt_ids: Iterable[str],
) -> None:
    """Publish an active interrupt set without exceeding minimum store capacity.

    Args:
        store: The existing hosting string-dictionary store.
        ref: The checkpoint whose interrupt history is being indexed.
        interrupt_ids: IDs actually emitted by the graph's updates stream.

    Raises:
        ValueError: An ID is empty or exceeds minimum dictionary capacity.
    """
    chunks: list[dict[str, str]] = []
    chunk: dict[str, str] = {}
    size = 0
    for interrupt_id in dict.fromkeys(interrupt_ids):
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise ValueError("Pending interrupt IDs must be nonempty strings.")
        item_size = len(interrupt_id) + 1
        if item_size > _MIN_RECORD_CAPACITY:
            raise ValueError("Pending interrupt ID exceeds minimum store capacity.")
        if size + item_size > _MIN_RECORD_CAPACITY:
            chunks.append(chunk)
            chunk = {}
            size = 0
        chunk[interrupt_id] = "1"
        size += item_size
    if chunk:
        chunks.append(chunk)

    key = f"{_PENDING_INTERRUPTS_PREFIX}{ref.checkpoint_id}"
    generation = uuid.uuid4().hex
    for index, data in enumerate(chunks):
        await store.set(ref.thread_id, f"{key}:{generation}:{index}", data)
    await store.set(
        ref.thread_id,
        key,
        {
            "generation": generation,
            "chunks": str(len(chunks)),
            "version": "2",
            "state": "ready",
        },
    )


async def consume_pending_interrupt_ids(
    store: ConversationChainStoreProtocol,
    ref: CheckpointRef,
) -> None:
    """Invalidate a pause before starting any graph execution.

    Args:
        store: The existing hosting string-dictionary store.
        ref: The exact checkpoint being inspected.
    """
    await store.set(
        ref.thread_id,
        f"{_PENDING_INTERRUPTS_PREFIX}{ref.checkpoint_id}",
        {
            "generation": uuid.uuid4().hex,
            "chunks": "0",
            "version": "2",
            "state": "consumed",
        },
    )


async def load_pending_interrupt_index(
    store: ConversationChainStoreProtocol,
    ref: CheckpointRef,
) -> PendingInterruptIndex | None:
    """Load a pause generation, or return ``None`` for unindexed state.

    Args:
        store: The existing hosting string-dictionary store.
        ref: The exact checkpoint being inspected.

    Returns:
        The ready or consumed generation, or ``None`` for an absent or legacy
        index. A consumed generation has no trustworthy active interrupt set.

    Raises:
        ValueError: The index is malformed or a referenced chunk is missing.
    """
    key = f"{_PENDING_INTERRUPTS_PREFIX}{ref.checkpoint_id}"
    marker = await store.get(ref.thread_id, key)
    if marker is None:
        return None
    generation = marker.get("generation")
    count = marker.get("chunks")
    if not generation or count is None or not count.isdecimal():
        raise ValueError("Invalid pending interrupt index.")

    active: set[str] = set()
    for index in range(int(count)):
        chunk = await store.get(ref.thread_id, f"{key}:{generation}:{index}")
        if (
            not chunk
            or any(
                not interrupt_id or value != "1"
                for interrupt_id, value in chunk.items()
            )
            or active.intersection(chunk)
        ):
            raise ValueError("Incomplete or invalid pending interrupt index.")
        active.update(chunk)
    if marker.get("version") != "2":
        return None
    state = marker.get("state")
    if state == "consumed" and not active and count == "0":
        return PendingInterruptIndex(generation, None)
    if state != "ready":
        raise ValueError("Invalid pending interrupt generation state.")
    return PendingInterruptIndex(generation, frozenset(active))
