# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Common checkpoint and process-crash helpers for E2E injection."""

from __future__ import annotations

import json
import os
from azure.ai.agentserver.responses.streaming._checkpoint import ResponseCheckpointEvent

from .crash_points import CRASH_EXIT_CODE
from .workflow import state_root, thread_key

CheckpointRef = tuple[str, str]


def crash_marker_exists(thread_id: str) -> bool:
    """Return whether this thread already triggered its configured crash."""

    return (state_root() / f"{thread_key(thread_id)}.crash.json").exists()


def event_checkpoint_ref(event: ResponseCheckpointEvent) -> CheckpointRef | None:
    """Read a LangGraph checkpoint reference from a Responses checkpoint."""

    metadata = event.response.get("metadata")
    internal = (
        metadata.get("_internal_metadata") if isinstance(metadata, dict) else None
    )
    if isinstance(internal, str):
        try:
            internal = json.loads(internal)
        except json.JSONDecodeError:
            return None
    if not isinstance(internal, dict):
        return None
    thread_id = internal.get("langgraph_thread_id")
    checkpoint_id = internal.get("langgraph_checkpoint_id")
    if isinstance(thread_id, str) and isinstance(checkpoint_id, str):
        return thread_id, checkpoint_id
    return None


def crash_once(
    checkpoint: tuple[str, str | None],
    crash_point: str,
) -> None:
    """Write one durable marker, then terminate the process abruptly."""

    thread_id, checkpoint_id = checkpoint
    marker_path = state_root() / f"{thread_key(thread_id)}.crash.json"
    try:
        with marker_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "crash_point": crash_point,
                    "checkpoint_id": checkpoint_id,
                },
                stream,
                sort_keys=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return

    print(f"E2E crash at {crash_point}: {checkpoint_id or 'no-checkpoint'}", flush=True)
    os._exit(CRASH_EXIT_CODE)