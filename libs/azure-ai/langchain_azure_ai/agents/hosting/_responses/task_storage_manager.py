# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Per-response LangGraph checkpoint storage management."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from azure.ai.agentserver.responses import ResponseEventStream, ResponseObject

from .checkpoint_ref import CheckpointRef

METADATA_LANGGRAPH_CHECKPOINT_ID = "langgraph_checkpoint_id"
METADATA_LANGGRAPH_THREAD_ID = "langgraph_thread_id"


class TaskStorageManager:
    """Manage LangGraph references persisted with one Responses task."""

    def __init__(self, metadata: MutableMapping[str, Any] | None) -> None:
        self._metadata = metadata

    @classmethod
    def from_stream(cls, stream: ResponseEventStream) -> TaskStorageManager:
        """Manage the writable metadata attached to an active response stream."""
        return cls(stream.internal_metadata)

    @classmethod
    def from_response(cls, response: ResponseObject | None) -> TaskStorageManager:
        """Manage metadata read from a persisted response."""
        return cls(getattr(response, "internal_metadata", None))

    @property
    def checkpoint_ref(self) -> CheckpointRef | None:
        """Return the reference persisted with this Responses task."""
        if self._metadata is None:
            return None
        thread_id = self._metadata.get(METADATA_LANGGRAPH_THREAD_ID)
        if not isinstance(thread_id, str) or not thread_id:
            return None
        checkpoint_id = self._metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID)
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return CheckpointRef(thread_id, checkpoint_id)

    def store_checkpoint_ref(self, checkpoint_ref: CheckpointRef) -> None:
        """Store a LangGraph reference on the active response."""
        if self._metadata is None:
            return
        self._metadata[METADATA_LANGGRAPH_THREAD_ID] = checkpoint_ref.thread_id
        self._metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = checkpoint_ref.checkpoint_id
