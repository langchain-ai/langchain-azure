# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""LangGraph checkpoint references for Responses API hosting."""

from dataclasses import dataclass

CONVERSATION_CHECKPOINT_KEY = "langgraph_checkpoint"
_CHECKPOINT_ID = "checkpoint_id"
_THREAD_ID = "thread_id"


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
