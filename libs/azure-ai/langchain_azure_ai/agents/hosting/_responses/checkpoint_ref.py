# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""LangGraph checkpoint references for Responses API hosting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointRef:
    """A LangGraph thread and checkpoint reference."""

    thread_id: str
    checkpoint_id: str
