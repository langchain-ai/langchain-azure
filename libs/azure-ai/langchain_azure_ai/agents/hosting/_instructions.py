# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Temporary instructions at Responses request and completion boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, create_checkpoint
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

_INSTRUCTIONS_ID_PREFIX = "__langchain_azure_response_instructions:"


def _is_request_instruction(value: Any) -> bool:
    return (
        isinstance(value, BaseMessage)
        and isinstance(value.id, str)
        and value.id.startswith(_INSTRUCTIONS_ID_PREFIX)
    )


@dataclass
class _ResponseInstructions:
    """Keep instructions scoped to one Responses request."""

    message: SystemMessage | None

    @classmethod
    def create(cls, instructions: object, response_id: str) -> _ResponseInstructions:
        if instructions is None:
            return cls(None)
        if not isinstance(instructions, str):
            raise ValueError("Response instructions must be a string or null.")
        return cls(
            SystemMessage(
                content=instructions,
                id=f"{_INSTRUCTIONS_ID_PREFIX}{response_id}",
                additional_kwargs={"__openai_role__": "developer"},
            )
            if instructions
            else None
        )

    def prepend(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        durable = [
            message for message in messages if not _is_request_instruction(message)
        ]
        return (
            [self.message.model_copy(deep=True), *durable] if self.message else durable
        )

    async def prepare(
        self,
        graph: CompiledStateGraph,
        config: RunnableConfig,
        graph_input: dict[str, Any] | Command | None,
    ) -> dict[str, Any] | Command | None:
        if (
            graph_input is None
            or not isinstance(graph, CompiledStateGraph)
            or not isinstance(graph.checkpointer, BaseCheckpointSaver)
        ):
            return graph_input
        if self.message is None and not (config.get("configurable") or {}).get(
            "thread_id"
        ):
            return graph_input
        state = await graph.aget_state(config)
        removals = [
            RemoveMessage(id=message.id)
            for message in state.values.get("messages", [])
            if _is_request_instruction(message)
        ]
        if self.message is None and not removals:
            return graph_input
        if isinstance(graph_input, Command):
            return replace(
                graph_input,
                update=[
                    ("messages", [*removals, *self.prepend([])]),
                    *graph_input._update_as_tuples(),
                ],
            )
        return {
            **graph_input,
            "messages": [*removals, *self.prepend(graph_input.get("messages", []))],
        }

    async def finish(
        self, graph: CompiledStateGraph, config: RunnableConfig
    ) -> RunnableConfig:
        if self.message is None:
            return config
        if not isinstance(graph, CompiledStateGraph) or not isinstance(
            graph.checkpointer, BaseCheckpointSaver
        ):
            return config
        state = await graph.aget_state(config)
        if state.next or state.tasks or state.interrupts:
            return config
        messages = state.values.get("messages", [])
        removals: list[MessageLikeRepresentation] = [
            RemoveMessage(id=message.id)
            for message in messages
            if _is_request_instruction(message)
        ]
        if not removals:
            return config
        saver = graph.checkpointer
        saved = await saver.aget_tuple(state.config)
        if saved is None:
            raise ValueError("The completed response's graph checkpoint is missing.")
        step = saved.metadata.get("step", -1) + 1
        checkpoint = create_checkpoint(saved.checkpoint, None, step)
        checkpoint["channel_values"] = {
            **saved.checkpoint["channel_values"],
            "messages": add_messages(messages, removals),
        }
        version = saver.get_next_version(
            saved.checkpoint["channel_versions"].get("messages"), None
        )
        checkpoint["channel_versions"] = {
            **saved.checkpoint["channel_versions"],
            "messages": version,
        }
        # Save a value-only child checkpoint without invoking node/routing writers.
        return await saver.aput(
            state.config,
            checkpoint,
            {**saved.metadata, "source": "update", "step": step},
            {"messages": version},
        )
