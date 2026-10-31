# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Temporary instructions at Responses request and completion boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.channels import BinaryOperatorAggregate, DeltaChannel, LastValue
from langgraph.checkpoint.base import BaseCheckpointSaver, create_checkpoint
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Overwrite

_INSTRUCTIONS_ID_PREFIX = "__langchain_azure_response_instructions:"


def _is_request_instruction(value: Any) -> bool:
    if isinstance(value, BaseMessage):
        message_id = value.id
    elif isinstance(value, dict):
        payload = (
            value.get("kwargs")
            if value.get("type") == "constructor"
            else value.get("data", value)
        )
        if not isinstance(payload, dict) or "content" not in payload:
            return False
        message_id = payload.get("id")
    else:
        return False
    return isinstance(message_id, str) and message_id.startswith(
        _INSTRUCTIONS_ID_PREFIX
    )


@dataclass
class _ResponseInstructions:
    """Keep instructions scoped to one Responses request."""

    message: SystemMessage | None
    _cleanup_required: bool = False

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
        if not isinstance(graph, CompiledStateGraph) or not isinstance(
            graph.checkpointer, BaseCheckpointSaver
        ):
            return graph_input
        channel = graph.channels.get("messages")
        if channel is None:
            return graph_input
        if self.message is None and not (config.get("configurable") or {}).get(
            "thread_id"
        ):
            return graph_input
        state = await graph.aget_state(config)
        previous = state.values.get("messages", [])
        self._cleanup_required = isinstance(previous, list) and any(
            _is_request_instruction(message) for message in previous
        )
        if self.message is None and not self._cleanup_required:
            return graph_input
        if not isinstance(previous, list):
            raise ValueError("Request instructions require a messages list.")
        if not isinstance(channel, (BinaryOperatorAggregate, DeltaChannel, LastValue)):
            raise ValueError(
                "Instructions require a last-value or reducer-backed messages channel."
            )
        if isinstance(graph_input, Command):
            # Keep LangGraph's repeated-write and unset-field semantics.
            writes = list(graph_input._update_as_tuples())
        elif isinstance(graph_input, dict):
            writes = list(graph_input.items())
        else:
            return graph_input
        updates = dict(writes)

        messages = self.prepend(previous)
        incoming_updates = []
        for key, incoming in writes:
            if key != "messages":
                continue
            if isinstance(incoming, Overwrite):
                incoming = Overwrite(self.prepend(incoming.value))
            elif isinstance(incoming, list):
                incoming = [
                    message
                    for message in incoming
                    if not _is_request_instruction(message)
                ]
            elif _is_request_instruction(incoming):
                incoming = []
            incoming_updates.append(incoming)
        if incoming_updates:
            incoming = incoming_updates[0]
            if (
                not previous
                and not isinstance(graph_input, Command)
                and isinstance(incoming, (list, Overwrite))
            ):
                # Let LangGraph persist the first input before applying its reducer.
                updates["messages"] = self.prepend(
                    incoming.value if isinstance(incoming, Overwrite) else incoming
                )
                return updates
            # Apply the application's reducer once, then prepend after the merge.
            durable = [
                message for message in previous if not _is_request_instruction(message)
            ]
            merged = channel.from_checkpoint(deepcopy(durable))
            merged.update(incoming_updates)
            messages = self.prepend(merged.get())
        updates["messages"] = (
            Overwrite(messages)
            if isinstance(channel, (BinaryOperatorAggregate, DeltaChannel))
            else messages
        )
        if isinstance(graph_input, Command):
            return replace(
                graph_input,
                update=(
                    updates
                    if graph_input.update is None
                    or isinstance(graph_input.update, dict)
                    else [
                        *(write for write in writes if write[0] != "messages"),
                        ("messages", updates["messages"]),
                    ]
                ),
            )
        return updates

    async def finish(
        self, graph: CompiledStateGraph, config: RunnableConfig
    ) -> RunnableConfig:
        if self.message is None and not self._cleanup_required:
            return config
        if not isinstance(graph, CompiledStateGraph) or not isinstance(
            graph.checkpointer, BaseCheckpointSaver
        ):
            return config
        state = await graph.aget_state(config)
        if state.next or state.tasks or state.interrupts:
            return config
        messages = state.values.get("messages", [])
        clean = [
            message for message in messages if not _is_request_instruction(message)
        ]
        if len(clean) == len(messages):
            return config
        saver = graph.checkpointer
        saved = await saver.aget_tuple(state.config)
        if saved is None:
            raise ValueError("The completed response's graph checkpoint is missing.")
        step = saved.metadata.get("step", -1) + 1
        checkpoint = create_checkpoint(saved.checkpoint, None, step)
        checkpoint["channel_values"] = {
            **saved.checkpoint["channel_values"],
            "messages": clean,
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
