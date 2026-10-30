# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local instructions overlaid on durable graph messages."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from operator import add
from typing import Any

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.channels import BinaryOperatorAggregate
from langgraph.checkpoint.base import CheckpointTuple, DeltaChannelHistory
from langgraph.types import Command, Interrupt, Overwrite, Send
from pydantic import BaseModel

_INSTRUCTIONS_CONFIG_KEY = "__langchain_azure_response_instructions"
_INSTRUCTIONS_ID_PREFIX = "__langchain_azure_response_instructions:"


def _checkpoint_messages(value: Any) -> list[AnyMessage]:
    # The public channel decoder also unwraps DeltaChannel snapshots.
    messages = BinaryOperatorAggregate(list, add).from_checkpoint(value).get()
    if not isinstance(messages, list):
        raise ValueError("Request instructions require a list-valued messages channel.")
    return messages


def _is_request_instruction(value: Any) -> bool:
    if isinstance(value, BaseMessage):
        message_id = value.id
    elif isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, dict) and value.get("type") == "system":
            return _is_request_instruction(data)
        message_id = value.get("id")
    else:
        return False
    return isinstance(message_id, str) and message_id.startswith(
        _INSTRUCTIONS_ID_PREFIX
    )


def _without_request_instructions(value: Any) -> Any:
    """Project only host-origin messages out of snapshots and task payloads."""
    if _is_request_instruction(value):
        return []
    if isinstance(value, BaseMessage):
        return value
    if isinstance(value, (list, deque)):
        result = [
            _without_request_instructions(item)
            for item in value
            if not _is_request_instruction(item)
        ]
        if len(result) == len(value) and all(
            left is right for left, right in zip(result, value)
        ):
            return value
        return (
            deque(result, maxlen=value.maxlen) if isinstance(value, deque) else result
        )
    if isinstance(value, tuple):
        result_tuple = tuple(_without_request_instructions(item) for item in value)
        if all(left is right for left, right in zip(result_tuple, value)):
            return value
        return type(value)(*result_tuple) if hasattr(value, "_fields") else result_tuple
    if isinstance(value, dict):
        result_dict = {
            key: _without_request_instructions(item) for key, item in value.items()
        }
        return (
            value
            if all(result_dict[key] is item for key, item in value.items())
            else result_dict
        )
    if isinstance(value, Send):
        argument = _without_request_instructions(value.arg)
        if argument is value.arg:
            return value
        result_send = copy(value)
        result_send.arg = argument
        return result_send
    if isinstance(value, Command):
        update = _without_request_instructions(value.update)
        resume = _without_request_instructions(value.resume)
        goto = _without_request_instructions(value.goto)
        if update is value.update and resume is value.resume and goto is value.goto:
            return value
        return replace(value, update=update, resume=resume, goto=goto)
    if isinstance(value, (Interrupt, Overwrite)):
        overwritten = _without_request_instructions(value.value)
        return (
            value if overwritten is value.value else replace(value, value=overwritten)
        )
    if isinstance(value, BaseModel):
        updates = {
            name: projected
            for name in type(value).model_fields
            if (projected := _without_request_instructions(getattr(value, name)))
            is not getattr(value, name)
        }
        return value.model_copy(update=updates) if updates else value
    if is_dataclass(value) and not isinstance(value, type):
        result_dataclass = value
        for item in fields(value):
            original = getattr(value, item.name)
            projected = _without_request_instructions(original)
            if projected is original:
                continue
            if result_dataclass is value:
                result_dataclass = copy(value)
            # Preserve frozen/slots and init=False fields without rerunning init.
            object.__setattr__(result_dataclass, item.name, projected)
        return result_dataclass
    return value


@dataclass
class _ResponseInstructions:
    """Keep one response's instructions separate from durable history."""

    message: SystemMessage | None
    checkpoint_restored: bool | None = field(default=None, init=False)

    @classmethod
    def create(cls, instructions: object, response_id: str) -> _ResponseInstructions:
        if instructions is None:
            return cls(None)
        if not isinstance(instructions, str):
            raise ValueError(
                "ResponsesHostServer instructions must be a string or null."
            )
        if not instructions:
            return cls(None)
        return cls(
            SystemMessage(
                content=instructions,
                id=f"{_INSTRUCTIONS_ID_PREFIX}{response_id}",
                additional_kwargs={"__openai_role__": "developer"},
            )
        )

    def with_config(self, config: RunnableConfig) -> RunnableConfig:
        return {
            **config,
            "configurable": {
                **(config.get("configurable") or {}),
                _INSTRUCTIONS_CONFIG_KEY: self,
            },
        }

    def prepend(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        if self.message is None:
            return list(messages)
        return [
            self.message.model_copy(deep=True),
            *(message for message in messages if not _is_request_instruction(message)),
        ]

    def overlay(self, value: CheckpointTuple | None) -> CheckpointTuple | None:
        if value is None or self.message is None:
            return value
        channel_values = value.checkpoint["channel_values"]
        namespace = (value.config.get("configurable") or {}).get("checkpoint_ns", "")
        if "messages" not in channel_values:
            if namespace or "messages" in value.checkpoint["channel_versions"]:
                return value
        checkpoint = value.checkpoint.copy()
        checkpoint["channel_values"] = {
            **channel_values,
            "messages": self.prepend(
                _checkpoint_messages(channel_values.get("messages", []))
            ),
        }
        return value._replace(checkpoint=checkpoint)

    def overlay_history(
        self, histories: Mapping[str, DeltaChannelHistory]
    ) -> Mapping[str, DeltaChannelHistory]:
        if self.message is None or "messages" not in histories:
            return histories
        history = histories["messages"]
        return {
            **histories,
            "messages": {
                **history,
                "seed": self.prepend(_checkpoint_messages(history.get("seed", []))),
            },
        }


def _instructions_from_config(
    config: RunnableConfig,
) -> _ResponseInstructions | None:
    value = (config.get("configurable") or {}).get(_INSTRUCTIONS_CONFIG_KEY)
    return value if isinstance(value, _ResponseInstructions) else None
