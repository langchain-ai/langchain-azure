# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local instructions overlaid on durable graph messages."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, TypeVar

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.constants import TASKS
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt, Overwrite, Send
from pydantic import BaseModel
from wrapt import ObjectProxy

_INSTRUCTIONS_CONFIG_KEY = "__langchain_azure_response_instructions"
_INSTRUCTIONS_ID_PREFIX = "__langchain_azure_response_instructions:"
_GraphT = TypeVar("_GraphT", bound=Pregel)


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

    @contextmanager
    def bind(self) -> Iterator[None]:
        token = _active_instructions.set(self)
        try:
            yield
        finally:
            _active_instructions.reset(token)

    def wrap_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        return _InstructionStream(stream, self)

    def prepend(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        if self.message is None:
            return list(messages)
        return [
            self.message.model_copy(deep=True),
            *(message for message in messages if not _is_request_instruction(message)),
        ]

    def _overlay_sends(self, value: Any) -> Any:
        if isinstance(value, list):
            result = [self._overlay_sends(item) for item in value]
            return (
                value
                if all(left is right for left, right in zip(result, value))
                else result
            )
        if not isinstance(value, Send):
            return value
        argument = value.arg
        if isinstance(argument, dict):
            messages = argument.get("messages")
        elif (
            isinstance(argument, BaseModel)
            and "messages" in type(argument).model_fields
        ):
            messages = getattr(argument, "messages")
        elif (
            is_dataclass(argument)
            and not isinstance(argument, type)
            and any(item.name == "messages" for item in fields(argument))
        ):
            messages = getattr(argument, "messages")
        else:
            return value
        if not isinstance(messages, list):
            return value
        messages = self.prepend(messages)
        if isinstance(argument, dict):
            argument = {**argument, "messages": messages}
        elif isinstance(argument, BaseModel):
            argument = argument.model_copy(update={"messages": messages})
        else:
            argument = copy(argument)
            object.__setattr__(argument, "messages", messages)
        result_send = copy(value)
        result_send.arg = argument
        return result_send

    def overlay(self, value: CheckpointTuple | None) -> CheckpointTuple | None:
        if value is None or self.message is None:
            return value
        channel_values = value.checkpoint["channel_values"]
        updates: dict[str, Any] = {}
        sends = channel_values.get(TASKS)
        overlaid_sends = self._overlay_sends(sends)
        if overlaid_sends is not sends:
            updates[TASKS] = overlaid_sends
        checkpoint = value.checkpoint
        if updates:
            checkpoint = {
                **checkpoint,
                "channel_values": {**channel_values, **updates},
            }
        pending_writes = value.pending_writes
        if pending_writes:
            overlaid_writes = [
                (task_id, channel, self._overlay_sends(write))
                if channel == TASKS
                else (task_id, channel, write)
                for task_id, channel, write in pending_writes
            ]
            if any(
                left[2] is not right[2]
                for left, right in zip(overlaid_writes, pending_writes)
            ):
                pending_writes = overlaid_writes
        if checkpoint is value.checkpoint and pending_writes is value.pending_writes:
            return value
        return value._replace(checkpoint=checkpoint, pending_writes=pending_writes)


_active_instructions: ContextVar[_ResponseInstructions | None] = ContextVar(
    "langchain_azure_response_instructions", default=None
)


class _InstructionStream(AsyncIterator[Any]):
    """Bind each operation without leaking context to stream consumers."""

    def __init__(
        self, stream: AsyncIterator[Any], instructions: _ResponseInstructions
    ) -> None:
        self._stream = stream
        self._instructions = instructions

    async def __anext__(self) -> Any:
        with self._instructions.bind():
            return await anext(self._stream)

    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if close is not None:
            with self._instructions.bind():
                await close()


class _InstructionMessagesChannel(ObjectProxy):
    """Keep instructions outside the reducer's value and checkpoint."""

    def __init__(self, channel: BaseChannel) -> None:
        super().__init__(channel)

    def copy(self) -> _InstructionMessagesChannel:
        return type(self)(self.__wrapped__.copy())

    def from_checkpoint(self, checkpoint: Any) -> _InstructionMessagesChannel:
        return type(self)(self.__wrapped__.from_checkpoint(checkpoint))

    def get(self) -> Any:
        value = self.__wrapped__.get()
        instructions = _active_instructions.get()
        if instructions is None:
            return value
        if instructions.message is None:
            return _without_request_instructions(value)
        if not isinstance(value, list):
            raise ValueError(
                "Request instructions require a list-valued messages channel."
            )
        return instructions.prepend(value)

    def update(self, values: Sequence[Any]) -> bool:
        if _active_instructions.get() is not None:
            values = _without_request_instructions(values)
        return self.__wrapped__.update(values)


def with_response_instructions(graph: _GraphT) -> _GraphT:
    """Enable request-scoped instructions in a compiled graph's message reads.

    ResponsesHostServer applies this automatically to its graph and statically
    discoverable local subgraphs. Apply it explicitly to subgraphs constructed at
    runtime or invoked through indirection.

    The returned graph is a shallow copy. Discoverable subgraphs share their
    existing graph objects and receive transparent message-channel adapters.
    Outside a hosted Responses request, channel behavior is unchanged.

    Args:
        graph: A local compiled graph. Its ``messages`` channels, when present,
            must be list-valued.

    Returns:
        A compiled graph whose reducers and checkpoints exclude temporary
        request instructions, while execution reads include the current ones.

    Examples:
        Prepare a subgraph inside a runtime factory before invoking it::

            child = with_response_instructions(builder.compile(checkpointer=True))
            result = await child.ainvoke(state)
    """
    prepared = graph.copy()
    for child in [prepared, *(child for _, child in graph.get_subgraphs(recurse=True))]:
        if not isinstance(child, Pregel):
            continue
        channel = child.channels.get("messages")
        if isinstance(channel, BaseChannel) and not isinstance(
            channel, _InstructionMessagesChannel
        ):
            child.channels = {
                **child.channels,
                "messages": _InstructionMessagesChannel(channel),
            }
    return prepared


def _instructions_from_config(
    config: RunnableConfig,
) -> _ResponseInstructions | None:
    value = (config.get("configurable") or {}).get(_INSTRUCTIONS_CONFIG_KEY)
    return value if isinstance(value, _ResponseInstructions) else None
