# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local instructions overlaid on durable graph messages."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, TypeVar, overload

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    SystemMessage,
    message_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langgraph.cache.base import BaseCache, FullKey
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.constants import TASKS
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt, Overwrite, Send
from pydantic import BaseModel
from wrapt import ObjectProxy

_INSTRUCTIONS_CONFIG_KEY = "__langchain_azure_response_instructions"
_INSTRUCTIONS_ID_PREFIX = "__langchain_azure_response_instructions:"
_INSTRUCTIONS_REFERENCE_ID = "__langchain_azure_response_instruction_reference__"
_TASK_RETURN_CHANNEL = "__return__"
_MESSAGE_TYPES = frozenset(
    {
        "ai",
        "chat",
        "function",
        "human",
        "remove",
        "system",
        "tool",
        "AIMessageChunk",
        "ChatMessageChunk",
        "FunctionMessageChunk",
        "HumanMessageChunk",
        "SystemMessageChunk",
        "ToolMessageChunk",
    }
)
_MESSAGE_KINDS = _MESSAGE_TYPES | {"assistant", "developer", "user"}
_GraphT = TypeVar("_GraphT", bound=Pregel)


def _message_payload(value: Any) -> BaseMessage | dict[str, Any] | None:
    if isinstance(value, BaseMessage):
        return value
    elif isinstance(value, dict):
        message_type = value.get("type")
        if isinstance(message_type, str) and message_type == "constructor":
            namespace = value.get("id")
            version = value.get("lc")
            if (
                isinstance(version, int)
                and version == 1
                and isinstance(namespace, list)
                and all(isinstance(part, str) for part in namespace)
                and (
                    namespace[:-1] == ["langchain", "schema", "messages"]
                    or (
                        len(namespace) >= 3
                        and namespace[:2] == ["langchain_core", "messages"]
                    )
                )
            ):
                return _message_payload(value.get("kwargs"))
            return None
        kind = (
            value.get("role")
            if message_type is None
            or (isinstance(message_type, str) and message_type == "message")
            else message_type
        )
        additional_kwargs = value.get("additional_kwargs")
        if (
            kind is None
            and isinstance(additional_kwargs, dict)
            and additional_kwargs.get("__openai_role__") == "developer"
        ):
            kind = "developer"
        if not isinstance(kind, str) or kind not in _MESSAGE_KINDS:
            return None
        if "content" in value or (kind == "remove" and "id" in value):
            return value
        elif isinstance(message_type, str) and message_type in _MESSAGE_TYPES:
            data = value.get("data")
            if not isinstance(data, dict) or (
                "content" not in data and message_type != "remove"
            ):
                return None
            return data
    return None


def _message_id(value: Any) -> Any:
    payload = _message_payload(value)
    if isinstance(payload, BaseMessage):
        return payload.id
    return payload.get("id") if payload is not None else None


def _is_request_instruction(value: Any) -> bool:
    message_id = _message_id(value)
    return isinstance(message_id, str) and message_id.startswith(
        _INSTRUCTIONS_ID_PREFIX
    )


def _is_instruction_slot(value: Any) -> bool:
    message_id = _message_id(value)
    return isinstance(message_id, str) and (
        message_id.startswith(_INSTRUCTIONS_ID_PREFIX)
        or message_id == _INSTRUCTIONS_REFERENCE_ID
    )


def _instruction_like(value: Any, message: SystemMessage) -> Any:
    if isinstance(value, BaseMessage):
        return message.model_copy(deep=True)
    if value.get("type") == "constructor":
        return message.to_json()
    if "role" in value:
        return {
            "role": "developer",
            "content": message.content,
            "id": message.id,
            **({"type": "message"} if value.get("type") == "message" else {}),
        }
    if "content" not in value and isinstance(value.get("data"), dict):
        return message_to_dict(message)
    return message.model_dump()


def _project_request_instructions(
    value: Any,
    replacement: SystemMessage | None = None,
    *,
    message_sequence: bool = False,
) -> Any:
    if _is_instruction_slot(value):
        return _instruction_like(value, replacement) if replacement is not None else []
    if isinstance(value, (list, deque)):
        result = [
            _project_request_instructions(
                item, replacement, message_sequence=message_sequence
            )
            for item in value
            if replacement is not None or not _is_instruction_slot(item)
        ]
        if len(result) == len(value) and all(
            left is right for left, right in zip(result, value)
        ):
            return value
        return (
            deque(result, maxlen=value.maxlen) if isinstance(value, deque) else result
        )
    if isinstance(value, tuple):
        message_value_index = -1
        if len(value) == 2 and isinstance(value[0], str) and value[0] == "messages":
            message_value_index = 1
        elif len(value) == 3 and isinstance(value[1], str) and value[1] == "messages":
            message_value_index = 2
        message_tuple = not hasattr(value, "_fields") and (
            message_sequence
            or all(_message_payload(item) is not None for item in value)
        )
        result_tuple = tuple(
            _project_request_instructions(
                item,
                replacement,
                message_sequence=message_tuple or index == message_value_index,
            )
            for index, item in enumerate(value)
            if not (
                replacement is None and message_tuple and _is_instruction_slot(item)
            )
        )
        if len(result_tuple) == len(value) and all(
            left is right for left, right in zip(result_tuple, value)
        ):
            return value
        return type(value)(*result_tuple) if hasattr(value, "_fields") else result_tuple
    if isinstance(value, dict):
        result_dict = {
            key: _project_request_instructions(
                item, replacement, message_sequence=key == "messages"
            )
            for key, item in value.items()
        }
        return (
            value
            if all(result_dict[key] is item for key, item in value.items())
            else result_dict
        )
    if isinstance(value, Send):
        argument = _project_request_instructions(value.arg, replacement)
        if argument is value.arg:
            return value
        result_send = copy(value)
        result_send.arg = argument
        return result_send
    if isinstance(value, Command):
        update = _project_request_instructions(value.update, replacement)
        resume = _project_request_instructions(value.resume, replacement)
        goto = _project_request_instructions(value.goto, replacement)
        if update is value.update and resume is value.resume and goto is value.goto:
            return value
        return replace(value, update=update, resume=resume, goto=goto)
    if isinstance(value, (Interrupt, Overwrite)):
        overwritten = _project_request_instructions(
            value.value, replacement, message_sequence=message_sequence
        )
        return (
            value if overwritten is value.value else replace(value, value=overwritten)
        )
    if isinstance(value, BaseModel):
        values = {name: getattr(value, name) for name in type(value).model_fields}
        values.update(value.model_extra or {})
        updates = {
            name: projected
            for name, original in values.items()
            if (
                projected := _project_request_instructions(
                    original, replacement, message_sequence=name == "messages"
                )
            )
            is not original
        }
        return value.model_copy(update=updates) if updates else value
    if is_dataclass(value) and not isinstance(value, type):
        result_dataclass = value
        for item in fields(value):
            original = getattr(value, item.name)
            projected = _project_request_instructions(
                original, replacement, message_sequence=item.name == "messages"
            )
            if projected is original:
                continue
            if result_dataclass is value:
                result_dataclass = copy(value)
            # Preserve frozen/slots and init=False fields without rerunning init.
            object.__setattr__(result_dataclass, item.name, projected)
        return result_dataclass
    return value


def _without_request_instructions(value: Any, *, message_sequence: bool = False) -> Any:
    """Project only host-origin messages out of snapshots and task payloads."""
    return _project_request_instructions(value, message_sequence=message_sequence)


def _instruction_references(value: Any) -> Any:
    """Preserve replay slots without persisting instruction text or response IDs."""
    return _project_request_instructions(
        value, SystemMessage(content="", id=_INSTRUCTIONS_REFERENCE_ID)
    )


def _prepare_instruction_writes(
    writes: Sequence[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    return [
        (
            channel,
            _instruction_references(value)
            if channel == _TASK_RETURN_CHANNEL
            else _without_request_instructions(value),
        )
        for channel, value in writes
    ]


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
        durable = [message for message in messages if not _is_instruction_slot(message)]
        if self.message is None:
            return durable
        return [
            self.message.model_copy(deep=True),
            *durable,
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
        elif isinstance(argument, BaseModel) and (
            "messages" in type(argument).model_fields
            or "messages" in (argument.model_extra or {})
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
        if not isinstance(messages, (list, tuple)):
            return value
        overlaid = self.prepend(messages)
        messages = tuple(overlaid) if isinstance(messages, tuple) else overlaid
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

    @overload
    def overlay(self, value: None) -> None: ...

    @overload
    def overlay(self, value: CheckpointTuple) -> CheckpointTuple: ...

    def overlay(self, value: CheckpointTuple | None) -> CheckpointTuple | None:
        if value is None:
            return value
        channel_values = value.checkpoint["channel_values"]
        updates: dict[str, Any] = {}
        sends = channel_values.get(TASKS)
        overlaid_sends = (
            self._overlay_sends(sends) if self.message is not None else sends
        )
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
            overlaid_writes = []
            for task_id, channel, write in pending_writes:
                if channel == TASKS and self.message is not None:
                    overlaid = self._overlay_sends(write)
                elif channel == _TASK_RETURN_CHANNEL:
                    overlaid = _project_request_instructions(write, self.message)
                else:
                    overlaid = write
                overlaid_writes.append((task_id, channel, overlaid))
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
            values = _without_request_instructions(values, message_sequence=True)
        return self.__wrapped__.update(values)


class _InstructionTasksChannel(_InstructionMessagesChannel):
    """Overlay fresh and restored Send payloads at the same task-read boundary."""

    def get(self) -> Any:
        value = self.__wrapped__.get()
        instructions = _active_instructions.get()
        if instructions is None:
            return value
        if instructions.message is None:
            return _without_request_instructions(value)
        return instructions._overlay_sends(value)


class _InstructionCache(ObjectProxy):
    """Keep cached instruction slots request-local without rerunning tasks."""

    def _restore(self, value: Any) -> Any:
        instructions = _active_instructions.get()
        return _project_request_instructions(
            value, instructions.message if instructions is not None else None
        )

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        return self._restore(self.__wrapped__.get(keys))

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        return self._restore(await self.__wrapped__.aget(keys))

    def set(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        self.__wrapped__.set(_instruction_references(dict(pairs)))

    async def aset(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        await self.__wrapped__.aset(_instruction_references(dict(pairs)))


def with_response_instructions(graph: _GraphT) -> _GraphT:
    """Enable request-scoped instructions in a compiled graph's message reads.

    ResponsesHostServer applies this automatically to its graph and statically
    discoverable local subgraphs. Apply it explicitly to subgraphs constructed at
    runtime or invoked through indirection.

    The returned graph is a shallow copy. Discoverable subgraphs share their
    existing graph objects and receive transparent message-channel, task-channel,
    and cache adapters. Apply this helper after configuration that creates copies.
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
    prepared = graph.copy({"channels": dict(graph.channels)})
    for child in [prepared, *(child for _, child in graph.get_subgraphs(recurse=True))]:
        if not isinstance(child, Pregel):
            continue
        for key, channel_type in (
            ("messages", _InstructionMessagesChannel),
            (TASKS, _InstructionTasksChannel),
        ):
            channel = child.channels.get(key)
            if isinstance(channel, BaseChannel) and not isinstance(
                channel, channel_type
            ):
                child.channels = {**child.channels, key: channel_type(channel)}
        if isinstance(child.cache, BaseCache) and not isinstance(
            child.cache, _InstructionCache
        ):
            child.cache = _InstructionCache(child.cache)
    return prepared


def _instructions_from_config(
    config: RunnableConfig,
) -> _ResponseInstructions | None:
    value = (config.get("configurable") or {}).get(_INSTRUCTIONS_CONFIG_KEY)
    return value if isinstance(value, _ResponseInstructions) else None
