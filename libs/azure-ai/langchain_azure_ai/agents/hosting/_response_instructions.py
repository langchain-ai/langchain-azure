"""Request-local instructions for hosted model calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import RemoveMessage, SystemMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES

_INSTRUCTIONS_CONFIG_KEY = "langchain_response_instructions"
_INSTRUCTIONS_MODE_HEADER = "x-client-langchain-instructions-mode"
_INSTRUCTIONS_PROVENANCE = "langchain_response_instructions_v1"
_INSTRUCTIONS_SOURCE = "langchain_response_instructions_source_v1"


class _InstructionProvenance:
    def __init__(self) -> None:
        self._verified: dict[tuple[str, str, str], tuple[str, bool]] = {}
        self._writes: dict[
            tuple[str, str, str], dict[str, tuple[bool, dict[str, bool]] | None]
        ] = {}

    @staticmethod
    def _key(config: RunnableConfig, checkpoint_id: str) -> tuple[str, str, str]:
        configurable = config.get("configurable") or {}
        return (
            configurable.get("thread_id", ""),
            configurable.get("checkpoint_ns", ""),
            checkpoint_id,
        )

    def record_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        if not any(channel == "messages" for channel, _ in writes):
            return
        checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")
        if not checkpoint_id:
            return
        key = self._key(config, checkpoint_id)
        tasks = self._writes.setdefault(key, {})
        if task_id in tasks:
            return
        cleared = False
        identities: dict[str, bool] = {}
        for channel, value in writes:
            if channel != "messages":
                continue
            try:
                messages = convert_to_messages(
                    value if isinstance(value, list) else [value]
                )
            except (TypeError, ValueError, NotImplementedError):
                tasks[task_id] = None
                return
            for message in messages:
                if isinstance(message, RemoveMessage):
                    if message.id == REMOVE_ALL_MESSAGES:
                        cleared = True
                        identities.clear()
                    elif message.id:
                        identities[message.id] = False
                else:
                    if message.id and message.id.startswith("response-instructions-"):
                        identities[message.id] = True
                    source = message.additional_kwargs.get(_INSTRUCTIONS_PROVENANCE)
                    if isinstance(source, str) and source:
                        identities[f"response-instructions-{source}"] = True
        tasks[task_id] = cleared, identities

    def observe(self, saved: CheckpointTuple | None) -> None:
        if saved is None:
            return
        self.checkpoint_metadata(saved.config, saved.checkpoint, saved.metadata)
        tasks: dict[str, list[tuple[str, Any]]] = {}
        for task_id, channel, value in saved.pending_writes or []:
            tasks.setdefault(task_id, []).append((channel, value))
        for task_id, writes in tasks.items():
            self.record_writes(saved.config, writes, task_id)

    def checkpoint_metadata(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> CheckpointMetadata:
        combined = {**(config.get("metadata") or {}), **metadata}
        source = combined.get(_INSTRUCTIONS_SOURCE)
        if (
            combined.get(_INSTRUCTIONS_PROVENANCE) != "1"
            or not isinstance(source, str)
            or not source
        ):
            return metadata
        instruction_id = f"response-instructions-{source}"
        messages = checkpoint["channel_values"].get("messages", [])
        tagged = [
            (message.id, message.additional_kwargs[_INSTRUCTIONS_PROVENANCE])
            for message in messages
            if isinstance(message, SystemMessage)
            and _INSTRUCTIONS_PROVENANCE in message.additional_kwargs
        ]
        key = self._key(config, checkpoint["id"])
        if tagged == [(instruction_id, source)]:
            self._verified[key] = source, False
            return metadata
        if tagged or any(
            getattr(message, "id", None) == instruction_id for message in messages
        ):
            return metadata
        parent_id = (config.get("configurable") or {}).get("checkpoint_id", "")
        parent_key = self._key(config, parent_id)
        parent = self._verified.get(parent_key)
        if parent is None or parent[0] != source:
            return metadata
        changes: list[bool | None] = []
        for update in self._writes.get(parent_key, {}).values():
            if update is None:
                return metadata
            cleared, identities = update
            changes.append(identities.get(instruction_id, False if cleared else None))
        if True in changes or not (parent[1] or False in changes):
            return metadata
        self._verified[key] = source, True
        return cast(CheckpointMetadata, {**metadata, _INSTRUCTIONS_SOURCE: ""})


@dataclass(frozen=True)
class _ResponseInstructions:
    value: str | None


def get_response_instructions(config: RunnableConfig | None = None) -> str | None:
    """Return the current host request's transient instructions, if enabled.

    Use this in custom graph model nodes when the host is configured with
    ``instructions_mode="context"``. Apply the returned text only to the model
    input, not to graph state. It is absent for the default ``"messages"`` mode
    and outside a hosted invocation. The application's runtime context is not
    changed.

    Args:
        config: The current execution config. Defaults to LangGraph's active
            runnable config when called inside a graph node or middleware.

    Returns:
        Nonempty instructions for this response, or ``None`` when omitted,
        cleared, disabled, or called outside a graph execution.

    Example:
        Read instructions inside a custom model node::

            instructions = get_response_instructions(config)
            model_messages = list(state["messages"])
            if instructions:
                model_messages.insert(0, SystemMessage(content=instructions))
            response = await model.ainvoke(model_messages, config)
    """
    if config is None:
        try:
            config = get_config()
        except RuntimeError:
            return None
    instructions = (config.get("configurable") or {}).get(_INSTRUCTIONS_CONFIG_KEY)
    return (
        instructions.value if isinstance(instructions, _ResponseInstructions) else None
    )


class ResponsesInstructionsMiddleware(AgentMiddleware):
    """Apply hosted instructions to model requests without persisting messages.

    Pair this middleware with ``ResponsesHostServer(instructions_mode="context")``.
    It appends the current response's instructions to the model's existing system
    message without mutating that message or graph state. Message summarization
    and persistent trimming therefore do not consume the temporary instructions.
    Both synchronous and asynchronous model calls are supported. Outside context
    mode the middleware passes requests through unchanged.

    Add this after middleware that replaces the model's system message. Do not
    also add the same instructions to graph state or another model middleware.

    Example:
        Opt in when constructing both the agent and its host::

            graph = create_agent(
                model,
                middleware=[ResponsesInstructionsMiddleware()],
                checkpointer=checkpointer,
            )
            host = ResponsesHostServer(graph, instructions_mode="context")
    """

    @staticmethod
    def _prepare(request: ModelRequest[Any]) -> ModelRequest[Any]:
        instructions = get_response_instructions()
        if not instructions:
            return request
        system_message = request.system_message
        if system_message is None:
            system_message = SystemMessage(content=instructions)
        else:
            content = system_message.content
            combined = (
                (f"{content}\n\n{instructions}" if content else instructions)
                if isinstance(content, str)
                else [
                    *content,
                    {"type": "text", "text": instructions},
                ]
            )
            system_message = system_message.model_copy(update={"content": combined})
        return request.override(system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Invoke the synchronous model with the current request's instructions.

        Args:
            request: The upstream model request, including its system message.
            handler: The next synchronous model handler.

        Returns:
            The handler's response, unchanged.
        """
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Invoke the asynchronous model with the current request's instructions.

        Args:
            request: The upstream model request, including its system message.
            handler: The next asynchronous model handler.

        Returns:
            The handler's response, unchanged.
        """
        return await handler(self._prepare(request))
