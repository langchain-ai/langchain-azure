"""Request-local instructions for hosted model calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config

_INSTRUCTIONS_CONFIG_KEY = "langchain_response_instructions"
_INSTRUCTIONS_MODE_HEADER = "x-client-langchain-instructions-mode"


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
