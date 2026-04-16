"""Runtime host for deploying LangGraph graphs on Azure AI Foundry Agent Server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_azure_ai._api.base import experimental

logger = logging.getLogger(__name__)

_IMPORT_ERROR_MSG = (
    "azure-ai-agentserver-invocations is required to use "
    "AzureAIAgentServerRuntime. Install it with: "
    "`pip install azure-ai-agentserver-invocations azure-ai-agentserver-core` "
    "or install the 'runtime' extra: `pip install langchain-azure-ai[runtime]`"
)


@experimental()
class AzureAIAgentServerRuntime:
    """Runtime host for deploying a LangGraph graph on Azure AI Foundry Agent Server.

    Wraps a compiled LangGraph graph and handles the boilerplate required to
    expose it as a hosted agent via ``azure-ai-agentserver-invocations``.

    The class registers an invoke handler on an ``InvocationAgentServerHost``
    instance.  When a request arrives, the handler:

    1. Parses the JSON body using ``request_parser`` (default: extracts
       ``message`` / ``input`` / ``query`` and wraps it in a
       ``HumanMessage``).
    2. Invokes the graph asynchronously.
    3. Formats the text response using ``response_formatter`` (default: last
       ``AIMessage`` in ``result["messages"]``, falling back to
       ``result["output"]`` / ``result["response"]`` /
       ``result["final_answer"]``).
    4. Returns ``{"response": <text>}`` as JSON.

    Note:
        The ``request_parser`` and ``response_formatter`` parameters are
        intentionally named differently from LangGraph's built-in
        ``graph.input_schema`` / ``graph.output_schema`` (which are Pydantic
        models that describe the graph state type).  These parameters are
        HTTP-layer adapters: ``request_parser`` converts an incoming HTTP
        request body into a graph input dict, and ``response_formatter``
        converts the graph output dict into an HTTP response string.

    Example:
        .. code-block:: python

            from app.graph import build_graph
            from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

            graph = build_graph()
            runtime = AzureAIAgentServerRuntime(graph=graph)

            if __name__ == "__main__":
                runtime.run()

    Args:
        graph: A compiled LangGraph graph (or any
            :class:`~langchain_core.runnables.Runnable`)
            that accepts a state dict with a ``messages`` key as input.
        request_parser: Optional callable
            ``(data: dict[str, Any]) -> dict[str, Any]`` that converts the
            raw HTTP request body to a graph-compatible input dict.  Defaults
            to :func:`default_request_parser`.
        response_formatter: Optional callable
            ``(result: dict[str, Any]) -> str`` that extracts the final text
            response from the graph output dict.  Defaults to
            :func:`default_response_formatter`.
        config: Optional :class:`~langchain_core.runnables.RunnableConfig`
            passed to every ``graph.ainvoke()`` call.  Use this to set a
            fixed ``thread_id`` or custom callbacks.
    """

    def __init__(
        self,
        *,
        graph: Runnable,
        request_parser: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        response_formatter: Optional[Callable[[dict[str, Any]], str]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> None:
        try:
            from azure.ai.agentserver.invocations import (  # type: ignore[import]
                InvocationAgentServerHost,
            )
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MSG) from exc

        self._graph = graph
        self._request_parser: Callable[[dict[str, Any]], dict[str, Any]] = (
            request_parser if request_parser is not None else default_request_parser
        )
        self._response_formatter: Callable[[dict[str, Any]], str] = (
            response_formatter
            if response_formatter is not None
            else default_response_formatter
        )
        self._config = config
        self._host: Any = InvocationAgentServerHost()
        self._register_handler()

    def _register_handler(self) -> None:
        """Register the invoke handler on the ``InvocationAgentServerHost``."""
        from starlette.requests import Request  # type: ignore[import]
        from starlette.responses import JSONResponse, Response  # type: ignore[import]

        graph = self._graph
        request_parser = self._request_parser
        response_formatter = self._response_formatter
        config = self._config

        @self._host.invoke_handler
        async def handle_invoke(request: Request) -> Response:
            data: dict[str, Any] = await request.json()
            initial_state = request_parser(data)
            try:
                result: dict[str, Any] = await graph.ainvoke(
                    initial_state, config=config
                )
            except Exception:
                logger.exception("Unhandled error while invoking graph")
                raise
            response_text = response_formatter(result)
            return JSONResponse({"response": response_text})

    def run(self, **kwargs: Any) -> None:
        """Start the agent server.

        Args:
            **kwargs: Additional keyword arguments forwarded to
                ``InvocationAgentServerHost.run()``.
        """
        self._host.run(**kwargs)

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Optional[Path] = None,
        graph_name: Optional[str] = None,
        request_parser: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        response_formatter: Optional[Callable[[dict[str, Any]], str]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> "AzureAIAgentServerRuntime":
        """Create an :class:`AzureAIAgentServerRuntime` from a ``langgraph.json`` file.

        Reads the standard ``langgraph.json`` configuration that LangGraph
        projects use to declare their graphs and environment.  This lets you
        write a minimal ``main.py``:

        .. code-block:: python

            from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

            runtime = AzureAIAgentServerRuntime.from_config()

            if __name__ == "__main__":
                runtime.run()

        The method:

        1. Locates ``langgraph.json`` (in the current working directory by
           default, or at the path you provide).
        2. Loads the ``.env`` file referenced by the ``env`` field, if present,
           using ``python-dotenv`` (existing environment variables are *not*
           overridden).
        3. Dynamically imports the graph specified by ``graph_name`` (or the
           only graph defined if there is exactly one).
        4. Constructs and returns an :class:`AzureAIAgentServerRuntime`.

        Args:
            config_path: Path to the ``langgraph.json`` file.  Defaults to
                ``langgraph.json`` in the current working directory.
            graph_name: Name of the graph entry to load from the ``graphs``
                mapping.  Required only when the config defines more than one
                graph.
            request_parser: Forwarded to
                :class:`AzureAIAgentServerRuntime.__init__`.
            response_formatter: Forwarded to
                :class:`AzureAIAgentServerRuntime.__init__`.
            config: Forwarded to :class:`AzureAIAgentServerRuntime.__init__`.

        Returns:
            A fully initialised :class:`AzureAIAgentServerRuntime`.

        Raises:
            FileNotFoundError: If ``langgraph.json`` or the referenced
                ``.env`` file cannot be found.
            ValueError: If the config is invalid or ``graph_name`` is
                ambiguous / not found.
            ImportError: If the graph module or ``python-dotenv`` cannot be
                imported.
        """
        from langchain_azure_ai.agents.runtime._config import (
            load_config,
            load_env,
            resolve_graph,
        )

        resolved_config_path = (
            Path(config_path)
            if config_path is not None
            else Path.cwd() / "langgraph.json"
        )
        base_dir = resolved_config_path.parent

        cfg = load_config(resolved_config_path)

        if "env" in cfg:
            load_env(base_dir / cfg["env"])

        graph = resolve_graph(cfg, base_dir, graph_name)

        return cls(
            graph=graph,
            request_parser=request_parser,
            response_formatter=response_formatter,
            config=config,
        )


def default_request_parser(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw HTTP request body to a LangGraph-compatible input dict.

    Looks for ``message``, ``input``, or ``query`` keys (in that order) in
    the request body and wraps the value in a ``HumanMessage``.

    Unlike ``graph.input_schema`` (a Pydantic model that describes the graph's
    state type), this function is an HTTP-layer adapter: it translates an
    incoming request payload into the dict that the graph expects as input.

    Args:
        data: The parsed JSON request body.

    Returns:
        A dict with a ``messages`` key containing a single
        :class:`~langchain_core.messages.HumanMessage`.
    """
    user_input: str = (
        data.get("message") or data.get("input") or data.get("query") or ""
    )
    return {"messages": [HumanMessage(content=user_input)]}


def default_response_formatter(result: dict[str, Any]) -> str:
    """Extract the final text response from a LangGraph output dict.

    Unlike ``graph.output_schema`` (a Pydantic model that describes the
    graph's output state type), this function is an HTTP-layer adapter: it
    converts the graph's output dict into the string sent back in the HTTP
    response body.

    Checks, in order:

    1. The content of the last :class:`~langchain_core.messages.AIMessage`
       in ``result["messages"]``.
    2. The values of ``result["output"]``, ``result["response"]``, and
       ``result["final_answer"]`` (first non-empty one wins).

    Args:
        result: The dict returned by ``graph.ainvoke()``.

    Returns:
        The final text response as a string, or an empty string if no
        response could be extracted.
    """
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))

    for key in ("output", "response", "final_answer"):
        val = result.get(key)
        if val:
            return str(val)

    return ""
