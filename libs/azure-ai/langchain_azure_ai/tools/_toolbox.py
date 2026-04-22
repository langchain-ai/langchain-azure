"""Azure AI Foundry Toolbox MCP integration for LangChain/LangGraph."""

from __future__ import annotations

import logging
import os
from types import TracebackType
from typing import Any, List, Optional, Type, Union

from azure.core.credentials import TokenCredential
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.utils.env import get_project_endpoint

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_CONSENT_ERROR_CODE: int = -32006
"""MCP error code returned by the Foundry gateway when OAuth consent is required."""

_TOKEN_AUDIENCE: str = "https://ai.azure.com/.default"
"""Azure AD token audience used to obtain Bearer tokens for Azure AI services."""

_DEFAULT_FEATURES: str = "Toolboxes=V1Preview"
"""Default value for the ``Foundry-Features`` request header."""

_FEATURES_HEADER: str = "Foundry-Features"
"""Header name for feature flags on Foundry MCP gateway requests."""

_DEFAULT_API_VERSION: str = "v1"
"""Default ``api-version`` value appended to the toolbox MCP URL."""


def _build_toolbox_mcp_url(
    project_endpoint: str, toolbox_name: str, api_version: str
) -> str:
    """Construct the full MCP URL for a named toolbox.

    The URL follows the Foundry REST convention::

        {project_endpoint}/toolboxes/{toolbox_name}/mcp?api-version={api_version}

    Args:
        project_endpoint: Azure AI Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
        toolbox_name: Name of the toolbox as configured in Azure AI Foundry.
        api_version: Toolbox API version string, e.g. ``"v1"``.

    Returns:
        The fully-qualified MCP endpoint URL string.
    """
    base = project_endpoint.rstrip("/")
    return f"{base}/toolboxes/{toolbox_name}/mcp?api-version={api_version}"


# ── OAuth consent-error helpers ────────────────────────────────────────────────


def _is_consent_error(exc: BaseException) -> bool:
    """Return True if *exc* contains an OAuth consent-URL error.

    The Foundry MCP gateway returns MCP error code -32006 when an OAuth
    connection has not yet been authorized. The MCP client may wrap this
    inside one or more ``ExceptionGroup`` / ``BaseExceptionGroup`` layers;
    this function recurses to find the error anywhere in the tree.

    Args:
        exc: The exception to inspect.

    Returns:
        ``True`` if the exception tree contains an MCP consent error.
    """
    error_data = getattr(exc, "error", None)
    if (
        error_data is not None
        and getattr(error_data, "code", None) == _CONSENT_ERROR_CODE
    ):
        return True
    if "consent.azure-apim.net" in str(exc):
        return True
    if hasattr(exc, "exceptions"):
        return any(_is_consent_error(sub) for sub in exc.exceptions)
    return False


def _extract_consent_url(exc: BaseException) -> str:
    """Walk nested exceptions and return the OAuth consent URL string.

    Args:
        exc: The exception containing the consent URL.

    Returns:
        The consent URL string, or ``str(exc)`` if no URL can be extracted.
    """
    error_data = getattr(exc, "error", None)
    if (
        error_data is not None
        and getattr(error_data, "code", None) == _CONSENT_ERROR_CODE
    ):
        return getattr(error_data, "message", str(exc))
    msg = str(exc)
    if "consent.azure-apim.net" in msg:
        return msg
    if hasattr(exc, "exceptions"):
        for sub in exc.exceptions:
            url = _extract_consent_url(sub)
            if url:
                return url
    return str(exc)


# ── Toolbox ────────────────────────────────────────────────────────────────────


@experimental()
class AzureAIProjectToolbox(BaseModel):
    """Load tools from an Azure AI Foundry Toolbox and use them via MCP.

    Azure AI Foundry Toolbox is a managed multi-MCP server that aggregates
    multiple configured tools behind a single MCP endpoint. This class wraps
    ``MultiServerMCPClient`` (from ``langchain-mcp-adapters``) and adds:

    - Azure Identity Bearer-token auth via ``get_bearer_token_provider``.
    - ``Foundry-Features`` header injection required by the Foundry MCP gateway.
    - Graceful OAuth consent-error handling: returns a fallback tool with the
      consent URL instead of raising, so the agent can surface it to the user.
    - Automatic tool-schema sanitization for MCP servers that emit incomplete
      JSON Schemas (missing ``properties`` on ``object`` types).
    - ``handle_tool_error = True`` on every tool so tool-call failures are
      returned as tool messages rather than propagating ``ToolException``.

    Each ``get_tools()`` call is **stateless** — it opens a fresh MCP session,
    loads tools, and returns, mirroring ``MultiServerMCPClient.get_tools()``.
    ``async with`` is supported as a convenience but does not change behavior.

    Primary usage::

        from langchain_azure_ai.tools import AzureAIProjectToolbox
        from langchain.agents import create_agent

        toolbox = AzureAIProjectToolbox(
            project_endpoint=(
                "https://<resource>.services.ai.azure.com/api/projects/<project>"
            ),
            toolbox_name="my-toolbox",
        )
        tools = await toolbox.get_tools()
        model = init_chat_model("azure_ai:gpt-5.4", credential=DefaultAzureCredential())
        agent = create_agent(
            model=model.bind_tools(tools),
            tools=tools
        )
        result = await agent.ainvoke({"messages": [HumanMessage("What can you do?")]})

    Using environment variables (zero-config in Foundry-hosted agents)::

        # Set in the environment / agent.manifest.yaml:
        #   FOUNDRY_PROJECT_ENDPOINT=https://<resource>.../api/projects/<project>
        #   FOUNDRY_AGENT_TOOLBOX_FEATURES=Toolboxes=V1Preview

        toolbox = AzureAIProjectToolbox(toolbox_name="my-toolbox")
        tools = await toolbox.get_tools()

    ``async with`` is also accepted (same behavior, returns self)::

        async with AzureAIProjectToolbox(
            project_endpoint=PROJECT_ENDPOINT, toolbox_name="my-toolbox"
        ) as toolbox:
            tools = await toolbox.get_tools()

    Note:
        Requires ``langchain-mcp-adapters`` and ``httpx``::

            pip install langchain-mcp-adapters httpx

    Args:
        project_endpoint: Azure AI Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
            Falls back to the ``AZURE_AI_PROJECT_ENDPOINT`` or
            ``FOUNDRY_PROJECT_ENDPOINT`` environment variables.
        toolbox_name: Name of the toolbox as configured in Azure AI Foundry.
            Falls back to the ``FOUNDRY_AGENT_TOOLBOX_NAME`` environment variable.
        api_version: Toolbox API version appended to the MCP URL.
            Defaults to ``"v1"``.
        credential: Azure credential used to obtain Bearer tokens. Accepts a
            plain string (static Bearer token), any ``TokenCredential`` such as
            ``DefaultAzureCredential`` or ``ManagedIdentityCredential``.
            Defaults to ``DefaultAzureCredential()``.
        features: Value for the ``Foundry-Features`` request header sent on
            every MCP request. Falls back to the
            ``FOUNDRY_AGENT_TOOLBOX_FEATURES`` environment variable, then to
            ``"Toolboxes=V1Preview"``.
    """

    project_endpoint: str = Field(default="")
    """Azure AI Foundry project endpoint URL."""

    toolbox_name: str = Field(default="")
    """Name of the toolbox as configured in Azure AI Foundry."""

    api_version: str = Field(default=_DEFAULT_API_VERSION)
    """Toolbox API version string appended to the MCP URL."""

    credential: Optional[Union[str, TokenCredential]] = Field(
        default=None, exclude=True
    )
    """Azure credential for Bearer-token authentication."""

    features: str = Field(default="")
    """Value for the ``Foundry-Features`` MCP gateway header."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _resolve_env_vars(cls, values: Any) -> Any:
        """Resolve fields from environment variables when not provided."""
        if isinstance(values, dict):
            if not values.get("project_endpoint"):
                values["project_endpoint"] = (
                    get_project_endpoint(values, nullable=True) or ""
                )
            if not values.get("toolbox_name"):
                values["toolbox_name"] = os.environ.get(
                    "FOUNDRY_AGENT_TOOLBOX_NAME", ""
                )
            if not values.get("features"):
                values["features"] = os.environ.get(
                    "FOUNDRY_AGENT_TOOLBOX_FEATURES", _DEFAULT_FEATURES
                )
        return values

    @property
    def toolbox_endpoint(self) -> str:
        """Compute the full MCP endpoint URL from project_endpoint + toolbox_name."""
        return _build_toolbox_mcp_url(
            self.project_endpoint, self.toolbox_name, self.api_version
        )

    def _build_mcp_client(self) -> Any:
        """Construct and return a ``MultiServerMCPClient`` for the toolbox endpoint."""
        try:
            import httpx
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as ex:
            raise ImportError(
                "AzureAIProjectToolbox requires 'langchain-mcp-adapters' and 'httpx'. "
                "Install them with:\n"
                "  pip install langchain-mcp-adapters httpx"
            ) from ex

        extra_headers = {_FEATURES_HEADER: self.features} if self.features else {}

        if isinstance(self.credential, str):
            # Static string credential — use as a pre-issued Bearer token.
            _static_token = self.credential

            class _StaticBearerAuth(httpx.Auth):
                def auth_flow(self, request: Any) -> Any:  # type: ignore[override]
                    request.headers["Authorization"] = f"Bearer {_static_token}"
                    yield request

            auth: httpx.Auth = _StaticBearerAuth()
        else:
            # TokenCredential (or default) — obtain a fresh token on each request.
            try:
                from azure.identity import (
                    DefaultAzureCredential,
                    get_bearer_token_provider,
                )
            except ImportError as ex:
                raise ImportError(
                    "AzureAIProjectToolbox requires 'azure-identity'. "
                    "Install it with:\n  pip install azure-identity"
                ) from ex

            credential = (
                self.credential
                if self.credential is not None
                else DefaultAzureCredential()
            )
            token_provider = get_bearer_token_provider(credential, _TOKEN_AUDIENCE)

            class _TokenBearerAuth(httpx.Auth):
                def __init__(self, _token_provider: Any) -> None:
                    self._get_token = _token_provider

                def auth_flow(self, request: Any) -> Any:  # type: ignore[override]
                    request.headers["Authorization"] = f"Bearer {self._get_token()}"
                    yield request

            auth = _TokenBearerAuth(token_provider)

        return MultiServerMCPClient(
            {
                "toolbox": {
                    "url": self.toolbox_endpoint,
                    "transport": "streamable_http",
                    "headers": extra_headers,
                    "auth": auth,
                }
            }
        )

    # ``async with`` is accepted for ergonomic compatibility but is a no-op:
    # MultiServerMCPClient.get_tools() manages its own session per call, so
    # there is no long-lived connection to open or close here.

    async def __aenter__(self) -> "AzureAIProjectToolbox":
        """Return self — no persistent connection to open."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """No-op — connections are managed per-call inside get_tools()."""

    async def get_tools(self) -> List[BaseTool]:
        """Fetch tools from the Azure AI Foundry Toolbox.

        Opens a fresh MCP session, loads all tools exposed by the toolbox,
        applies post-processing, and returns them. Each call is stateless,
        matching ``MultiServerMCPClient.get_tools()`` behavior.

        Returns:
            List of LangChain ``BaseTool`` instances ready for use with
            ``create_react_agent`` or any ``ToolNode``.

        Raises:
            ValueError: If ``project_endpoint`` or ``toolbox_name`` is not set.
        """
        if not self.project_endpoint:
            raise ValueError(
                "project_endpoint is required. Pass it as a constructor argument "
                "or set the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )
        if not self.toolbox_name:
            raise ValueError(
                "toolbox_name is required. Pass it as a constructor argument "
                "or set the FOUNDRY_AGENT_TOOLBOX_NAME environment variable."
            )
        client = self._build_mcp_client()
        return await self._fetch_tools(client)

    async def aget_tools(self) -> List[BaseTool]:
        """Async alias for ``get_tools()``.

        Provided for consistency with the LangChain async naming convention.

        Returns:
            List of LangChain ``BaseTool`` instances.
        """
        return await self.get_tools()

    async def _fetch_tools(self, client: Any) -> List[BaseTool]:
        """Post-process tools returned by the MCP client.

        Applies handle_tool_error, schema sanitization, and consent-error
        recovery before returning the final tool list.
        """
        from langchain_core.tools import tool as _tool

        try:
            tools: List[BaseTool] = await client.get_tools()
        except BaseException as exc:
            if _is_consent_error(exc):
                consent_url = _extract_consent_url(exc)
                logger.warning(
                    "OAuth consent required for toolbox at %s. "
                    "Visit the URL below to authorize, then restart the agent:\n\n"
                    "  %s\n",
                    self.toolbox_endpoint,
                    consent_url,
                )

                @_tool
                def oauth_consent_required(query: str) -> str:  # type: ignore[misc]
                    """Return instructions for completing the required OAuth consent."""
                    return (
                        "OAuth consent is required before this toolbox can be used. "
                        "Open the following URL in a browser to authorize access, "
                        f"then restart the agent:\n\n  {consent_url}"
                    )

                return [oauth_consent_required]
            raise

        # Ensure tool-call failures become tool messages rather than raising
        # ToolException, which would break conversation state when a tool_call
        # message has no corresponding tool_message response.
        for t in tools:
            t.handle_tool_error = True

        # Some MCP servers return tool schemas that omit ``properties`` on
        # object-typed inputs; fix them so the framework accepts the schema.
        for t in tools:
            schema = t.args_schema if isinstance(t.args_schema, dict) else None
            if schema is None:
                continue
            if schema.get("type") == "object" and "properties" not in schema:
                schema["properties"] = {}
            props: dict = schema.get("properties", {})
            required: List[str] = schema.get("required", [])
            if required and not props:
                for field_name in required:
                    props[field_name] = {"type": "string"}
                schema["properties"] = props

        logger.info(
            "Loaded %d tools from toolbox %s", len(tools), self.toolbox_endpoint
        )
        return tools
