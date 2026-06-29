"""Azure AI Foundry Toolbox MCP integration for LangChain/LangGraph."""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, List, Optional, Type, Union
from urllib.parse import urlparse

from azure.core.credentials import TokenCredential
from langchain_core.documents.base import Blob
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai._user_agent import get_user_agent
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

_MCP_SERVER_NAME: str = "toolbox"
"""Internal server name used when registering the toolbox with the MCP client."""

_SKILL_SCHEME: str = "skill"
"""URI scheme used by Foundry toolbox skills (``skill://{name}``)."""

_SKILL_INDEX_NAME: str = "index.json"
"""Name of the special skills-index resource exposed by the Foundry toolbox.

The toolbox publishes a ``skill://index.json`` resource that aggregates every
skill's name and description for progressive discovery. It is a Foundry
extension, not a skill under the Agent Skills specification
(https://agentskills.io/specification), so it is excluded from ``get_skills``.
"""

_DEFAULT_SKILLS_BASE_PATH: str = "/skills/"
"""Default virtual base path under which skill files are placed for deepagents."""


_FOUNDRY_CALL_ID_HEADER: str = "x-agent-foundry-call-id"
"""The only platform identity header forwarded on outbound Foundry 1P calls.

Per the Foundry container outbound identity contract, the opaque per-request
call ID is the *sole* identity header stamped on outbound calls to Foundry 1P
services (Storage, Toolbox/MCP). The receiver resolves the caller's identity
server-side from this value. Other inbound platform headers (e.g.
``x-agent-user-id``) are consumed container-side only and MUST NOT be echoed
outbound — 1P services neither accept nor trust them.
"""


def _apply_platform_headers(request: Any) -> None:
    """Stamp the request-scoped Foundry per-request call ID on an outbound request.

    AgentServer binds the inbound platform context in a context variable. Reading
    it here keeps the header per-request, even when the toolbox object or MCP
    client was constructed before a hosted request began.

    Only ``x-agent-foundry-call-id`` is forwarded, matching the Foundry outbound
    identity contract: 1P services resolve the caller from the opaque call ID and
    do not accept other identity headers (notably ``x-agent-user-id``, which is
    container-side only). No-ops when AgentServer is unavailable (local
    development) or no call ID is present (container protocol version ``1.0.0``).
    """
    try:
        from azure.ai.agentserver.core import get_request_context
    except ImportError:
        return

    try:
        platform_headers = get_request_context().platform_headers()
    except Exception:
        logger.debug("Failed to resolve AgentServer platform headers", exc_info=True)
        return

    call_id = platform_headers.get(_FOUNDRY_CALL_ID_HEADER)
    if call_id and _FOUNDRY_CALL_ID_HEADER not in request.headers:
        request.headers[_FOUNDRY_CALL_ID_HEADER] = call_id


def _normalize_scheme(scheme: str) -> str:
    """Normalize a URI scheme for tolerant comparison.

    Lower-cases the scheme and removes a single trailing ``s`` so that callers
    can pass either the singular or plural form (e.g. both ``"skill"`` and
    ``"skills"`` match ``skill://`` resource URIs).

    Args:
        scheme: The raw scheme string to normalize.

    Returns:
        The normalized scheme string.
    """
    normalized = scheme.lower()
    return normalized[:-1] if normalized.endswith("s") else normalized


def _resource_name_from_uri(uri: str) -> str:
    """Extract the resource name from an MCP resource URI.

    For a URI such as ``skill://my-skill`` this returns ``"my-skill"`` — the
    portion following the ``://`` separator, with any trailing slash removed.

    Args:
        uri: The resource URI string.

    Returns:
        The resource name portion of the URI.
    """
    _, _, remainder = uri.partition("://")
    return (remainder or uri).rstrip("/")


def _run_async(coro: Any) -> Any:
    """Run a coroutine to completion from synchronous code.

    Uses :func:`asyncio.run` when no event loop is running. If an event loop is
    already running on the current thread, the coroutine is executed in a fresh
    loop on a separate worker thread to avoid ``RuntimeError``.

    Args:
        coro: The coroutine to run.

    Returns:
        The result returned by the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


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


async def _fetch_require_approval_tools(
    endpoint: str,
    auth: Any,
    extra_headers: dict[str, str],
) -> dict[str, str]:
    """Fetch tool approval configuration from the toolbox MCP endpoint.

    Returns a mapping of tool name to the ``require_approval`` value for
    tools where that field is present.
    """
    try:
        import httpx
    except ImportError as ex:
        raise ImportError(
            "AzureAIProjectToolbox requires 'httpx'. "
            "Install it with:\n  pip install httpx"
        ) from ex

    async with httpx.AsyncClient(auth=auth, headers=extra_headers, timeout=30.0) as hc:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        resp = await hc.post(endpoint, json=payload)
        resp.raise_for_status()

    return {
        t["name"]: t["_meta"]["tool_configuration"]["require_approval"]
        for t in resp.json().get("result", {}).get("tools", [])
        if t.get("_meta", {}).get("tool_configuration", {}).get("require_approval")
    }


# ── OAuth consent-error helpers ────────────────────────────────────────────────


def _has_consent_host(text: str) -> bool:
    """Return True when *text* contains a URL hosted on consent.azure-apim.net.

    URL-like tokens are parsed and checked by hostname instead of substring
    matching to avoid false positives from arbitrary string positions.
    """
    for token in re.findall(r"https?://[^\s'\"<>]+", text):
        host = urlparse(token).hostname
        if host and (
            host == "consent.azure-apim.net" or host.endswith(".consent.azure-apim.net")
        ):
            return True
    return False


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
    if _has_consent_host(str(exc)):
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
    if _has_consent_host(msg):
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

        from azure.identity import DefaultAzureCredential
        from langchain.chat_models import init_chat_model
        from langchain.agents import create_agent
        from langchain.messages import HumanMessage
        from langchain_azure_ai.tools import AzureAIProjectToolbox

        async def main():
            toolbox = AzureAIProjectToolbox(
                project_endpoint=(
                    "https://<resource>.services.ai.azure.com/api/projects/<project>"
                ),
                toolbox_name="my-toolbox",
            )
            tools = await toolbox.get_tools()
            model = init_chat_model("azure_ai:gpt-5.4")
            agent = create_agent(
                model=model.bind_tools(tools),
                tools=tools
            )
            return await agent.ainvoke({"messages": [HumanMessage("What can you do?")]})

    You can also rely on environment variables for configuration instead of passing
    constructor arguments::

        # Set in the environment / agent.manifest.yaml:
        # FOUNDRY_PROJECT_ENDPOINT=https://<resource>.../api/projects/<project>

        toolbox = AzureAIProjectToolbox(toolbox_name="my-toolbox")
        tools = await toolbox.get_tools()

    ``async with`` is also accepted (same behavior, returns self)::

        async with AzureAIProjectToolbox(toolbox_name="my-toolbox") as toolbox:
            tools = await toolbox.get_tools()

    Toolbox skills are exposed as MCP resources (URIs of the form
    ``skill://{name}``) and can be loaded as LangChain ``Blob`` objects with
    ``get_resources`` / ``aget_resources``::

        toolbox = AzureAIProjectToolbox(toolbox_name="my-toolbox")
        skill_blobs = toolbox.get_resources(scheme="skills")
        for blob in skill_blobs:
            backend.write(
                path=f"skills/{blob.source}/SKILL.md",
                content=blob.as_string(),
                encoding="utf-8",
            )

    For ``deepagents`` users, ``get_skills`` / ``aget_skills`` removes that
    boilerplate by returning a ready-to-use ``files`` mapping for
    ``create_deep_agent``::

        from deepagents import create_deep_agent
        from deepagents.backends import StateBackend

        toolbox = AzureAIProjectToolbox(toolbox_name="my-toolbox")
        skill_files = toolbox.get_skills()

        agent = create_deep_agent(
            model="azure_ai:gpt-5.2",
            backend=StateBackend(),
            skills=["/skills/"],
        )
        agent.invoke({"messages": [...], "files": skill_files})

    Note:
        Requires ``langchain-mcp-adapters`` and ``httpx``::

            pip install langchain-mcp-adapters httpx

    Args:
        project_endpoint: Azure AI Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
            Falls back to the ``AZURE_AI_PROJECT_ENDPOINT`` or
            ``FOUNDRY_PROJECT_ENDPOINT`` environment variables.
        toolbox_name: Name of the toolbox as configured in Azure AI Foundry.
            This parameter is required.
        api_version: Toolbox API version appended to the MCP URL.
            Defaults to ``"v1"``.
        credential: Azure credential used to obtain Bearer tokens. Accepts a
            plain string (static Bearer token), any ``TokenCredential`` such as
            ``DefaultAzureCredential`` or ``ManagedIdentityCredential``.
            Defaults to ``DefaultAzureCredential()``.
        extra_headers: Additional HTTP headers to include in MCP requests. The
            ``Foundry-Features`` header is automatically added with the default
            value unless already present in ``extra_headers``. Defaults to ``{}``.
    """

    project_endpoint: str = Field(default="")
    """Azure AI Foundry project endpoint URL."""

    toolbox_name: str
    """Name of the toolbox as configured in Azure AI Foundry."""

    api_version: str = Field(default="v1")
    """Toolbox API version string appended to the MCP URL."""

    credential: Optional[Union[str, TokenCredential]] = Field(
        default=None, exclude=True
    )
    """Azure credential for Bearer-token authentication."""

    extra_headers: dict[str, str] = Field(default_factory=dict)
    """Additional HTTP headers to include in MCP requests."""

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
            if not values.get("extra_headers"):
                values["extra_headers"] = {}
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
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as ex:
            raise ImportError(
                "AzureAIProjectToolbox requires 'langchain-mcp-adapters' and 'httpx'. "
                "Install them with:\n"
                "  pip install langchain-mcp-adapters httpx"
            ) from ex

        auth, extra_headers = self._build_auth_and_headers()

        return MultiServerMCPClient(
            {
                _MCP_SERVER_NAME: {
                    "url": self.toolbox_endpoint,
                    "transport": "streamable_http",
                    "headers": extra_headers,
                    "auth": auth,
                }
            }
        )

    def _build_auth_and_headers(self) -> tuple[Any, dict[str, str]]:
        """Build request auth and headers used for toolbox MCP calls."""
        try:
            import httpx
        except ImportError as ex:
            raise ImportError(
                "AzureAIProjectToolbox requires 'httpx'. "
                "Install it with:\n  pip install httpx"
            ) from ex

        # Start with user-provided extra headers and merge in
        # the default features header
        extra_headers = dict(self.extra_headers) if self.extra_headers else {}
        if _FEATURES_HEADER not in extra_headers:
            extra_headers[_FEATURES_HEADER] = _DEFAULT_FEATURES

        # Inject this package's UA token so toolbox/MCP traffic is attributable
        # in Foundry telemetry. The toolbox uses raw ``httpx`` (not azure-core),
        # so it doesn't pick up ``AZURE_HTTP_USER_AGENT`` automatically.
        # ``setdefault`` keeps any caller-provided override.
        ua = get_user_agent()
        if ua:
            extra_headers.setdefault("User-Agent", ua)

        if isinstance(self.credential, str):
            # Static string credential — use as a pre-issued Bearer token.
            _static_token = self.credential

            class _StaticBearerAuth(httpx.Auth):
                def auth_flow(self, request: Any) -> Any:  # type: ignore[override]
                    request.headers["Authorization"] = f"Bearer {_static_token}"
                    _apply_platform_headers(request)
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
                    _apply_platform_headers(request)
                    yield request

            auth = _TokenBearerAuth(token_provider)

        return auth, extra_headers

    def _validate_required_fields(self) -> None:
        """Validate required toolbox configuration fields."""
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
        self._validate_required_fields()
        client = self._build_mcp_client()
        return await self._fetch_tools(client)

    async def get_tools_requiring_approval(self) -> List[str]:
        """Return names of toolbox tools that require runtime approval.

        This inspects the toolbox ``tools/list`` metadata and returns tool names
        whose ``_meta.tool_configuration.require_approval`` value is ``"always"``.
        This capability is independent from OAuth consent handling.

        Returns:
            List of tool names that require approval before execution.

        Raises:
            ValueError: If ``project_endpoint`` or ``toolbox_name`` is not set.
        """
        self._validate_required_fields()
        auth, extra_headers = self._build_auth_and_headers()
        approval_map = await _fetch_require_approval_tools(
            self.toolbox_endpoint,
            auth,
            extra_headers,
        )
        return [name for name, val in approval_map.items() if val == "always"]

    async def aget_tools(self) -> List[BaseTool]:
        """Async alias for ``get_tools()``.

        Provided for consistency with the LangChain async naming convention.

        Returns:
            List of LangChain ``BaseTool`` instances.
        """
        return await self.get_tools()

    async def aget_resources(
        self,
        scheme: Optional[str] = None,
        *,
        uris: Optional[Union[str, List[str]]] = None,
    ) -> List[Blob]:
        """Fetch resources exposed by the Azure AI Foundry Toolbox.

        Toolbox skills are surfaced as MCP resources with URIs of the form
        ``skill://{name}``. This opens a fresh MCP session, lists the available
        resources, optionally filters them by URI ``scheme``, reads their
        contents, and returns them as LangChain ``Blob`` objects.

        Each returned ``Blob`` carries the resource name in its ``source``
        property (derived from the URI, e.g. ``skill://my-skill`` becomes
        ``"my-skill"``) and its raw URI under ``metadata["uri"]``.

        Args:
            scheme: Optional URI scheme used to filter resources, e.g.
                ``"skills"`` to load only ``skill://`` resources. The match is
                case-insensitive and tolerant of singular/plural forms. When
                ``None``, all resources are returned. Ignored when ``uris`` is
                provided.
            uris: Optional resource URI or list of URIs to load explicitly. When
                provided, ``scheme`` filtering is skipped.

        Returns:
            List of LangChain ``Blob`` instances, one per resource content.

        Raises:
            ValueError: If ``project_endpoint`` or ``toolbox_name`` is not set.
        """
        self._validate_required_fields()

        try:
            from langchain_mcp_adapters.resources import load_mcp_resources
        except ImportError as ex:
            raise ImportError(
                "AzureAIProjectToolbox requires 'langchain-mcp-adapters' and 'httpx'. "
                "Install them with:\n"
                "  pip install langchain-mcp-adapters httpx"
            ) from ex

        client = self._build_mcp_client()
        async with client.session(_MCP_SERVER_NAME) as session:
            if uris is not None:
                uri_list = [uris] if isinstance(uris, str) else list(uris)
            else:
                listed = await session.list_resources()
                uri_list = [str(resource.uri) for resource in listed.resources]
                if scheme is not None:
                    target = _normalize_scheme(scheme)
                    uri_list = [
                        uri
                        for uri in uri_list
                        if _normalize_scheme(urlparse(uri).scheme) == target
                    ]

            blobs: List[Blob] = await load_mcp_resources(session, uris=uri_list)

        for blob in blobs:
            uri = blob.metadata.get("uri") if blob.metadata else None
            if not uri:
                continue
            if blob.metadata is None:
                blob.metadata = {}
            blob.metadata.setdefault("source", _resource_name_from_uri(str(uri)))

        logger.info(
            "Loaded %d resource(s) from toolbox %s",
            len(blobs),
            self.toolbox_endpoint,
        )
        return blobs

    def get_resources(
        self,
        scheme: Optional[str] = None,
        *,
        uris: Optional[Union[str, List[str]]] = None,
    ) -> List[Blob]:
        """Fetch resources exposed by the Azure AI Foundry Toolbox.

        Synchronous wrapper around :meth:`aget_resources`. See that method for
        full details on ``scheme`` filtering and the returned ``Blob`` objects.

        Args:
            scheme: Optional URI scheme used to filter resources, e.g.
                ``"skills"`` to load only ``skill://`` resources.
            uris: Optional resource URI or list of URIs to load explicitly.

        Returns:
            List of LangChain ``Blob`` instances, one per resource content.

        Raises:
            ValueError: If ``project_endpoint`` or ``toolbox_name`` is not set.
        """
        return _run_async(self.aget_resources(scheme, uris=uris))

    async def aget_skills(
        self,
        *,
        base_path: str = _DEFAULT_SKILLS_BASE_PATH,
        backend: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Load toolbox skills as a ``deepagents``-ready file mapping.

        This is an opinionated convenience built on top of
        :meth:`aget_resources`. It fetches the toolbox skills (MCP resources
        with ``skill://`` URIs) and returns a mapping of virtual file paths to
        ``deepagents`` ``FileData`` objects, laid out under ``base_path`` in the
        directory structure that ``create_deep_agent`` expects::

            {f"{base_path}{skill_name}/SKILL.md": <FileData>, ...}

        How the files reach the agent depends on the backend:

        - **StateBackend** (default): leave ``backend`` as ``None`` and pass the
          returned mapping as the ``files`` payload on ``invoke``. State writes
          go through the LangGraph runtime, so the backend cannot be seeded
          standalone.
        - **FilesystemBackend / StoreBackend / sandbox backends**: pass the
          backend as ``backend``. The skills are written into it via
          ``aupload_files`` and the same mapping is also returned.

        Seeding a ``StateBackend`` (default)::

            from deepagents import create_deep_agent
            from deepagents.backends import StateBackend
            from langchain_azure_ai.tools import AzureAIProjectToolbox

            toolbox = AzureAIProjectToolbox(toolbox_name="my-tools")
            skill_files = await toolbox.aget_skills()

            agent = create_deep_agent(
                model="azure_ai:gpt-5.2",
                backend=StateBackend(),
                skills=[base_path],
            )
            await agent.ainvoke(
                {"messages": [...], "files": skill_files}
            )

        Seeding any other backend (e.g. ``FilesystemBackend``)::

            from deepagents.backends import FilesystemBackend

            backend = FilesystemBackend(root_dir="./my-project")
            toolbox = AzureAIProjectToolbox(toolbox_name="my-tools")
            await toolbox.aget_skills(backend=backend)

            agent = create_deep_agent(
                model="azure_ai:gpt-5.2",
                backend=backend,
                skills=["/skills/"],
            )

        Args:
            base_path: Virtual directory under which skill files are placed.
                Must start and end with ``"/"``. Defaults to ``"/skills/"``.
                Pass the same value (or a list containing it) as the ``skills``
                argument to ``create_deep_agent``.
            backend: Optional ``deepagents`` backend to write the skills into via
                ``aupload_files``. Intended for backends with standalone storage
                (``FilesystemBackend``, ``StoreBackend``, sandbox backends).
                When ``None`` (default), files are only returned, for use with
                ``StateBackend`` seeding via ``invoke(files=...)``.

        Returns:
            A mapping of virtual ``SKILL.md`` paths to ``deepagents``
            ``FileData`` objects.

        Raises:
            ValueError: If ``base_path`` does not start and end with ``"/"``, or
                if ``project_endpoint`` or ``toolbox_name`` is not set.
            ImportError: If ``deepagents`` is not installed.
        """
        if not (base_path.startswith("/") and base_path.endswith("/")):
            raise ValueError(
                "base_path must start and end with '/', " f"got {base_path!r}."
            )

        try:
            from deepagents.backends.utils import create_file_data
        except ImportError as ex:
            raise ImportError(
                "AzureAIProjectToolbox.get_skills() requires 'deepagents'. "
                "Install it with:\n  pip install deepagents"
            ) from ex

        blobs = await self.aget_resources(scheme=_SKILL_SCHEME)

        skill_files: dict[str, Any] = {}
        uploads: List[tuple[str, bytes]] = []
        for blob in blobs:
            name = blob.source
            if not name:
                continue
            # Skip the toolbox's skills index (``skill://index.json``): it is a
            # discovery aid, not a skill under the Agent Skills specification.
            if name == _SKILL_INDEX_NAME:
                continue
            # Foundry surfaces each skill's resource name as its full path
            # (e.g. ``jokes-teller/SKILL.md``), so place it directly under
            # ``base_path``.
            path = f"{base_path}{name.strip('/')}"
            content = blob.as_string()
            skill_files[path] = create_file_data(content)
            uploads.append((path, content.encode("utf-8")))

        if backend is not None:
            await backend.aupload_files(uploads)

        logger.info(
            "Prepared %d skill file(s) from toolbox %s under %s",
            len(skill_files),
            self.toolbox_endpoint,
            base_path,
        )
        return skill_files

    def get_skills(
        self,
        *,
        base_path: str = _DEFAULT_SKILLS_BASE_PATH,
        backend: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Load toolbox skills as a ``deepagents``-ready file mapping.

        Synchronous wrapper around :meth:`aget_skills`. See that method for full
        details on ``base_path``, ``backend``, and the returned mapping.

        Args:
            base_path: Virtual directory under which skill files are placed.
                Must start and end with ``"/"``. Defaults to ``"/skills/"``.
            backend: Optional ``deepagents`` backend to write the skills into via
                ``aupload_files``. When ``None`` (default), files are only
                returned, for use with ``StateBackend`` seeding via
                ``invoke(files=...)``.

        Returns:
            A mapping of virtual ``SKILL.md`` paths to ``deepagents``
            ``FileData`` objects.

        Raises:
            ValueError: If ``base_path`` does not start and end with ``"/"``, or
                if ``project_endpoint`` or ``toolbox_name`` is not set.
            ImportError: If ``deepagents`` is not installed.
        """
        return _run_async(self.aget_skills(base_path=base_path, backend=backend))

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
