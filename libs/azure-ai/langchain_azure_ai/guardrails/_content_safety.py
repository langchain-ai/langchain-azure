"""Azure AI Content Safety middleware for LangGraph agents."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from langgraph.graph import MessagesState
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class ContentSafetyViolationError(ValueError):
    """Raised when content safety violations are detected with ``action='block'``.

    Attributes:
        violations: List of detected violations, each a dict with ``category``
            and ``severity`` keys.
    """

    def __init__(self, message: str, violations: List[Dict[str, Any]]) -> None:
        """Create a ContentSafetyViolationError.

        Args:
            message: Human-readable description of the violation.
            violations: List of detected violations.
        """
        super().__init__(message)
        self.violations = violations


class _ContentSafetyState(MessagesState, total=False):
    """Extended state that carries content-safety violation results."""

    content_safety_violations: List[Dict[str, Any]]


class AzureContentSafetyMiddleware:
    """AgentMiddleware that screens messages with Azure AI Content Safety.

    Integrates with any LangGraph ``StateGraph`` via the
    :func:`~langchain_azure_ai.guardrails.apply_middleware` utility or directly
    with :meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`.

    The middleware analyses text content using the Azure AI Content Safety API
    and takes one of three actions when violations are detected:

    * ``"block"`` – raises :exc:`ContentSafetyViolationError`, halting the graph.
    * ``"warn"`` – logs a warning and lets execution continue unchanged.
    * ``"flag"`` – returns ``{"content_safety_violations": [...]}`` which is
      merged into the agent state so downstream nodes can inspect it.

    Both synchronous (``before_agent`` / ``after_agent``) and asynchronous
    (``abefore_agent`` / ``aafter_agent``) hooks are implemented.

    Args:
        endpoint: Azure Content Safety resource endpoint URL.  Falls back to
            the ``AZURE_CONTENT_SAFETY_ENDPOINT`` environment variable.
        credential: Azure credential.  Accepts a
            :class:`~azure.core.credentials.TokenCredential`,
            :class:`~azure.core.credentials.AzureKeyCredential`, or a plain
            API-key string.  Defaults to
            :class:`~azure.identity.DefaultAzureCredential` when ``None``.
        categories: Harm categories to analyse.  Valid values are ``"Hate"``,
            ``"SelfHarm"``, ``"Sexual"``, and ``"Violence"``.  Defaults to all
            four.
        severity_threshold: Minimum severity score (0–6) that triggers the
            configured action.  Defaults to ``4`` (medium).
        action: What to do when a violation is detected.  One of ``"block"``
            (default), ``"warn"``, or ``"flag"``.
        apply_to_input: Whether to screen the agent's input (last
            ``HumanMessage``).  Defaults to ``True``.
        apply_to_output: Whether to screen the agent's output (last
            ``AIMessage``).  Defaults to ``True``.
        blocklist_names: Names of custom blocklists configured in your Azure
            Content Safety resource.  Matches against these lists in addition to
            the built-in harm classifiers.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_content_safety"``.

    Example:
        .. code-block:: python

            from langchain_azure_ai.guardrails import (
                AzureContentSafetyMiddleware,
                apply_middleware,
            )
            from langgraph.graph import END, START, MessagesState, StateGraph

            safety = AzureContentSafetyMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            )

            builder = StateGraph(MessagesState)
            builder.add_node("agent", my_agent_fn)

            entry, after = apply_middleware(builder, [safety], agent_node="agent")
            builder.add_edge(START, entry)
            builder.add_edge("agent", after)

            graph = builder.compile()
    """

    #: State schema contributed by this middleware.  When ``action="flag"`` the
    #: merged graph state will include a ``content_safety_violations`` field.
    state_schema: type = _ContentSafetyState

    #: Extra LangGraph tools contributed by this middleware (none by default).
    tools: list = []

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        categories: Optional[List[str]] = None,
        severity_threshold: int = 4,
        action: Literal["block", "warn", "flag"] = "block",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        blocklist_names: Optional[List[str]] = None,
        name: str = "azure_content_safety",
    ) -> None:
        """Initialise the middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            categories: Harm categories to analyse.
            severity_threshold: Minimum severity score that triggers action.
            action: ``"block"``, ``"warn"``, or ``"flag"``.
            apply_to_input: Screen the last HumanMessage before agent runs.
            apply_to_output: Screen the last AIMessage after agent runs.
            blocklist_names: Custom blocklist names in your resource.
            name: Node-name prefix for LangGraph wiring.
        """
        try:
            import azure.ai.contentsafety  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'azure-ai-contentsafety' package is required to use "
                "AzureContentSafetyMiddleware.  Install it with:\n"
                "  pip install azure-ai-contentsafety\n"
                "or add the 'content_safety' extra:\n"
                "  pip install langchain-azure-ai[content_safety]"
            ) from exc

        resolved_endpoint = endpoint or os.environ.get(
            "AZURE_CONTENT_SAFETY_ENDPOINT"
        )
        if not resolved_endpoint:
            raise ValueError(
                "An endpoint is required.  Pass 'endpoint' or set the "
                "AZURE_CONTENT_SAFETY_ENDPOINT environment variable."
            )
        self._endpoint = resolved_endpoint

        if credential is None:
            from azure.identity import DefaultAzureCredential

            self._credential: Any = DefaultAzureCredential()
        elif isinstance(credential, str):
            from azure.core.credentials import AzureKeyCredential

            self._credential = AzureKeyCredential(credential)
        else:
            self._credential = credential

        self._categories: List[str] = categories or [
            "Hate",
            "SelfHarm",
            "Sexual",
            "Violence",
        ]
        self._severity_threshold = severity_threshold
        self.action = action
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self._blocklist_names: List[str] = blocklist_names or []
        self._name = name

        # Clients are created lazily on first use.
        self.__sync_client: Optional[Any] = None
        self.__async_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # AgentMiddleware protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Node-name prefix used for LangGraph wiring."""
        return self._name

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen the last HumanMessage before the agent runs.

        Args:
            state: Current LangGraph state dict (must contain a ``messages`` key).

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and violations
                are detected.
        """
        if not self.apply_to_input:
            return None
        text = self._extract_human_text(state.get("messages", []))
        if not text:
            return None
        violations = self._analyze_sync(text)
        return self._handle_violations(violations, "agent input")

    def after_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen the last AIMessage after the agent runs.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and violations
                are detected.
        """
        if not self.apply_to_output:
            return None
        text = self._extract_ai_text(state.get("messages", []))
        if not text:
            return None
        violations = self._analyze_sync(text)
        return self._handle_violations(violations, "agent output")

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(
        self, state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Async version of :meth:`before_agent`.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` or a state-patch dict.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and violations
                are detected.
        """
        if not self.apply_to_input:
            return None
        text = self._extract_human_text(state.get("messages", []))
        if not text:
            return None
        violations = await self._analyze_async(text)
        return self._handle_violations(violations, "agent input")

    async def aafter_agent(
        self, state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Async version of :meth:`after_agent`.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` or a state-patch dict.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and violations
                are detected.
        """
        if not self.apply_to_output:
            return None
        text = self._extract_ai_text(state.get("messages", []))
        if not text:
            return None
        violations = await self._analyze_async(text)
        return self._handle_violations(violations, "agent output")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> Any:
        """Return (creating if necessary) the synchronous ContentSafetyClient."""
        if self.__sync_client is None:
            from azure.ai.contentsafety import ContentSafetyClient

            self.__sync_client = ContentSafetyClient(
                self._endpoint, self._credential
            )
        return self.__sync_client

    def _get_async_client(self) -> Any:
        """Return (creating if necessary) the async ContentSafetyClient."""
        if self.__async_client is None:
            from azure.ai.contentsafety.aio import (
                ContentSafetyClient as AsyncContentSafetyClient,
            )

            self.__async_client = AsyncContentSafetyClient(
                self._endpoint, self._credential
            )
        return self.__async_client

    def _analyze_sync(self, text: str) -> List[Dict[str, Any]]:
        """Call the synchronous Content Safety API and return violations.

        Args:
            text: The text to analyse (truncated to 10 000 characters).

        Returns:
            List of violation dicts with ``category`` and ``severity`` keys.
        """
        from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

        options = AnalyzeTextOptions(
            text=text[:10000],
            categories=[TextCategory(c) for c in self._categories],
            blocklist_names=self._blocklist_names or None,
        )
        response = self._get_sync_client().analyze_text(options)
        return self._collect_violations(response)

    async def _analyze_async(self, text: str) -> List[Dict[str, Any]]:
        """Call the async Content Safety API and return violations.

        Args:
            text: The text to analyse (truncated to 10 000 characters).

        Returns:
            List of violation dicts with ``category`` and ``severity`` keys.
        """
        from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

        options = AnalyzeTextOptions(
            text=text[:10000],
            categories=[TextCategory(c) for c in self._categories],
            blocklist_names=self._blocklist_names or None,
        )
        response = await self._get_async_client().analyze_text(options)
        return self._collect_violations(response)

    def _collect_violations(self, response: Any) -> List[Dict[str, Any]]:
        """Extract violations from an AnalyzeTextResult.

        Args:
            response: The ``AnalyzeTextResult`` returned by the SDK.

        Returns:
            List of violation dicts.
        """
        violations: List[Dict[str, Any]] = []
        for cat in response.categories_analysis:
            if (
                cat.severity is not None
                and cat.severity >= self._severity_threshold
            ):
                violations.append(
                    {
                        "category": str(cat.category),
                        "severity": cat.severity,
                    }
                )
        if self._blocklist_names and getattr(response, "blocklists_match", None):
            for match in response.blocklists_match:
                violations.append(
                    {
                        "category": "blocklist",
                        "blocklist_name": match.blocklist_name,
                        "text": match.blocklist_item_text,
                    }
                )
        return violations

    def _handle_violations(
        self,
        violations: List[Dict[str, Any]],
        context: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply the configured action to detected violations.

        Args:
            violations: List of violation dicts (may be empty).
            context: Human-readable context label (e.g. ``"agent input"``).

        Returns:
            ``None`` when no violations or action is ``"warn"``.  A state-patch
            dict ``{"content_safety_violations": [...]}`` when action is
            ``"flag"``.

        Raises:
            ContentSafetyViolationError: When action is ``"block"`` and there
                are violations.
        """
        if not violations:
            return None

        if self.action == "block":
            categories = ", ".join(v["category"] for v in violations)
            raise ContentSafetyViolationError(
                f"Content safety violations detected in {context}: {categories}",
                violations,
            )
        if self.action == "warn":
            logger.warning(
                "Content safety violations in %s: %s",
                context,
                violations,
            )
            return None
        # action == "flag"
        return {"content_safety_violations": violations}

    @staticmethod
    def _extract_human_text(messages: list) -> Optional[str]:
        """Return text from the most recent HumanMessage, or ``None``.

        Args:
            messages: List of LangChain messages.

        Returns:
            Extracted text string, or ``None`` if no usable message found.
        """
        from langchain_core.messages import HumanMessage

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return AzureContentSafetyMiddleware._message_text(msg)
        return None

    @staticmethod
    def _extract_ai_text(messages: list) -> Optional[str]:
        """Return text from the most recent AIMessage, or ``None``.

        Args:
            messages: List of LangChain messages.

        Returns:
            Extracted text string, or ``None`` if no usable message found.
        """
        from langchain_core.messages import AIMessage

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return AzureContentSafetyMiddleware._message_text(msg)
        return None

    @staticmethod
    def _message_text(msg: Any) -> Optional[str]:
        """Extract plain text from a LangChain message's content.

        Args:
            msg: A LangChain message object.

        Returns:
            Combined text string, or ``None`` if no text found.
        """
        if isinstance(msg.content, str):
            return msg.content or None
        if isinstance(msg.content, list):
            parts = [
                block["text"]
                for block in msg.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = " ".join(parts)
            return text or None
        return None
