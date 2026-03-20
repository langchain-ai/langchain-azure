"""Azure AI Content Safety middleware for LangGraph agents."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import MessagesState

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


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class _AzureContentSafetyBaseMiddleware:
    """Base class with shared credential, client, and violation-handling logic.

    This class centralises the functionality common to all Azure AI Content
    Safety middleware implementations:

    * **Credential resolution** – accepts a ``TokenCredential``,
      ``AzureKeyCredential``, or plain API-key string, defaulting to
      ``DefaultAzureCredential``.
    * **Lazy client construction** – both the synchronous
      ``ContentSafetyClient`` and its async counterpart are created on
      first use and reused across calls.
    * **Violation handling** – the ``_handle_violations`` method applies the
      ``"block"`` / ``"warn"`` / ``"flag"`` action uniformly regardless of
      which API endpoint detected the issue.
    * **Category violation parsing** – ``_collect_category_violations``
      extracts violations from ``AnalyzeTextResult`` or
      ``AnalyzeImageResult`` responses that expose a
      ``categories_analysis`` attribute.

    Concrete subclasses add the ``before_agent`` / ``after_agent`` hooks and
    call the appropriate Azure Content Safety API endpoint.
    """

    #: State schema contributed by this middleware.
    state_schema: type = _ContentSafetyState

    #: Extra LangGraph tools contributed by this middleware (none by default).
    tools: list = []

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        severity_threshold: int = 4,
        action: Literal["block", "warn", "flag"] = "block",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        name: str,
    ) -> None:
        """Initialise the base middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.  Falls back to
                the ``AZURE_CONTENT_SAFETY_ENDPOINT`` environment variable.
            credential: Azure credential (``TokenCredential``,
                ``AzureKeyCredential``, or API-key string).  Defaults to
                ``DefaultAzureCredential``.
            severity_threshold: Minimum severity score (0–6) that triggers the
                configured action.  Defaults to ``4`` (medium).
            action: ``"block"``, ``"warn"``, or ``"flag"``.
            apply_to_input: Screen the incoming message before the agent runs.
            apply_to_output: Screen the outgoing message after the agent runs.
            name: Node-name prefix for LangGraph wiring.
        """
        try:
            import azure.ai.contentsafety  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'azure-ai-contentsafety' package is required to use "
                f"{type(self).__name__}.  Install it with:\n"
                "  pip install azure-ai-contentsafety\n"
                "or add the 'content_safety' extra:\n"
                "  pip install langchain-azure-ai[content_safety]"
            ) from exc

        resolved_endpoint = endpoint or os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
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

        self._severity_threshold = severity_threshold
        self.action = action
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self._name = name

        # Clients are created lazily on first use.
        self.__sync_client: Optional[Any] = None
        self.__async_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # AgentMiddleware name protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Node-name prefix used for LangGraph wiring."""
        return self._name

    # ------------------------------------------------------------------
    # Client accessors (lazy construction)
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> Any:
        """Return (creating if necessary) the synchronous ContentSafetyClient."""
        if self.__sync_client is None:
            from azure.ai.contentsafety import ContentSafetyClient

            self.__sync_client = ContentSafetyClient(self._endpoint, self._credential)
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

    # ------------------------------------------------------------------
    # Shared violation handling
    # ------------------------------------------------------------------

    def _collect_category_violations(self, response: Any) -> List[Dict[str, Any]]:
        """Extract category violations from an Analyze*Result.

        Args:
            response: The ``AnalyzeTextResult`` or ``AnalyzeImageResult``
                returned by the SDK.

        Returns:
            List of violation dicts.
        """
        violations: List[Dict[str, Any]] = []
        for cat in response.categories_analysis:
            if cat.severity is not None and cat.severity >= self._severity_threshold:
                violations.append(
                    {
                        "category": str(cat.category),
                        "severity": cat.severity,
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


# ---------------------------------------------------------------------------
# Text content safety middleware
# ---------------------------------------------------------------------------


class AzureContentSafetyMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that screens **text** messages with Azure AI Content Safety.

    Pass this class (or multiple instances) in the ``middleware`` parameter of
    :meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`
    or any LangChain ``create_agent`` call:

    .. code-block:: python

        from langchain_azure_ai.agents.v2 import AgentServiceFactory
        from langchain_azure_ai.agents.middleware import AzureContentSafetyMiddleware

        factory = AgentServiceFactory(
            project_endpoint="https://my-project.api.azureml.ms/",
        )
        agent = factory.create_prompt_agent(
            model="gpt-4.1",
            middleware=[
                # Screen both input and output text for all harm categories
                AzureContentSafetyMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    action="block",
                ),
            ],
        )

    You can compose multiple instances with different configurations:

    .. code-block:: python

        agent = factory.create_prompt_agent(
            model="gpt-4.1",
            middleware=[
                # Block hate/violence on input only
                AzureContentSafetyMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    categories=["Hate", "Violence"],
                    action="block",
                    apply_to_input=True,
                    apply_to_output=False,
                    name="input_safety",
                ),
                # Flag (but don't block) self-harm in model output
                AzureContentSafetyMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    categories=["SelfHarm"],
                    action="flag",
                    apply_to_input=False,
                    apply_to_output=True,
                    name="output_safety",
                ),
            ],
        )

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
    """

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
        """Initialise the text content safety middleware.

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
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            severity_threshold=severity_threshold,
            action=action,
            apply_to_input=apply_to_input,
            apply_to_output=apply_to_output,
            name=name,
        )
        self._categories: List[str] = categories or [
            "Hate",
            "SelfHarm",
            "Sexual",
            "Violence",
        ]
        self._blocklist_names: List[str] = blocklist_names or []

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

    async def abefore_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    async def aafter_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        violations = self._collect_category_violations(response)
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
        violations = self._collect_category_violations(response)
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


# ---------------------------------------------------------------------------
# Image content safety middleware
# ---------------------------------------------------------------------------


class AzureContentSafetyImageMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that screens **image** content with Azure AI Content Safety.

    Use this middleware alongside :class:`AzureContentSafetyMiddleware` when
    your agent handles visual content.  Because image analysis uses a separate
    API endpoint (``analyze_image``) and different category enumerations, a
    dedicated class keeps each concern focused and composable.

    Pass this class in the ``middleware`` parameter of
    :meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`
    or any LangChain ``create_agent`` call:

    .. code-block:: python

        from langchain_azure_ai.agents.v2 import AgentServiceFactory
        from langchain_azure_ai.agents.middleware import (
            AzureContentSafetyMiddleware,
            AzureContentSafetyImageMiddleware,
        )

        factory = AgentServiceFactory(
            project_endpoint="https://my-project.api.azureml.ms/",
        )
        agent = factory.create_prompt_agent(
            model="gpt-4.1",
            middleware=[
                # Screen text content
                AzureContentSafetyMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    action="block",
                ),
                # Screen image content separately
                AzureContentSafetyImageMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    action="block",
                ),
            ],
        )

    The middleware extracts images from the most recent ``HumanMessage`` (input)
    and, optionally, from the most recent ``AIMessage`` (output).  It supports:

    * **Base64 data URLs** – ``data:image/png;base64,<data>``
    * **HTTP(S) URLs** – publicly accessible image URLs

    Content is analyzed using the Azure AI Content Safety image analysis API.
    When violations are detected, the middleware takes one of three actions:

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
        categories: Image harm categories to analyse.  Valid values are
            ``"Hate"``, ``"SelfHarm"``, ``"Sexual"``, and ``"Violence"``.
            Defaults to all four.
        severity_threshold: Minimum severity score (0–6) that triggers the
            configured action.  Defaults to ``4`` (medium).
        action: What to do when a violation is detected.  One of ``"block"``
            (default), ``"warn"``, or ``"flag"``.
        apply_to_input: Whether to screen images in the last ``HumanMessage``.
            Defaults to ``True``.
        apply_to_output: Whether to screen images in the last ``AIMessage``.
            Defaults to ``False`` (agents rarely produce images directly).
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_content_safety_image"``.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        categories: Optional[List[str]] = None,
        severity_threshold: int = 4,
        action: Literal["block", "warn", "flag"] = "block",
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        name: str = "azure_content_safety_image",
    ) -> None:
        """Initialise the image content safety middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            categories: Image harm categories to analyse.
            severity_threshold: Minimum severity score that triggers action.
            action: ``"block"``, ``"warn"``, or ``"flag"``.
            apply_to_input: Screen images in the last HumanMessage.
            apply_to_output: Screen images in the last AIMessage.
            name: Node-name prefix for LangGraph wiring.
        """
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            severity_threshold=severity_threshold,
            action=action,
            apply_to_input=apply_to_input,
            apply_to_output=apply_to_output,
            name=name,
        )
        self._categories: List[str] = categories or [
            "Hate",
            "SelfHarm",
            "Sexual",
            "Violence",
        ]

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen images in the last HumanMessage before the agent runs.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and violations
                are detected.
        """
        if not self.apply_to_input:
            return None
        messages = state.get("messages", [])
        images = self._extract_images_from_last_human(messages)
        return self._screen_images_sync(images, "agent input")

    def after_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen images in the last AIMessage after the agent runs.

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
        messages = state.get("messages", [])
        images = self._extract_images_from_last_ai(messages)
        return self._screen_images_sync(images, "agent output")

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        messages = state.get("messages", [])
        images = self._extract_images_from_last_human(messages)
        return await self._screen_images_async(images, "agent input")

    async def aafter_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        messages = state.get("messages", [])
        images = self._extract_images_from_last_ai(messages)
        return await self._screen_images_async(images, "agent output")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _screen_images_sync(
        self, images: List[Dict[str, Any]], context: str
    ) -> Optional[Dict[str, Any]]:
        """Analyse a list of images synchronously and handle violations.

        Args:
            images: List of image descriptor dicts from
                :meth:`_extract_images_from_last_human` /
                :meth:`_extract_images_from_last_ai`.
            context: Human-readable context label.

        Returns:
            ``None`` or a state-patch dict.
        """
        if not images:
            return None
        all_violations: List[Dict[str, Any]] = []
        for img in images:
            violations = self._analyze_image_sync(img)
            all_violations.extend(violations)
        return self._handle_violations(all_violations, context)

    async def _screen_images_async(
        self, images: List[Dict[str, Any]], context: str
    ) -> Optional[Dict[str, Any]]:
        """Analyse a list of images asynchronously and handle violations.

        Args:
            images: List of image descriptor dicts.
            context: Human-readable context label.

        Returns:
            ``None`` or a state-patch dict.
        """
        if not images:
            return None
        all_violations: List[Dict[str, Any]] = []
        for img in images:
            violations = await self._analyze_image_async(img)
            all_violations.extend(violations)
        return self._handle_violations(all_violations, context)

    def _analyze_image_sync(self, image: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call the synchronous image analysis API.

        Args:
            image: A dict with either ``"content"`` (bytes) or ``"url"`` (str).

        Returns:
            List of violation dicts.
        """
        from azure.ai.contentsafety.models import (
            AnalyzeImageOptions,
            ImageCategory,
            ImageData,
        )

        image_data = (
            ImageData(content=image["content"])
            if "content" in image
            else ImageData(url=image["url"])
        )
        options = AnalyzeImageOptions(
            image=image_data,
            categories=[ImageCategory(c) for c in self._categories],
        )
        response = self._get_sync_client().analyze_image(options)
        return self._collect_category_violations(response)

    async def _analyze_image_async(self, image: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call the asynchronous image analysis API.

        Args:
            image: A dict with either ``"content"`` (bytes) or ``"url"`` (str).

        Returns:
            List of violation dicts.
        """
        from azure.ai.contentsafety.models import (
            AnalyzeImageOptions,
            ImageCategory,
            ImageData,
        )

        image_data = (
            ImageData(content=image["content"])
            if "content" in image
            else ImageData(url=image["url"])
        )
        options = AnalyzeImageOptions(
            image=image_data,
            categories=[ImageCategory(c) for c in self._categories],
        )
        response = await self._get_async_client().analyze_image(options)
        return self._collect_category_violations(response)

    @staticmethod
    def _extract_images_from_last_human(
        messages: list,
    ) -> List[Dict[str, Any]]:
        """Extract images from the most recent HumanMessage.

        Args:
            messages: List of LangChain messages.

        Returns:
            List of image dicts (``{"content": bytes}`` or ``{"url": str}``).
        """
        from langchain_core.messages import HumanMessage

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return AzureContentSafetyImageMiddleware._images_from_message(msg)
        return []

    @staticmethod
    def _extract_images_from_last_ai(
        messages: list,
    ) -> List[Dict[str, Any]]:
        """Extract images from the most recent AIMessage.

        Args:
            messages: List of LangChain messages.

        Returns:
            List of image dicts (``{"content": bytes}`` or ``{"url": str}``).
        """
        from langchain_core.messages import AIMessage

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return AzureContentSafetyImageMiddleware._images_from_message(msg)
        return []

    @staticmethod
    def _images_from_message(msg: Any) -> List[Dict[str, Any]]:
        """Extract image descriptors from a LangChain message's content.

        Handles both the OpenAI vision format and LangChain's multi-modal
        content blocks.

        Supported block shapes:

        * ``{"type": "image_url", "image_url": "data:image/png;base64,<b64>"}``
        * ``{"type": "image_url", "image_url": "https://..."}``
        * ``{"type": "image_url", "image_url": {"url": "data:..." | "https://..."}}``

        Args:
            msg: A LangChain message object.

        Returns:
            List of image dicts with either ``"content"`` (bytes for base64
            images) or ``"url"`` (str for URL-based images).
        """
        images: List[Dict[str, Any]] = []
        if not isinstance(msg.content, list):
            return images

        for block in msg.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image_url":
                continue

            raw = block.get("image_url", "")
            # Normalise: may be a string or a dict {"url": ...}
            url_str: str = raw if isinstance(raw, str) else raw.get("url", "")
            if not url_str:
                continue

            if url_str.startswith("data:"):
                # Base64 data URL: data:<mime>;base64,<data>
                try:
                    _, rest = url_str.split(",", 1)
                    images.append({"content": base64.b64decode(rest)})
                except Exception:
                    logger.warning("Skipping malformed base64 image in message.")
            elif url_str.startswith(("http://", "https://")):
                images.append({"url": url_str})
            else:
                logger.warning(
                    "Skipping image with unsupported URL scheme: %s",
                    url_str[:40],
                )

        return images


# ---------------------------------------------------------------------------
# Protected material middleware
# ---------------------------------------------------------------------------


class AzureProtectedMaterialMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that detects protected material using Azure AI Content Safety.

    Protected material detection checks whether text contains copyright-protected
    content such as song lyrics, news articles, book passages, or other protected
    intellectual property.  Use this middleware to prevent agents from reproducing
    or accepting copyrighted content.

    Pass this class in the ``middleware`` parameter of
    :meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`
    or any LangChain ``create_agent`` call:

    .. code-block:: python

        from langchain_azure_ai.agents.v2 import AgentServiceFactory
        from langchain_azure_ai.agents.middleware import (
            AzureProtectedMaterialMiddleware
        )

        factory = AgentServiceFactory(
            project_endpoint="https://my-project.api.azureml.ms/",
        )
        agent = factory.create_prompt_agent(
            model="gpt-4.1",
            middleware=[
                AzureProtectedMaterialMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    action="block",
                ),
            ],
        )

    When protected material is detected, the middleware takes one of three
    actions:

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
        action: What to do when protected material is detected.  One of
            ``"block"`` (default), ``"warn"``, or ``"flag"``.
        apply_to_input: Whether to screen the agent's input (last
            ``HumanMessage``).  Defaults to ``True``.
        apply_to_output: Whether to screen the agent's output (last
            ``AIMessage``).  Defaults to ``True``.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_protected_material"``.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        action: Literal["block", "warn", "flag"] = "block",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        name: str = "azure_protected_material",
    ) -> None:
        """Initialise the protected material middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            action: ``"block"``, ``"warn"``, or ``"flag"``.
            apply_to_input: Screen the last HumanMessage before agent runs.
            apply_to_output: Screen the last AIMessage after agent runs.
            name: Node-name prefix for LangGraph wiring.
        """
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            severity_threshold=0,  # unused – API returns a boolean, not severity
            action=action,
            apply_to_input=apply_to_input,
            apply_to_output=apply_to_output,
            name=name,
        )

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen the last HumanMessage for protected material before the agent runs.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and protected
                material is detected.
        """
        if not self.apply_to_input:
            return None
        text = AzureContentSafetyMiddleware._extract_human_text(
            state.get("messages", [])
        )
        if not text:
            return None
        violations = self._detect_sync(text)
        return self._handle_violations(violations, "agent input")

    def after_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen the last AIMessage for protected material after the agent runs.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and protected
                material is detected.
        """
        if not self.apply_to_output:
            return None
        text = AzureContentSafetyMiddleware._extract_ai_text(state.get("messages", []))
        if not text:
            return None
        violations = self._detect_sync(text)
        return self._handle_violations(violations, "agent output")

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async version of :meth:`before_agent`.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` or a state-patch dict.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and protected
                material is detected.
        """
        if not self.apply_to_input:
            return None
        text = AzureContentSafetyMiddleware._extract_human_text(
            state.get("messages", [])
        )
        if not text:
            return None
        violations = await self._detect_async(text)
        return self._handle_violations(violations, "agent input")

    async def aafter_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async version of :meth:`after_agent`.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` or a state-patch dict.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and protected
                material is detected.
        """
        if not self.apply_to_output:
            return None
        text = AzureContentSafetyMiddleware._extract_ai_text(state.get("messages", []))
        if not text:
            return None
        violations = await self._detect_async(text)
        return self._handle_violations(violations, "agent output")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_sync(self, text: str) -> List[Dict[str, Any]]:
        """Call the synchronous protected material detection API.

        Args:
            text: The text to screen (truncated to 10 000 characters).

        Returns:
            List with one violation dict when protected material is detected,
            otherwise an empty list.
        """
        from azure.ai.contentsafety.models import DetectTextProtectedMaterialOptions

        options = DetectTextProtectedMaterialOptions(text=text[:10000])
        response = self._get_sync_client().detect_text_protected_material(options)
        return self._collect_protected_violations(response)

    async def _detect_async(self, text: str) -> List[Dict[str, Any]]:
        """Call the asynchronous protected material detection API.

        Args:
            text: The text to screen (truncated to 10 000 characters).

        Returns:
            List with one violation dict when protected material is detected,
            otherwise an empty list.
        """
        from azure.ai.contentsafety.models import DetectTextProtectedMaterialOptions

        options = DetectTextProtectedMaterialOptions(text=text[:10000])
        response = await self._get_async_client().detect_text_protected_material(
            options
        )
        return self._collect_protected_violations(response)

    @staticmethod
    def _collect_protected_violations(response: Any) -> List[Dict[str, Any]]:
        """Extract a violation entry from a DetectTextProtectedMaterialResult.

        Args:
            response: The ``DetectTextProtectedMaterialResult`` from the SDK.

        Returns:
            A single-element list when protected material is detected,
            otherwise an empty list.
        """
        analysis = getattr(response, "protected_material_analysis", None)
        if analysis and getattr(analysis, "detected", False):
            return [{"category": "ProtectedMaterial", "detected": True}]
        return []


# ---------------------------------------------------------------------------
# Prompt shield middleware
# ---------------------------------------------------------------------------


class AzurePromptShieldMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that detects prompt injection using Azure AI Content Safety.

    Prompt shield protects agents from adversarial inputs designed to hijack the
    agent's behavior.  Two types of injection are detected:

    * **Direct prompt injection** – malicious instructions in the user's own
      prompt (``user_prompt`` in the API).
    * **Indirect prompt injection** – malicious instructions embedded in external
      documents fed back to the agent (``documents`` in the API), such as web
      search results, retrieved knowledge-base chunks, or email bodies.

    The middleware extracts the last ``HumanMessage`` as the user prompt.  Any
    ``ToolMessage`` items in the state (tool/function outputs) are forwarded to
    the API as ``documents`` so indirect injection via tool results is also caught.

    Pass this class in the ``middleware`` parameter of
    :meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`
    or any LangChain ``create_agent`` call:

    .. code-block:: python

        from langchain_azure_ai.agents.v2 import AgentServiceFactory
        from langchain_azure_ai.agents.middleware import AzurePromptShieldMiddleware

        factory = AgentServiceFactory(
            project_endpoint="https://my-project.api.azureml.ms/",
        )
        agent = factory.create_prompt_agent(
            model="gpt-4.1",
            middleware=[
                AzurePromptShieldMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    action="block",
                ),
            ],
        )

    When an injection attack is detected, the middleware takes one of three
    actions:

    * ``"block"`` – raises :exc:`ContentSafetyViolationError`, halting the graph.
    * ``"warn"`` – logs a warning and lets execution continue unchanged.
    * ``"flag"`` – returns ``{"content_safety_violations": [...]}`` which is
      merged into the agent state so downstream nodes can inspect it.

    Note:
        ``apply_to_output`` defaults to ``False`` because prompt injection is
        an input-side attack.  Set it to ``True`` if you want to screen AI
        output as well.

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
        action: What to do when an injection is detected.  One of ``"block"``
            (default), ``"warn"``, or ``"flag"``.
        apply_to_input: Whether to screen the agent's input (last
            ``HumanMessage``).  Defaults to ``True``.
        apply_to_output: Whether to screen the agent's output (last
            ``AIMessage``).  Defaults to ``False``.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_prompt_shield"``.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        action: Literal["block", "warn", "flag"] = "block",
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        name: str = "azure_prompt_shield",
    ) -> None:
        """Initialise the prompt shield middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            action: ``"block"``, ``"warn"``, or ``"flag"``.
            apply_to_input: Screen the last HumanMessage before agent runs.
            apply_to_output: Screen the last AIMessage after agent runs.
            name: Node-name prefix for LangGraph wiring.
        """
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            severity_threshold=0,  # unused – API returns a boolean, not severity
            action=action,
            apply_to_input=apply_to_input,
            apply_to_output=apply_to_output,
            name=name,
        )

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen the last HumanMessage for prompt injection before the agent runs.

        Also screens any ``ToolMessage`` content in the state as documents to
        detect indirect injection through tool outputs.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and an
                injection attack is detected.
        """
        if not self.apply_to_input:
            return None
        messages = state.get("messages", [])
        user_prompt = AzureContentSafetyMiddleware._extract_human_text(messages)
        if not user_prompt:
            return None
        documents = self._extract_tool_texts(messages)
        violations = self._shield_sync(user_prompt=user_prompt, documents=documents)
        return self._handle_violations(violations, "agent input")

    def after_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Screen the last AIMessage for prompt injection after the agent runs.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` to continue unchanged, or a state-patch dict when
            ``action="flag"``.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and an
                injection attack is detected.
        """
        if not self.apply_to_output:
            return None
        text = AzureContentSafetyMiddleware._extract_ai_text(state.get("messages", []))
        if not text:
            return None
        violations = self._shield_sync(user_prompt=text, documents=[])
        return self._handle_violations(violations, "agent output")

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async version of :meth:`before_agent`.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` or a state-patch dict.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and an
                injection attack is detected.
        """
        if not self.apply_to_input:
            return None
        messages = state.get("messages", [])
        user_prompt = AzureContentSafetyMiddleware._extract_human_text(messages)
        if not user_prompt:
            return None
        documents = self._extract_tool_texts(messages)
        violations = await self._shield_async(
            user_prompt=user_prompt, documents=documents
        )
        return self._handle_violations(violations, "agent input")

    async def aafter_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async version of :meth:`after_agent`.

        Args:
            state: Current LangGraph state dict.

        Returns:
            ``None`` or a state-patch dict.

        Raises:
            ContentSafetyViolationError: When ``action="block"`` and an
                injection attack is detected.
        """
        if not self.apply_to_output:
            return None
        text = AzureContentSafetyMiddleware._extract_ai_text(state.get("messages", []))
        if not text:
            return None
        violations = await self._shield_async(user_prompt=text, documents=[])
        return self._handle_violations(violations, "agent output")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shield_sync(
        self, *, user_prompt: str, documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Call the synchronous prompt shield API.

        Args:
            user_prompt: The user's prompt text to screen for direct injection.
            documents: Grounding documents to screen for indirect injection.

        Returns:
            List of violation dicts describing detected injections.
        """
        from azure.ai.contentsafety.models import ShieldPromptOptions

        options = ShieldPromptOptions(
            user_prompt=user_prompt[:10000],
            documents=[d[:10000] for d in documents] or None,
        )
        response = self._get_sync_client().shield_prompt(options)
        return self._collect_injection_violations(response)

    async def _shield_async(
        self, *, user_prompt: str, documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Call the asynchronous prompt shield API.

        Args:
            user_prompt: The user's prompt text to screen for direct injection.
            documents: Grounding documents to screen for indirect injection.

        Returns:
            List of violation dicts describing detected injections.
        """
        from azure.ai.contentsafety.models import ShieldPromptOptions

        options = ShieldPromptOptions(
            user_prompt=user_prompt[:10000],
            documents=[d[:10000] for d in documents] or None,
        )
        response = await self._get_async_client().shield_prompt(options)
        return self._collect_injection_violations(response)

    @staticmethod
    def _collect_injection_violations(response: Any) -> List[Dict[str, Any]]:
        """Extract injection violation entries from a ShieldPromptResult.

        Args:
            response: The ``ShieldPromptResult`` returned by the SDK.

        Returns:
            List of violation dicts, each with ``category``, ``source``, and
            ``detected`` keys.
        """
        violations: List[Dict[str, Any]] = []
        prompt_analysis = getattr(response, "user_prompt_analysis", None)
        if prompt_analysis and getattr(prompt_analysis, "attack_detected", False):
            violations.append(
                {
                    "category": "PromptInjection",
                    "source": "user_prompt",
                    "detected": True,
                }
            )
        for i, doc_analysis in enumerate(
            getattr(response, "documents_analysis", None) or []
        ):
            if getattr(doc_analysis, "attack_detected", False):
                violations.append(
                    {
                        "category": "PromptInjection",
                        "source": f"document[{i}]",
                        "detected": True,
                    }
                )
        return violations

    @staticmethod
    def _extract_tool_texts(messages: list) -> List[str]:
        """Extract text content from all ToolMessage items in the message list.

        These represent tool/function call outputs, which can be a vector for
        indirect prompt injection when they contain external content.

        Args:
            messages: List of LangChain messages.

        Returns:
            List of non-empty text strings from ToolMessage items.
        """
        from langchain_core.messages import ToolMessage

        texts: List[str] = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            text = AzureContentSafetyMiddleware._message_text(msg)
            if text:
                texts.append(text)
        return texts
