"""Base middleware class for Azure AI Content Safety."""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.content import NonStandardAnnotation
from langgraph.graph import MessagesState

from langchain_azure_ai._resources import _get_base_url_from_endpoint

try:
    from azure.ai.contentsafety import ContentSafetyClient
    from azure.ai.contentsafety.aio import (
        ContentSafetyClient as AsyncContentSafetyClient,
    )
except ImportError:
    ContentSafetyClient = None  # type: ignore[assignment, misc]
    AsyncContentSafetyClient = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ContentSafetyViolationError(ValueError):
    """Raised when content safety violations with ``exit_behavior='error'``.

    Attributes:
        violations: List of typed violation instances.
    """

    def __init__(
        self, message: str, violations: Sequence[ContentSafetyEvaluation]
    ) -> None:
        super().__init__(message)
        self.violations = violations


# ---------------------------------------------------------------------------
# Violation data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentSafetyEvaluation:
    """Base class for all content-safety violations."""

    category: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the violation to a plain dict."""
        return asdict(self)


@dataclass(frozen=True)
class ContentModerationEvaluation(ContentSafetyEvaluation):
    """A harm-category evaluation from text or image content analysis."""

    severity: int = 0


@dataclass(frozen=True)
class BlocklistEvaluation(ContentSafetyEvaluation):
    """A blocklist-match evaluation from text content analysis."""

    blocklist_name: str = ""
    text: str = ""


@dataclass(frozen=True)
class ProtectedMaterialEvaluation(ContentSafetyEvaluation):
    """A protected-material evaluation."""

    detected: bool = True
    code_citations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PromptInjectionEvaluation(ContentSafetyEvaluation):
    """A prompt-injection evaluation."""

    source: str = ""
    detected: bool = True


@dataclass(frozen=True)
class GroundednessEvaluation(ContentSafetyEvaluation):
    """A groundedness evaluation."""

    category: Literal["Groundedness"] = "Groundedness"
    is_grounded: bool = True
    ungrounded_percentage: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotation payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentSafetyAnnotationPayload:
    """Value payload stored inside a ``NonStandardAnnotation`` for violations."""

    provider: str = "azure_content_safety"
    detection_type: str = ""
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the payload to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# LangGraph state schemas
# ---------------------------------------------------------------------------


class _ContentSafetyState(MessagesState, total=False):
    """Extended state that carries content-safety violation results."""

    content_safety_violations: List[Dict[str, Any]]


class _GroundednessState(MessagesState, total=False):
    """Extended state that carries groundedness evaluation annotations."""

    groundedness_evaluation: Dict[str, Any]


logger = logging.getLogger(__name__)


class _AzureContentSafetyBaseMiddleware(AgentMiddleware[AgentState[Any], Any]):
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
      ``"error"`` / ``"continue"`` exit behaviour uniformly regardless of
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
        project_endpoint: Optional[str] = None,
        exit_behavior: Literal["error", "continue", "replace"] = "error",
        violation_message: Optional[str] = None,
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        name: str,
    ) -> None:
        """Initialise the base middleware.

        Provide exactly one of ``endpoint`` or ``project_endpoint``.  When
        neither is given the middleware looks for the
        ``AZURE_CONTENT_SAFETY_ENDPOINT`` (mapped to ``endpoint``) and
        ``AZURE_AI_PROJECT_ENDPOINT`` (mapped to ``project_endpoint``)
        environment variables, in that order.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.  Falls back to
                the ``AZURE_CONTENT_SAFETY_ENDPOINT`` environment variable.
            credential: Azure credential (``TokenCredential``,
                ``AzureKeyCredential``, or API-key string).  Defaults to
                ``DefaultAzureCredential``.
            project_endpoint: Azure AI Foundry project endpoint URL
                (e.g. ``https://<resource>.services.ai.azure.com/api/projects/<project>``).
                Falls back to the ``AZURE_AI_PROJECT_ENDPOINT`` environment
                variable.  Mutually exclusive with ``endpoint``.
            exit_behavior: ``"error"`` (default) raises
                :exc:`ContentSafetyViolationError`; ``"replace"`` replaces the
                offending message with a violation notice and lets execution
                proceed; ``"continue"`` appends a new message with the violation
                notice and lets execution proceed.
            violation_message: Custom text used to replace the offending message
                when ``exit_behavior="continue"``.  When ``None`` (the default),
                a message is built from the service response.
            apply_to_input: Screen the incoming message before the agent runs.
            apply_to_output: Screen the outgoing message after the agent runs.
            name: Node-name prefix for LangGraph wiring.
        """
        # Check SDK availability at instantiation time so that mocking
        # sys.modules after the module has been imported still works.
        try:
            from azure.ai.contentsafety import (  # noqa: F401
                ContentSafetyClient as _CS,
            )
        except ImportError:
            raise ImportError(
                "The 'azure-ai-contentsafety' package is required to use "
                "Azure AI Content Safety middleware.  Install it with:\n"
                "  `pip install azure-ai-contentsafety`"
            )

        # Validate mutual exclusivity before falling back to env vars.
        if endpoint and project_endpoint:
            raise ValueError(
                "'endpoint' and 'project_endpoint' are mutually exclusive. "
                "Provide only one."
            )

        # Resolve from environment variables when not explicitly provided.
        if not endpoint and not project_endpoint:
            endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
        if not endpoint and not project_endpoint:
            project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")

        if not endpoint and not project_endpoint:
            raise ValueError(
                "An endpoint is required.  Pass 'endpoint' or "
                "'project_endpoint', or set the "
                "AZURE_CONTENT_SAFETY_ENDPOINT / AZURE_AI_PROJECT_ENDPOINT "
                "environment variable."
            )

        if project_endpoint:
            if "/api/projects/" not in project_endpoint:
                raise ValueError(
                    f"project_endpoint '{project_endpoint}' does not look like "
                    "a valid Azure AI Foundry project endpoint "
                    "(expected '.../api/projects/<project>')."
                )
            self._endpoint = _get_base_url_from_endpoint(project_endpoint)
        else:
            self._endpoint = endpoint  # type: ignore[assignment]

        if credential is None:
            self._credential: Any = DefaultAzureCredential()
        elif isinstance(credential, str):
            self._credential = AzureKeyCredential(credential)
        else:
            self._credential = credential

        self.exit_behavior = exit_behavior
        self._violation_message = violation_message
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
    # Annotations and evaluations handling
    # ------------------------------------------------------------------

    @abstractmethod
    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` from a list of typed evaluations."""
        pass

    @abstractmethod
    def get_evaluation_response(
        self, response: Any
    ) -> Sequence[ContentSafetyEvaluation]:
        """Extract typed evaluations from an Azure Content Safety API response."""
        pass

    def _handle_violations(
        self,
        evaluations: Sequence[ContentSafetyEvaluation],
        context: Literal["agent.input", "agent.output", "model.input", "model.output"],
        offending_message: Optional[BaseMessage] = None,
    ) -> dict[str, Any] | None:
        """Apply the configured exit behaviour to detected violations.

        Args:
            evaluations: List of typed evaluation instances (may be empty).
            context: Human-readable context label (e.g. ``"agent.input"``).
            offending_message: The original message that triggered the
                violation.  Used to build a replacement when
                ``exit_behavior="replace"``.

        Returns:
            ``None`` when no violations are found.  A state-patch dict
            containing a replacement message when ``exit_behavior="replace"``.

        Raises:
            ContentSafetyViolationError: When ``exit_behavior="error"`` and
                there are violations.
        """
        if not evaluations:
            logger.debug("[%s] No violations found in %s", self.name, context)
            return None

        categories = ", ".join(v.category for v in evaluations)
        logger.info(
            "[%s] %d violation(s) detected in %s for: %s",
            self.name,
            len(evaluations),
            context,
            categories,
        )

        if self.exit_behavior == "error":
            raise ContentSafetyViolationError(
                f"Content safety violations detected in {context}: {categories}",
                evaluations,
            )
        else:
            # exit_behavior="continue" or "replace"
            if offending_message is None:
                return None

            if self.exit_behavior == "replace":
                offending_message.content = (
                    self._violation_message or self._build_violation_text(evaluations)
                )
            else:
                annotation = self.get_annotation_from_evaluations(evaluations)
                sanitized_content = offending_message.content
                if isinstance(sanitized_content, str):
                    offending_message.content = [
                        {"type": "text", "text": sanitized_content},
                        dict(annotation),
                    ]
                else:
                    offending_message.content = list(sanitized_content) + [  # type: ignore[assignment]
                        dict(annotation)
                    ]

        return None

    @staticmethod
    def _build_violation_text(violations: Sequence[ContentSafetyEvaluation]) -> str:
        """Build a human-readable violation description from typed evaluations."""
        parts = []
        for v in violations:
            if isinstance(v, ContentModerationEvaluation):
                parts.append(f"{v.category} (severity: {v.severity})")
            else:
                parts.append(v.category)
        return "Content safety violation detected: " + ", ".join(parts)

    # ------------------------------------------------------------------
    # Client accessors (lazy construction)
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> ContentSafetyClient:
        """Return (creating if necessary) the synchronous ContentSafetyClient."""
        if self.__sync_client is None:
            self.__sync_client = ContentSafetyClient(self._endpoint, self._credential)
        return self.__sync_client

    def _get_async_client(self) -> AsyncContentSafetyClient:
        """Return (creating if necessary) the async ContentSafetyClient."""
        if self.__async_client is None:
            self.__async_client = AsyncContentSafetyClient(
                self._endpoint, self._credential
            )
        return self.__async_client

    # ------------------------------------------------------------------
    # REST request helpers
    # ------------------------------------------------------------------

    def _send_rest_sync(
        self,
        path: str,
        body: Dict[str, Any],
        api_version: str,
    ) -> Dict[str, Any]:
        """Send a synchronous REST request through the Content Safety client.

        Uses the client's authenticated pipeline so that credential handling
        (``TokenCredential`` or ``AzureKeyCredential``) is applied
        automatically.

        Args:
            path: The REST action path (e.g.
                ``"text:detectProtectedMaterial"``).
            body: JSON-serialisable request body.
            api_version: The API version to use for the request.

        Returns:
            Parsed JSON response as a dict.
        """
        from azure.core.rest import HttpRequest

        endpoint = self._endpoint.rstrip("/")
        url = f"{endpoint}/contentsafety/{path}"
        logger.debug("[%s] Sending sync REST request to %s", self.name, url)
        request = HttpRequest(
            method="POST",
            url=url,
            params={"api-version": api_version},
            json=body,
            headers={"Content-Type": "application/json"},
        )
        client = self._get_sync_client()
        response = client.send_request(request)
        response.raise_for_status()
        result = response.json()
        logger.debug("[%s] REST response received from %s", self.name, path)
        return result

    async def _send_rest_async(
        self,
        path: str,
        body: Dict[str, Any],
        api_version: str,
    ) -> Dict[str, Any]:
        """Send an asynchronous REST request through the Content Safety client.

        Args:
            path: The REST action path.
            body: JSON-serialisable request body.
            api_version: The API version to use for the request.

        Returns:
            Parsed JSON response as a dict.
        """
        from azure.core.rest import HttpRequest

        endpoint = self._endpoint.rstrip("/")
        url = f"{endpoint}/contentsafety/{path}"
        logger.debug("[%s] Sending async REST request to %s", self.name, url)
        request = HttpRequest(
            method="POST",
            url=url,
            params={"api-version": api_version},
            json=body,
            headers={"Content-Type": "application/json"},
        )
        client = self._get_async_client()
        response = await client.send_request(request)
        response.raise_for_status()
        result = response.json()
        logger.debug("[%s] REST response received from %s", self.name, path)
        return result

    # ------------------------------------------------------------------
    # Message text extraction helpers
    # ------------------------------------------------------------------

    def get_human_message_from_state(self, state: AgentState) -> Optional[HumanMessage]:
        """Extract text from the most recent HumanMessage in the state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg
        return None

    def get_ai_message_from_state(self, state: AgentState) -> Optional[AIMessage]:
        """Extract text from the most recent AIMessage in the state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    def get_text_from_message(self, msg: Optional[BaseMessage]) -> Optional[str]:
        """Extract plain text from a LangChain message's content.

        Args:
            msg: A LangChain message object, or ``None``.

        Returns:
            Combined text string, or ``None`` if no text found.
        """
        if msg is None:
            return None
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


def get_content_safety_annotations(
    msg: BaseMessage,
) -> List[ContentSafetyAnnotationPayload]:
    """Extract content-safety annotation payloads from a message.

    Inspects the message's ``content`` for ``non_standard_annotation``
    blocks whose ``value`` contains ``provider == "azure_content_safety"``.

    Args:
        msg: A LangChain message to inspect.

    Returns:
        List of :class:`ContentSafetyAnnotationPayload` instances found
        in the message.  Empty list if none are present.
    """
    annotations: List[ContentSafetyAnnotationPayload] = []
    content = msg.content
    if not isinstance(content, list):
        return annotations
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "non_standard_annotation":
            continue
        value = block.get("value")
        if not isinstance(value, dict):
            continue
        if value.get("provider") != "azure_content_safety":
            continue
        annotations.append(
            ContentSafetyAnnotationPayload(
                provider=value.get("provider", "azure_content_safety"),
                detection_type=value.get("detection_type", ""),
                violations=value.get("violations", []),
            )
        )
    return annotations


def print_content_safety_annotations(msg: BaseMessage) -> None:
    """Print a formatted summary of content-safety annotations on a message.

    Useful for troubleshooting and inspecting middleware results in
    notebooks or the console.  Extracts all ``non_standard`` content
    blocks produced by Azure Content Safety middleware and renders them
    in a human-readable, colour-free table format.

    Usage::

        result = agent.invoke({"messages": [HumanMessage(content="...")]})
        print_content_safety_annotations(
            result["messages"][-1]
        )

    Args:
        msg: A LangChain message to inspect.
    """
    annotations = get_content_safety_annotations(msg)
    if not annotations:
        print("No content-safety annotations found.")
        return

    _LABEL_MAP = {
        "text_content_safety": "Text Content Safety",
        "image_content_safety": "Image Content Safety",
        "protected_material": "Protected Material",
        "prompt_injection": "Prompt Injection",
        "groundedness": "Groundedness",
    }

    for i, annotation in enumerate(annotations, 1):
        label = _LABEL_MAP.get(annotation.detection_type, annotation.detection_type)
        header = f"[{i}] {label}"
        print(header)
        print("=" * len(header))

        if not annotation.violations:
            print("  No evaluations recorded.\n")
            continue

        for j, v in enumerate(annotation.violations, 1):
            category = v.get("category", "Unknown")
            print(f"\n  Evaluation #{j}: {category}")
            print(f"  {'-' * 30}")

            dt = annotation.detection_type

            if dt == "groundedness":
                is_grounded = v.get("is_grounded")
                pct = v.get("ungrounded_percentage", 0)
                status = "Grounded" if is_grounded else "UNGROUNDED"
                print(f"  Status           : {status}")
                print(f"  Ungrounded %     : {pct * 100:.1f}%")
                details = v.get("details", [])
                if details:
                    print(f"  Ungrounded spans : {len(details)}")
                    for k, d in enumerate(details, 1):
                        text = d.get("text", "")
                        preview = text[:80] + "..." if len(text) > 80 else text
                        print(f'    [{k}] "{preview}"')

            elif dt in ("text_content_safety", "image_content_safety"):
                severity = v.get("severity")
                if severity is not None:
                    print(f"  Severity         : {severity}/6")
                bl = v.get("blocklist_name")
                if bl:
                    print(f"  Blocklist        : {bl}")
                    print(f"  Matched text     : \"{v.get('text', '')}\"")

            elif dt == "protected_material":
                detected = v.get("detected", False)
                status = "DETECTED" if detected else "Not detected"
                print(f"  Status           : {status}")
                citations = v.get("codeCitations", [])
                if citations:
                    print(f"  Code citations   : {len(citations)}")
                    for k, cite in enumerate(citations, 1):
                        license_val = cite.get("license", "Unknown")
                        print(f"    [{k}] License: {license_val}")
                        for url in cite.get("sourceUrls", []):
                            print(f"        {url}")

            elif dt == "prompt_injection":
                detected = v.get("detected", False)
                source = v.get("source", "unknown")
                status = "DETECTED" if detected else "Not detected"
                print(f"  Source           : {source}")
                print(f"  Status           : {status}")

            else:
                # Generic fallback for unknown detection types.
                for key, val in v.items():
                    if key == "category":
                        continue
                    print(f"  {key:<17}: {val}")

        print()
