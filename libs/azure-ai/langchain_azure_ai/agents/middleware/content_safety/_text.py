"""Text content safety middleware for Azure AI Content Safety."""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, Sequence

from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from langchain.agents.middleware import AgentState, Runtime
from langchain_core.messages.content import NonStandardAnnotation

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.middleware.content_safety._base import (
    BlocklistEvaluation,
    ContentModerationEvaluation,
    ContentSafetyAnnotationPayload,
    ContentSafetyEvaluation,
    _AzureContentSafetyBaseMiddleware,
)

logger = logging.getLogger(__name__)


@experimental()
class AzureContentModerationMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that screens **text** messages with Azure AI Content Safety.

    Pass this class (or multiple instances) in the ``middleware`` parameter of
    any LangChain ``create_agent`` call:

    .. code-block:: python

        from langchain.agents import create_agent
        from langchain_azure_ai.agents.middleware import (
            AzureContentModerationMiddleware
        )

        agent = create_agent(
            model="azure_ai:gpt-4.1",
            middleware=[
                # Screen both input and output text for all harm categories
                AzureContentModerationMiddleware(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    exit_behavior="error",
                ),
            ],
        )

    You can compose multiple instances with different configurations:

    .. code-block:: python

        agent = create_agent(
            model="azure_ai:gpt-4.1",
            middleware=[
                # Raise on hate/violence on input only
                AzureContentModerationMiddleware(
                    categories=["Hate", "Violence"],
                    exit_behavior="error",
                    apply_to_input=True,
                    apply_to_output=False,
                    name="input_safety",
                ),
                # Replace self-harm content in model output and continue
                AzureContentModerationMiddleware(
                    categories=["SelfHarm"],
                    exit_behavior="continue",
                    apply_to_input=False,
                    apply_to_output=True,
                    name="output_safety",
                ),
            ],
        )

    The middleware analyses text content using the Azure AI Content Safety API
    and takes one of three actions when violations are detected:

    * ``"error"`` – raises :exc:`ContentSafetyViolationError`, halting the graph.
    * ``"replace"`` – replaces the offending message with a violation notice
      (either a service-derived description or a custom ``violation_message``)
      and lets execution proceed.
    * ``"continue"`` – ignores the violation and lets execution proceed by
      adding annotations to the message metadata with details of the violation(s).

    Both synchronous (``before_agent`` / ``after_agent``) and asynchronous
    (``abefore_agent`` / ``aafter_agent``) hooks are implemented.

    Args:
        endpoint: Azure Content Safety resource endpoint URL.  Falls back to
            the ``AZURE_CONTENT_SAFETY_ENDPOINT`` environment variable.
            Mutually exclusive with ``project_endpoint``.
        credential: Azure credential.  Accepts a
            :class:`~azure.core.credentials.TokenCredential`,
            :class:`~azure.core.credentials.AzureKeyCredential`, or a plain
            API-key string.  Defaults to
            :class:`~azure.identity.DefaultAzureCredential` when ``None``.
        project_endpoint: Azure AI Foundry project endpoint URL (e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``).
            Falls back to the ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
            Mutually exclusive with ``endpoint``.
        categories: Harm categories to analyse.  Valid values are ``"Hate"``,
            ``"SelfHarm"``, ``"Sexual"``, and ``"Violence"``.  Defaults to all
            four.
        severity_threshold: Minimum severity score (0–6) that triggers the
            configured exit behaviour.  Defaults to ``4`` (medium).
        exit_behavior: What to do when a violation is detected.  One of
            ``"error"`` (default), ``"continue"``, or ``"replace"``.
        violation_message: Custom text used to replace the offending message
            when ``exit_behavior="replace"``. Defaults to a message built
            from the service response.
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
        project_endpoint: Optional[str] = None,
        categories: Optional[
            List[Literal["Hate", "SelfHarm", "Sexual", "Violence"]]
        ] = None,
        severity_threshold: int = 4,
        exit_behavior: Literal["error", "continue", "replace"] = "error",
        violation_message: Optional[str] = None,
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        blocklist_names: Optional[List[str]] = None,
        name: str = "azure_content_safety",
    ) -> None:
        """Initialise the text content moderation middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            project_endpoint: Azure AI Foundry project endpoint URL.
                Mutually exclusive with ``endpoint``.
            categories: Harm categories to analyse.
            severity_threshold: Minimum severity score that triggers action.
            exit_behavior: ``"error"``, ``"continue"``, or ``"replace"``.
            violation_message: Custom replacement text for ``"replace"`` mode.
            apply_to_input: Screen the last HumanMessage before agent runs.
            apply_to_output: Screen the last AIMessage after agent runs.
            blocklist_names: Custom blocklist names in your resource.
            name: Node-name prefix for LangGraph wiring.
        """
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            project_endpoint=project_endpoint,
            exit_behavior=exit_behavior,
            violation_message=violation_message,
            apply_to_input=apply_to_input,
            apply_to_output=apply_to_output,
            name=name,
        )
        self._severity_threshold = severity_threshold
        self._categories: List[str] = (
            list(categories)
            if categories
            else [
                "Hate",
                "SelfHarm",
                "Sexual",
                "Violence",
            ]
        )
        self._blocklist_names: List[str] = blocklist_names or []

    # ------------------------------------------------------------------
    # Annotation / violation helpers
    # ------------------------------------------------------------------

    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` for text content safety evaluations."""
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="text_content_safety",
                violations=[v.to_dict() for v in evaluations],
            ).to_dict(),
        )

    def get_evaluation_response(self, response: Any) -> List[ContentSafetyEvaluation]:
        """Parse an ``AnalyzeTextResult`` into typed evaluation objects.

        Returns all category evaluations and any blocklist matches. Threshold
        filtering is NOT applied here — callers decide which evaluations
        constitute violations.
        """
        evaluations: List[ContentSafetyEvaluation] = []
        for cat in response.categories_analysis:
            evaluations.append(
                ContentModerationEvaluation(
                    category=str(cat.category),
                    severity=cat.severity,
                )
            )
        if self._blocklist_names and getattr(response, "blocklists_match", None):
            for match in response.blocklists_match:
                evaluations.append(
                    BlocklistEvaluation(
                        category="blocklist",
                        blocklist_name=match.blocklist_name,
                        text=match.blocklist_item_text,
                    )
                )
        return evaluations

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen the last HumanMessage before the agent runs."""
        if not self.apply_to_input:
            return None
        offending = self.get_human_message_from_state(state)
        text = self.get_text_from_message(offending)
        if not text:
            logger.debug("[%s] before_agent: no HumanMessage text found", self.name)
            return None
        logger.debug(
            "[%s] before_agent: screening input text (%d chars)", self.name, len(text)
        )
        violations = self._analyze_sync(text)
        return self._handle_violations(violations, "agent.input", offending)

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen the last AIMessage after the agent runs."""
        if not self.apply_to_output:
            return None
        offending = self.get_ai_message_from_state(state)
        text = self.get_text_from_message(offending)
        if not text:
            logger.debug("[%s] after_agent: no AIMessage text found", self.name)
            return None
        logger.debug(
            "[%s] after_agent: screening output text (%d chars)", self.name, len(text)
        )
        violations = self._analyze_sync(text)
        return self._handle_violations(violations, "agent.output", offending)

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_agent`."""
        if not self.apply_to_input:
            return None
        offending = self.get_human_message_from_state(state)
        text = self.get_text_from_message(offending)
        if not text:
            logger.debug("[%s] abefore_agent: no HumanMessage text found", self.name)
            return None
        logger.debug(
            "[%s] abefore_agent: screening input text (%d chars)", self.name, len(text)
        )
        violations = await self._analyze_async(text)
        return self._handle_violations(violations, "agent.input", offending)

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_agent`."""
        if not self.apply_to_output:
            return None
        offending = self.get_ai_message_from_state(state)
        text = self.get_text_from_message(offending)
        if not text:
            logger.debug("[%s] aafter_agent: no AIMessage text found", self.name)
            return None
        logger.debug(
            "[%s] aafter_agent: screening output text (%d chars)", self.name, len(text)
        )
        violations = await self._analyze_async(text)
        return self._handle_violations(violations, "agent.output", offending)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_sync(self, text: str) -> List[ContentSafetyEvaluation]:
        """Call the synchronous Content Safety API and return violations.

        Calls the SDK, parses the response into evaluations via
        :meth:`get_evaluation_response`, then filters by severity threshold.
        """
        options = AnalyzeTextOptions(
            text=text[:10000],
            categories=[TextCategory(c) for c in self._categories],
            blocklist_names=self._blocklist_names or None,
        )
        logger.debug(
            "[%s] Calling analyze_text (categories=%s, blocklists=%s)",
            self.name,
            self._categories,
            self._blocklist_names or [],
        )
        response = self._get_sync_client().analyze_text(options)
        evaluations = self.get_evaluation_response(response)
        return [
            e
            for e in evaluations
            if isinstance(e, BlocklistEvaluation)
            or (
                isinstance(e, ContentModerationEvaluation)
                and e.severity >= self._severity_threshold
            )
        ]

    async def _analyze_async(self, text: str) -> List[ContentSafetyEvaluation]:
        """Call the async Content Safety API and return violations.

        Calls the SDK, parses the response into evaluations via
        :meth:`get_evaluation_response`, then filters by severity threshold.
        """
        options = AnalyzeTextOptions(
            text=text[:10000],
            categories=[TextCategory(c) for c in self._categories],
            blocklist_names=self._blocklist_names or None,
        )
        logger.debug(
            "[%s] Calling async analyze_text (categories=%s, blocklists=%s)",
            self.name,
            self._categories,
            self._blocklist_names or [],
        )
        response = await self._get_async_client().analyze_text(options)
        evaluations = self.get_evaluation_response(response)
        return [
            e
            for e in evaluations
            if isinstance(e, BlocklistEvaluation)
            or (
                isinstance(e, ContentModerationEvaluation)
                and e.severity >= self._severity_threshold
            )
        ]
