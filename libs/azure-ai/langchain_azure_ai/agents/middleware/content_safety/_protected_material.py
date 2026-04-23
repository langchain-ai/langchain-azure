"""Protected material detection middleware for Azure AI Content Safety."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from langchain.agents.middleware import AgentState, Runtime
from langchain_core.messages.content import NonStandardAnnotation

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.middleware.content_safety._base import (
    ContentSafetyAnnotationPayload,
    ContentSafetyEvaluation,
    _AzureContentSafetyBaseMiddleware,
)
from langchain_azure_ai.agents.middleware.content_safety._text import (
    TextModerationInput,
)


@dataclass(frozen=True)
class ProtectedMaterialEvaluation(ContentSafetyEvaluation):
    """A protected-material evaluation."""

    detected: bool = True
    code_citations: List[Dict[str, Any]] = field(default_factory=list)


logger = logging.getLogger(__name__)

_PROTECTED_MATERIAL_API_VERSION = "2024-09-15-preview"


@experimental()
class AzureProtectedMaterialMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that detects protected material using Azure AI Content Safety.

    Protected material detection checks whether text contains copyright-protected
    content such as song lyrics, news articles, book passages, or other protected
    intellectual property.  Use this middleware to prevent agents from reproducing
    or accepting copyrighted content.

    When protected material is detected, the middleware takes one of two
    actions:

    * ``"error"`` – raises :exc:`ContentSafetyViolationError`, halting the graph.
    * ``"continue"`` – replaces the offending message with a violation notice
      (either a service-derived description or a custom ``violation_message``)
      and lets execution proceed.

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
        exit_behavior: What to do when protected material is detected.  One of
            ``"error"`` (default) or ``"continue"``.
        violation_message: Custom text used to replace the offending message
            when ``exit_behavior="continue"``.  Defaults to a message built
            from the service response.
        apply_to_input: Whether to screen the agent's input (last
            ``HumanMessage``).  Defaults to ``True``.
        apply_to_output: Whether to screen the agent's output (last
            ``AIMessage``).  Defaults to ``True``.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_protected_material"``.
        context_extractor: Optional callable with signature
            ``(state, runtime) -> Optional[TextModerationInput]``
            that receives the current graph state and the LangGraph
            :class:`~langchain.agents.middleware.Runtime` execution context,
            and returns the text to screen, or ``None`` to skip evaluation
            entirely.  When ``None`` (default) the middleware uses its
            built-in extraction logic.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        project_endpoint: Optional[str] = None,
        type: Literal["code", "text"] = "text",
        exit_behavior: Literal["error", "continue", "replace"] = "error",
        violation_message: Optional[str] = None,
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        name: str = "azure_protected_material",
        context_extractor: Optional[
            Callable[[AgentState[Any], Runtime[Any]], Optional[TextModerationInput]]
        ] = None,
    ) -> None:
        """Initialise the protected material middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            project_endpoint: Azure AI Foundry project endpoint URL.
                Mutually exclusive with ``endpoint``.
            type: The type of content to screen, either ``"code"`` or ``"text"``.
            exit_behavior: The behavior when protected material is detected, either
                 ``"error"``, ``"continue"``, or ``"replace"``.
                 When set to ``"error"``, a
                 :class:`ContentSafetyViolationError` is raised.
                 When set to ``"continue"``,

                 the offending message is replaced with a violation notice and execution
                continues. When set to ``"replace"``, the offending message is replaced
                with the text specified in ``violation_message``.
            violation_message: Custom replacement text for ``"replace"`` mode.
            apply_to_input: Screen the last HumanMessage before agent runs.
            apply_to_output: Screen the last AIMessage after agent runs.
            name: Node-name prefix for LangGraph wiring.
            context_extractor: Optional callable that extracts the text to
                screen from the agent state and the LangGraph execution context
                instead of using the built-in heuristics.
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
        self.type = type
        self._context_extractor = context_extractor

    # ------------------------------------------------------------------
    # Annotation / violation helpers
    # ------------------------------------------------------------------

    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` for protected material evaluations."""
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="protected_material",
                violations=[v.to_dict() for v in evaluations],
            ).to_dict(),
        )

    def get_evaluation_response(
        self, response: Dict[str, Any]
    ) -> List[ProtectedMaterialEvaluation]:
        """Parse a ``detectProtectedMaterial`` REST response into evaluations."""
        analysis = response.get("protectedMaterialAnalysis", {})
        return [
            ProtectedMaterialEvaluation(
                category="ProtectedMaterial",
                detected=analysis.get("detected", False),
                code_citations=analysis.get("codeCitations", []),
            )
        ]

    # ------------------------------------------------------------------
    # Input extraction
    # ------------------------------------------------------------------

    def _extract_text_input(
        self, state: AgentState[Any], runtime: Runtime[Any], *, is_input: bool
    ) -> Optional[TextModerationInput]:
        """Extract the text to screen from the agent state.

        When a ``context_extractor`` was provided at construction time it is
        called with the current state and the LangGraph execution context.
        Otherwise the default heuristics are used: the last ``HumanMessage``
        text for input screening and the last ``AIMessage`` text for output
        screening.

        Args:
            state: Current LangGraph state dict.
            runtime: The LangGraph execution context.
            is_input: ``True`` when screening agent input (``before_agent``),
                ``False`` when screening agent output (``after_agent``).

        Returns:
            A :class:`TextModerationInput` instance, or ``None`` to skip
            evaluation.
        """
        if self._context_extractor is not None:
            return self._context_extractor(state, runtime)

        msg = (
            self.get_human_message_from_state(state)
            if is_input
            else self.get_ai_message_from_state(state)
        )
        text = self.get_text_from_message(msg)
        return TextModerationInput(text=text) if text else None

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen the last HumanMessage for protected material before the agent runs."""
        if not self.apply_to_input:
            return None
        inputs = self._extract_text_input(state, runtime, is_input=True)
        if inputs is None:
            logger.debug("[%s] before_agent: no HumanMessage text found", self.name)
            return None
        operation = (
            "text:detectProtectedMaterial"
            if self.type == "text"
            else "text:detectProtectedMaterialForCode"
        )
        logger.debug(
            "[%s] before_agent: screening input %s (%d chars)",
            self.name,
            self.type,
            len(inputs.text),
        )
        offending = self.get_human_message_from_state(state)
        violations = self._detect_sync(operation, inputs.text)
        return self._handle_violations(violations, "agent.input", offending)

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen the last AIMessage for protected material after the agent runs."""
        if not self.apply_to_output:
            return None
        inputs = self._extract_text_input(state, runtime, is_input=False)
        if inputs is None:
            logger.debug("[%s] after_agent: no AIMessage text found", self.name)
            return None
        operation = (
            "text:detectProtectedMaterial"
            if self.type == "text"
            else "text:detectProtectedMaterialForCode"
        )
        logger.debug(
            "[%s] after_agent: screening output %s (%d chars)",
            self.name,
            self.type,
            len(inputs.text),
        )
        offending = self.get_ai_message_from_state(state)
        violations = self._detect_sync(operation, inputs.text)
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
        inputs = self._extract_text_input(state, runtime, is_input=True)
        if inputs is None:
            logger.debug("[%s] abefore_agent: no HumanMessage text found", self.name)
            return None
        operation = (
            "text:detectProtectedMaterial"
            if self.type == "text"
            else "text:detectProtectedMaterialForCode"
        )
        logger.debug(
            "[%s] abefore_agent: screening input %s (%d chars)",
            self.name,
            self.type,
            len(inputs.text),
        )
        offending = self.get_human_message_from_state(state)
        violations = await self._detect_async(operation, inputs.text)
        return self._handle_violations(violations, "agent.input", offending)

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_agent`."""
        if not self.apply_to_output:
            return None
        inputs = self._extract_text_input(state, runtime, is_input=False)
        if inputs is None:
            logger.debug("[%s] aafter_agent: no AIMessage text found", self.name)
            return None
        operation = (
            "text:detectProtectedMaterial"
            if self.type == "text"
            else "text:detectProtectedMaterialForCode"
        )
        logger.debug(
            "[%s] aafter_agent: screening output %s (%d chars)",
            self.name,
            self.type,
            len(inputs.text),
        )
        offending = self.get_ai_message_from_state(state)
        violations = await self._detect_async(operation, inputs.text)
        return self._handle_violations(violations, "agent.output", offending)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_sync(
        self, operation: str, text: str
    ) -> List[ProtectedMaterialEvaluation]:
        """Call the synchronous protected material detection REST API."""
        result = self._send_rest_sync(
            operation,
            {self.type: text[:10000]},
            _PROTECTED_MATERIAL_API_VERSION,
        )
        evaluations = self.get_evaluation_response(result)
        violations = [e for e in evaluations if e.detected]
        logger.debug(
            "[%s] Protected material detection: detected=%s",
            self.name,
            bool(violations),
        )
        return violations

    async def _detect_async(
        self, operation: str, text: str
    ) -> List[ProtectedMaterialEvaluation]:
        """Call the asynchronous protected material detection REST API."""
        result = await self._send_rest_async(
            operation,
            {self.type: text[:10000]},
            _PROTECTED_MATERIAL_API_VERSION,
        )
        evaluations = self.get_evaluation_response(result)
        violations = [e for e in evaluations if e.detected]
        logger.debug(
            "[%s] Protected material detection: detected=%s",
            self.name,
            bool(violations),
        )
        return violations
