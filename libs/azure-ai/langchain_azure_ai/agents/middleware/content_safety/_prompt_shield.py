"""Prompt shield middleware for Azure AI Content Safety."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from langchain.agents.middleware import AgentState, Runtime
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.content import NonStandardAnnotation

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.middleware.content_safety._base import (
    ContentSafetyAnnotationPayload,
    ContentSafetyEvaluation,
    _AzureContentSafetyBaseMiddleware,
)


@dataclass(frozen=True)
class PromptInjectionEvaluation(ContentSafetyEvaluation):
    """A prompt-injection evaluation."""

    source: str = ""
    detected: bool = True


@dataclass
class PromptShieldInput:
    """Input extracted from an agent state for prompt shield evaluation.

    This is the return type for a ``context_extractor`` callable passed to
    :class:`~langchain_azure_ai.agents.middleware.content_safety.AzurePromptShieldMiddleware`.

    Attributes:
        user_prompt: The user's input text to evaluate for direct prompt injection.
        documents: External document texts (e.g. tool / function-call results)
            to evaluate for indirect prompt injection.  Defaults to an empty list.
    """

    user_prompt: str
    documents: List[str] = field(default_factory=list)


logger = logging.getLogger(__name__)

_PROMPT_SHIELD_API_VERSION = "2024-09-01"


@experimental()
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

    When an injection attack is detected, the middleware takes one of two
    actions:

    * ``"error"`` – raises :exc:`ContentSafetyViolationError`, halting the graph.
    * ``"continue"`` – replaces the offending message with a violation notice
      (either a service-derived description or a custom ``violation_message``)
      and lets execution proceed.

    Note:
        ``apply_to_output`` defaults to ``False`` because prompt injection is
        an input-side attack.  Set it to ``True`` if you want to screen AI
        output as well.

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
        exit_behavior: What to do when an injection is detected.  One of ``"error"``
            (default) or ``"continue"``.
        violation_message: Custom text used to replace the offending message
            when ``exit_behavior="continue"``.  Defaults to a message built
            from the service response.
        apply_to_input: Whether to screen the agent's input (last
            ``HumanMessage``).  Defaults to ``True``.
        apply_to_output: Whether to screen the agent's output (last
            ``AIMessage``).  Defaults to ``False``.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_prompt_shield"``.
        context_extractor: Optional callable with signature
            ``(state, runtime) -> Optional[PromptShieldInput]``
            that receives the current graph state and the LangGraph
            :class:`~langchain.agents.middleware.Runtime` execution context,
            and returns the user prompt and documents to screen, or ``None``
            to skip evaluation entirely.  When ``None`` (default) the
            middleware uses its built-in extraction logic.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        project_endpoint: Optional[str] = None,
        exit_behavior: Literal["error", "continue", "replace"] = "error",
        violation_message: Optional[str] = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        name: str = "azure_prompt_shield",
        context_extractor: Optional[
            Callable[[AgentState[Any], Runtime[Any]], Optional[PromptShieldInput]]
        ] = None,
    ) -> None:
        """Initialise the prompt shield middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            project_endpoint: Azure AI Foundry project endpoint URL.
                Mutually exclusive with ``endpoint``.
            exit_behavior: ``"error"``, ``"continue"``, or ``"replace"``.
            violation_message: Custom replacement text for ``"replace"`` mode.
            apply_to_input: Screen the last HumanMessage before agent runs.
            apply_to_output: Screen the last AIMessage after agent runs.
            name: Node-name prefix for LangGraph wiring.
            context_extractor: Optional callable that extracts the user prompt
                and documents to screen from the agent state and the LangGraph
                execution context instead of using the built-in heuristics.
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
        self._context_extractor = context_extractor

    # ------------------------------------------------------------------
    # Annotation / violation helpers
    # ------------------------------------------------------------------

    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` for prompt injection evaluations."""
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="prompt_injection",
                violations=[v.to_dict() for v in evaluations],
            ).to_dict(),
        )

    def get_evaluation_response(
        self, response: Dict[str, Any]
    ) -> List[PromptInjectionEvaluation]:
        """Parse a ``shieldPrompt`` REST response into evaluation objects.

        Returns one evaluation per source (user prompt + each document),
        regardless of whether an attack was detected.
        """
        evaluations: List[PromptInjectionEvaluation] = []
        prompt_analysis = response.get("userPromptAnalysis")
        if prompt_analysis:
            evaluations.append(
                PromptInjectionEvaluation(
                    category="PromptInjection",
                    source="user_prompt",
                    detected=prompt_analysis.get("attackDetected", False),
                )
            )
        for i, doc_analysis in enumerate(response.get("documentsAnalysis") or []):
            evaluations.append(
                PromptInjectionEvaluation(
                    category="PromptInjection",
                    source=f"document[{i}]",
                    detected=doc_analysis.get("attackDetected", False),
                )
            )
        return evaluations

    # ------------------------------------------------------------------
    # Input extraction
    # ------------------------------------------------------------------

    def _extract_prompt_shield_inputs(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> Optional[PromptShieldInput]:
        """Extract the user prompt and documents from the agent state.

        When a ``context_extractor`` was provided at construction time it is
        called with the current state and the LangGraph execution context.
        Otherwise the default heuristics are used: the last ``HumanMessage``
        as the user prompt and any ``ToolMessage`` items as documents.

        Args:
            state: Current LangGraph state dict.
            runtime: The LangGraph execution context.

        Returns:
            A :class:`PromptShieldInput` instance, or ``None`` when no
            user-prompt text can be found.
        """
        if self._context_extractor is not None:
            return self._context_extractor(state, runtime)

        offending = self.get_human_message_from_state(state)
        user_prompt = self.get_text_from_message(offending)
        if not user_prompt:
            return None
        documents = self._extract_tool_texts(state.get("messages", []))
        return PromptShieldInput(user_prompt=user_prompt, documents=documents)

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen the last HumanMessage for prompt injection before the agent runs."""
        if not self.apply_to_input:
            return None
        inputs = self._extract_prompt_shield_inputs(state, runtime)
        if inputs is None:
            logger.debug("[%s] before_agent: no HumanMessage text found", self.name)
            return None
        logger.debug(
            "[%s] before_agent: shielding input (%d chars, %d documents)",
            self.name,
            len(inputs.user_prompt),
            len(inputs.documents),
        )
        offending = self.get_human_message_from_state(state)
        violations = self._shield_sync(
            user_prompt=inputs.user_prompt, documents=inputs.documents
        )
        return self._handle_violations(violations, "agent.input", offending)

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_agent`."""
        if not self.apply_to_input:
            return None
        inputs = self._extract_prompt_shield_inputs(state, runtime)
        if inputs is None:
            logger.debug("[%s] abefore_agent: no HumanMessage text found", self.name)
            return None
        logger.debug(
            "[%s] abefore_agent: shielding input (%d chars, %d documents)",
            self.name,
            len(inputs.user_prompt),
            len(inputs.documents),
        )
        offending = self.get_human_message_from_state(state)
        violations = await self._shield_async(
            user_prompt=inputs.user_prompt, documents=inputs.documents
        )
        return self._handle_violations(violations, "agent.input", offending)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shield_sync(
        self, *, user_prompt: str, documents: List[str]
    ) -> List[PromptInjectionEvaluation]:
        """Call the synchronous prompt shield REST API."""
        body: Dict[str, Any] = {"userPrompt": user_prompt[:10000]}
        if documents:
            body["documents"] = [d[:10000] for d in documents]
        result = self._send_rest_sync(
            "text:shieldPrompt", body, _PROMPT_SHIELD_API_VERSION
        )
        evaluations = self.get_evaluation_response(result)
        violations = [e for e in evaluations if e.detected]
        logger.debug(
            "[%s] Prompt shield detection: %d injection(s) found",
            self.name,
            len(violations),
        )
        return violations

    async def _shield_async(
        self, *, user_prompt: str, documents: List[str]
    ) -> List[PromptInjectionEvaluation]:
        """Call the asynchronous prompt shield REST API."""
        body: Dict[str, Any] = {"userPrompt": user_prompt[:10000]}
        if documents:
            body["documents"] = [d[:10000] for d in documents]
        result = await self._send_rest_async(
            "text:shieldPrompt", body, _PROMPT_SHIELD_API_VERSION
        )
        evaluations = self.get_evaluation_response(result)
        violations = [e for e in evaluations if e.detected]
        logger.debug(
            "[%s] Prompt shield detection: %d injection(s) found",
            self.name,
            len(violations),
        )
        return violations

    @staticmethod
    def _collect_injection_violations(
        response: Dict[str, Any],
    ) -> List[PromptInjectionEvaluation]:
        """Extract injection violations from a ``shieldPrompt`` REST response.

        .. deprecated::
            Use :meth:`get_evaluation_response` instead.
        """
        violations: List[PromptInjectionEvaluation] = []
        prompt_analysis = response.get("userPromptAnalysis")
        if prompt_analysis and prompt_analysis.get("attackDetected"):
            violations.append(
                PromptInjectionEvaluation(
                    category="PromptInjection",
                    source="user_prompt",
                )
            )
        for i, doc_analysis in enumerate(response.get("documentsAnalysis") or []):
            if doc_analysis.get("attackDetected"):
                violations.append(
                    PromptInjectionEvaluation(
                        category="PromptInjection",
                        source=f"document[{i}]",
                    )
                )
        return violations

    def _extract_tool_texts(self, messages: Sequence[BaseMessage]) -> List[str]:
        """Extract text content from all ToolMessage items in the message list."""
        texts: List[str] = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            text = self.get_text_from_message(msg)
            if text:
                texts.append(text)
        return texts
