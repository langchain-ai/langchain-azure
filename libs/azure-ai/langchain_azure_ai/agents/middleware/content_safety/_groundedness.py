"""Groundedness detection middleware for Azure AI Content Safety."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain.agents.middleware import AgentState, Runtime
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.content import NonStandardAnnotation

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.middleware.content_safety._base import (
    ContentSafetyAnnotationPayload,
    ContentSafetyEvaluation,
    GroundednessEvaluation,
    _AzureContentSafetyBaseMiddleware,
    _GroundednessState,
)

logger = logging.getLogger(__name__)

_GROUNDEDNESS_API_VERSION = "2024-02-15-preview"


@experimental()
class AzureGroundednessMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that evaluates groundedness of model outputs.

    Groundedness detection analyses language model outputs to determine whether
    they are factually aligned with user-provided information or contain
    fictional/hallucinated content.

    The middleware runs as an ``after_model`` hook — it evaluates every model
    response immediately after generation.  The behaviour when ungrounded
    content is detected depends on ``exit_behavior``:

    * ``"error"`` (default) – raises :exc:`ContentSafetyViolationError`,
      halting the graph.  The exception carries the evaluation details.
    * ``"continue"`` – annotates the state with a ``groundedness_evaluation``
      key containing the evaluation results and lets execution proceed.

    In both modes the state is annotated with the evaluation result so callers
    can inspect it.

    **Grounding sources** are collected automatically from the chat history:

    * ``SystemMessage`` content – the system prompt often contains the
      authoritative context the model should stay grounded in.
    * ``ToolMessage`` content – tool / function-call results such as RAG
      retrieval chunks, web-search snippets, or database lookups.
    * ``AIMessage`` annotation titles – citation metadata attached to model
      responses (e.g. ``url_citation`` annotations from web-search grounding).

    After the model runs (in ``continue`` mode), the state will contain a
    ``groundedness_evaluation`` key with the evaluation results::

        {
            "is_grounded": True,
            "ungrounded_percentage": 0.0,
            "details": []
        }

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
        domain: The domain of the text for analysis.  ``"Generic"`` (default) or
            ``"Medical"``.
        task: The task type for the analysis.  ``"Summarization"`` (default) or
            ``"QnA"``.
        exit_behavior: What to do when ungrounded content is detected.  One of
            ``"error"`` (default) or ``"continue"``.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_groundedness"``.
    """

    #: State schema contributed by this middleware.
    state_schema: type = _GroundednessState

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        project_endpoint: Optional[str] = None,
        domain: Literal["Generic", "Medical"] = "Generic",
        task: Literal["Summarization", "QnA"] = "Summarization",
        exit_behavior: Literal["error", "continue"] = "error",
        name: str = "azure_groundedness",
    ) -> None:
        """Initialise the groundedness detection middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            project_endpoint: Azure AI Foundry project endpoint URL.
                Mutually exclusive with ``endpoint``.
            domain: ``"Generic"`` or ``"Medical"``.
            task: ``"Summarization"`` or ``"QnA"``.
            exit_behavior: ``"error"`` (default) raises
                :exc:`ContentSafetyViolationError`; ``"continue"`` appends the
                groundedness evaluation to the state and continues.
            name: Node-name prefix for LangGraph wiring.
        """
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            project_endpoint=project_endpoint,
            exit_behavior=exit_behavior,
            apply_to_input=False,
            apply_to_output=False,
            name=name,
        )
        self._domain = domain
        self._task = task

    # ------------------------------------------------------------------
    # Annotation / violation helpers
    # ------------------------------------------------------------------

    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` for groundedness violations."""
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="groundedness",
                violations=[v.to_dict() for v in evaluations],
            ).to_dict(),
        )

    def get_evaluation_response(
        self, response: Dict[str, Any]
    ) -> List[GroundednessEvaluation]:
        """Build an annotation dict from a ``detectGroundedness`` API response."""
        return [
            GroundednessEvaluation(
                is_grounded=not response.get("ungroundedDetected", False),
                ungrounded_percentage=response.get("ungroundedPercentage", 0),
                details=response.get("ungroundedDetails", []),
            )
        ]

    # ------------------------------------------------------------------
    # Synchronous hook
    # ------------------------------------------------------------------

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Evaluate groundedness of the last model response.

        Collects grounding sources from ``SystemMessage`` and ``ToolMessage``
        items in the conversation, sends the last ``AIMessage`` to the
        groundedness detection API, and either annotates the state with the
        result (``exit_behavior="continue"``) or raises an error
        (``exit_behavior="error"``).

        Args:
            state: Current LangGraph state dict.
            runtime: The runtime context.

        Returns:
            A state-patch dict containing a ``groundedness_evaluation`` key,
            or ``None`` when evaluation cannot be performed (no AI message
            or no grounding sources).

        Raises:
            ContentSafetyViolationError: When ``exit_behavior="error"`` and
                the model output is ungrounded.
        """
        prompt = self.get_human_message_from_state(state)
        prompt_text = self.get_text_from_message(prompt)
        to_ground = self.get_ai_message_from_state(state)
        to_ground_text = self.get_text_from_message(to_ground)

        if not to_ground_text:
            logger.debug("[%s] after_model: no AIMessage text found", self.name)
            return None

        sources = self._gather_grounding_sources(state.get("messages", []))

        if not sources:
            logger.debug("[%s] after_model: no grounding sources found", self.name)
            return None

        logger.debug(
            "[%s] after_model: evaluating groundedness (%d sources)",
            self.name,
            len(sources),
        )
        body = self._build_request_body(to_ground_text, sources, prompt_text)
        result = self._send_rest_sync(
            "text:detectGroundedness", body, _GROUNDEDNESS_API_VERSION
        )
        evaluations = self.get_evaluation_response(result)
        logger.debug(
            "[%s] Groundedness result: is_grounded=%s, ungrounded_percentage=%s",
            self.name,
            evaluations[0].is_grounded,
            evaluations[0].ungrounded_percentage,
        )
        if not evaluations[0].is_grounded:
            self._handle_violations(evaluations, "model.output", to_ground)

        return {
            "groundedness_evaluation": [
                evaluation.to_dict() for evaluation in evaluations
            ]
        }

    # ------------------------------------------------------------------
    # Asynchronous hook
    # ------------------------------------------------------------------

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_model`."""
        prompt = self.get_human_message_from_state(state)
        prompt_text = self.get_text_from_message(prompt)
        to_ground = self.get_ai_message_from_state(state)
        to_ground_text = self.get_text_from_message(to_ground)

        if not to_ground_text:
            logger.debug("[%s] aafter_model: no AIMessage text found", self.name)
            return None

        sources = self._gather_grounding_sources(state.get("messages", []))

        if not sources:
            logger.debug("[%s] aafter_model: no grounding sources found", self.name)
            return None

        logger.debug(
            "[%s] aafter_model: evaluating groundedness (%d sources)",
            self.name,
            len(sources),
        )
        body = self._build_request_body(to_ground_text, sources, prompt_text)
        result = await self._send_rest_async(
            "text:detectGroundedness", body, _GROUNDEDNESS_API_VERSION
        )
        evaluations = self.get_evaluation_response(result)
        logger.debug(
            "[%s] Groundedness result: is_grounded=%s, ungrounded_percentage=%s",
            self.name,
            evaluations[0].is_grounded,
            evaluations[0].ungrounded_percentage,
        )
        if not evaluations[0].is_grounded:
            self._handle_violations(evaluations, "model.output", to_ground)

        return {
            "groundedness_evaluation": [
                evaluation.to_dict() for evaluation in evaluations
            ]
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_grounding_sources(self, messages: Sequence[BaseMessage]) -> List[str]:
        """Collect grounding sources from the conversation history.

        Sources are gathered from:

        * ``SystemMessage`` content – the system prompt.
        * ``ToolMessage`` content – tool / function-call results.
        * ``AIMessage`` annotation titles – citation metadata attached to
          model responses (e.g. ``url_citation`` annotations produced by
          web-search grounding).
        """
        sources: List[str] = []
        for msg in messages:
            if isinstance(msg, (SystemMessage, ToolMessage)):
                text = self.get_text_from_message(msg)
                if text:
                    sources.append(text)
            elif isinstance(msg, AIMessage):
                self._collect_annotation_sources(msg, sources)
        logger.debug("Gathered %d grounding source(s) from messages", len(sources))
        return sources

    @staticmethod
    def _collect_annotation_sources(msg: AIMessage, sources: List[str]) -> None:
        """Append grounding-source strings from AIMessage annotations."""
        if not isinstance(msg.content, list):
            return
        for block in msg.content:
            if not isinstance(block, dict):
                continue
            for ann in block.get("annotations", []):
                if not isinstance(ann, dict):
                    continue
                title = ann.get("title")
                if title:
                    sources.append(title)

    def _build_request_body(
        self, text: str, sources: List[str], prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Build the JSON request body for the groundedness detection API."""
        body: Dict[str, Any] = {
            "domain": self._domain,
            "task": self._task,
            "text": text[-7500:],  # grab the last 7500 chars to stay within the 10k
            "groundingSources": sources,
        }
        if self._task == "QnA" and prompt:
            body["qna"] = {"query": prompt[:7500]}
        return body
