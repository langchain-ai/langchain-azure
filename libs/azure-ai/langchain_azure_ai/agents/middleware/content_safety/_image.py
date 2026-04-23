"""Image content safety middleware for Azure AI Content Safety."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageCategory, ImageData
from langchain.agents.middleware import AgentState, Runtime
from langchain_core.messages.content import NonStandardAnnotation

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.middleware.content_safety._base import (
    ContentModerationEvaluation,
    ContentSafetyAnnotationPayload,
    ContentSafetyEvaluation,
    _AzureContentSafetyBaseMiddleware,
)


@dataclass
class ImageModerationInput:
    """Input extracted from an agent state for image content moderation.

    This is the return type for a ``context_extractor`` callable passed to
    :class:`~langchain_azure_ai.agents.middleware.content_safety.AzureContentModerationForImagesMiddleware`.

    Attributes:
        images: List of image descriptors.  Each entry is a dict with either a
            ``"content"`` key (``bytes`` for base64-decoded images) or a
            ``"url"`` key (``str`` for HTTP(S) image URLs).
    """

    images: List[Dict[str, Any]]


logger = logging.getLogger(__name__)


@experimental()
class AzureContentModerationForImagesMiddleware(_AzureContentSafetyBaseMiddleware):
    """AgentMiddleware that screens **image** content with Azure AI Content Safety.

    Use this middleware alongside :class:`AzureContentModerationMiddleware` when
    your agent handles visual content.  Because image analysis uses a separate
    API endpoint (``analyze_image``) and different category enumerations, a
    dedicated class keeps each concern focused and composable.

    The middleware extracts images from the most recent ``HumanMessage`` (input)
    and, optionally, from the most recent ``AIMessage`` (output).  It supports:

    * **Base64 data URLs** – ``data:image/png;base64,<data>``
    * **HTTP(S) URLs** – publicly accessible image URLs

    Content is analyzed using the Azure AI Content Safety image analysis API.
    When violations are detected, the middleware takes one of two actions:

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
        categories: Image harm categories to analyse.  Valid values are
            ``"Hate"``, ``"SelfHarm"``, ``"Sexual"``, and ``"Violence"``.
            Defaults to all four.
        severity_threshold: Minimum severity score (0–6) that triggers the
            configured exit behaviour.  Defaults to ``4`` (medium).
        exit_behavior: What to do when a violation is detected.  One of
            ``"error"`` (default) or ``"continue"``.
        violation_message: Custom text used to replace the offending message
            when ``exit_behavior="continue"``.  Defaults to a message built
            from the service response.
        apply_to_input: Whether to screen images in the last ``HumanMessage``.
            Defaults to ``True``.
        apply_to_output: Whether to screen images in the last ``AIMessage``.
            Defaults to ``False`` (agents rarely produce images directly).
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_content_safety_image"``.
        context_extractor: Optional callable with signature
            ``(state, runtime) -> Optional[ImageModerationInput]``
            that receives the current graph state and the LangGraph
            :class:`~langchain.agents.middleware.Runtime` execution context,
            and returns the images to screen, or ``None`` to skip evaluation
            entirely.  When ``None`` (default) the middleware uses its
            built-in extraction logic.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        project_endpoint: Optional[str] = None,
        categories: Optional[List[str]] = None,
        severity_threshold: int = 4,
        exit_behavior: Literal["error", "continue"] = "error",
        violation_message: Optional[str] = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        name: str = "azure_content_safety_image",
        context_extractor: Optional[
            Callable[[AgentState[Any], Runtime[Any]], Optional[ImageModerationInput]]
        ] = None,
    ) -> None:
        """Initialise the image content safety middleware.

        Args:
            endpoint: Azure Content Safety resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            project_endpoint: Azure AI Foundry project endpoint URL.
                Mutually exclusive with ``endpoint``.
            categories: Image harm categories to analyse.
            severity_threshold: Minimum severity score that triggers action.
            exit_behavior: ``"error"`` or ``"continue"``.
            violation_message: Custom replacement text for ``"continue"`` mode.
            apply_to_input: Screen images in the last HumanMessage.
            apply_to_output: Screen images in the last AIMessage.
            name: Node-name prefix for LangGraph wiring.
            context_extractor: Optional callable that extracts the images to
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
        self._severity_threshold = severity_threshold
        self._categories: List[str] = categories or [
            "Hate",
            "SelfHarm",
            "Sexual",
            "Violence",
        ]
        self._context_extractor = context_extractor

    # ------------------------------------------------------------------
    # Annotation / violation helpers
    # ------------------------------------------------------------------

    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` for image content safety evaluations."""
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="image_content_safety",
                violations=[v.to_dict() for v in evaluations],
            ).to_dict(),
        )

    def get_evaluation_response(
        self, response: Any
    ) -> List[ContentModerationEvaluation]:
        """Parse an ``AnalyzeImageResult`` into typed evaluation objects."""
        evaluations: List[ContentModerationEvaluation] = []
        for cat in response.categories_analysis:
            evaluations.append(
                ContentModerationEvaluation(
                    category=str(cat.category),
                    severity=cat.severity,
                )
            )
        return evaluations

    # ------------------------------------------------------------------
    # Input extraction
    # ------------------------------------------------------------------

    def _extract_image_input(
        self, state: AgentState[Any], runtime: Runtime[Any], *, is_input: bool
    ) -> Optional[ImageModerationInput]:
        """Extract the images to screen from the agent state.

        When a ``context_extractor`` was provided at construction time it is
        called with the current state and the LangGraph execution context.
        Otherwise the default heuristics are used: images from the last
        ``HumanMessage`` for input screening and from the last ``AIMessage``
        for output screening.

        Args:
            state: Current LangGraph state dict.
            runtime: The LangGraph execution context.
            is_input: ``True`` when screening agent input (``before_agent``),
                ``False`` when screening agent output (``after_agent``).

        Returns:
            An :class:`ImageModerationInput` instance, or ``None`` to skip
            evaluation.
        """
        if self._context_extractor is not None:
            return self._context_extractor(state, runtime)

        msg = (
            self.get_human_message_from_state(state)
            if is_input
            else self.get_ai_message_from_state(state)
        )
        images = self._images_from_message(msg) if msg else []
        return ImageModerationInput(images=images) if images else None

    # ------------------------------------------------------------------
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen images in the last HumanMessage before the agent runs."""
        if not self.apply_to_input:
            return None
        inputs = self._extract_image_input(state, runtime, is_input=True)
        if inputs is None:
            logger.debug("[%s] before_agent: no images found in input", self.name)
            return None
        logger.debug(
            "[%s] before_agent: found %d image(s) in input",
            self.name,
            len(inputs.images),
        )
        offending = self.get_human_message_from_state(state)
        return self._screen_images_sync(inputs.images, "agent.input", offending)

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Screen images in the last AIMessage after the agent runs."""
        if not self.apply_to_output:
            return None
        inputs = self._extract_image_input(state, runtime, is_input=False)
        if inputs is None:
            logger.debug("[%s] after_agent: no images found in output", self.name)
            return None
        logger.debug(
            "[%s] after_agent: found %d image(s) in output",
            self.name,
            len(inputs.images),
        )
        offending = self.get_ai_message_from_state(state)
        return self._screen_images_sync(inputs.images, "agent.output", offending)

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_agent`."""
        if not self.apply_to_input:
            return None
        inputs = self._extract_image_input(state, runtime, is_input=True)
        if inputs is None:
            logger.debug("[%s] abefore_agent: no images found in input", self.name)
            return None
        logger.debug(
            "[%s] abefore_agent: found %d image(s) in input",
            self.name,
            len(inputs.images),
        )
        offending = self.get_human_message_from_state(state)
        return await self._screen_images_async(inputs.images, "agent.input", offending)

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_agent`."""
        if not self.apply_to_output:
            return None
        inputs = self._extract_image_input(state, runtime, is_input=False)
        if inputs is None:
            logger.debug("[%s] aafter_agent: no images found in output", self.name)
            return None
        logger.debug(
            "[%s] aafter_agent: found %d image(s) in output",
            self.name,
            len(inputs.images),
        )
        offending = self.get_ai_message_from_state(state)
        return await self._screen_images_async(inputs.images, "agent.output", offending)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _screen_images_sync(
        self,
        images: List[Dict[str, Any]],
        context: Literal["agent.input", "agent.output", "model.input", "model.output"],
        offending_message: Optional[Any] = None,
    ) -> dict[str, Any] | None:
        """Analyse a list of images synchronously and handle violations."""
        if not images:
            return None
        logger.debug(
            "[%s] Screening %d image(s) for %s", self.name, len(images), context
        )
        all_violations: List[ContentSafetyEvaluation] = []
        for img in images:
            violations = self._analyze_image_sync(img)
            all_violations.extend(violations)
        return self._handle_violations(all_violations, context, offending_message)

    async def _screen_images_async(
        self,
        images: List[Dict[str, Any]],
        context: Literal["agent.input", "agent.output", "model.input", "model.output"],
        offending_message: Optional[Any] = None,
    ) -> dict[str, Any] | None:
        """Analyse a list of images asynchronously and handle violations."""
        if not images:
            return None
        logger.debug(
            "[%s] Screening %d image(s) for %s", self.name, len(images), context
        )
        all_violations: List[ContentSafetyEvaluation] = []
        for img in images:
            violations = await self._analyze_image_async(img)
            all_violations.extend(violations)
        return self._handle_violations(all_violations, context, offending_message)

    def _analyze_image_sync(
        self, image: Dict[str, Any]
    ) -> List[ContentModerationEvaluation]:
        """Call the synchronous image analysis API."""
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
        evaluations = self.get_evaluation_response(response)
        return [v for v in evaluations if v.severity >= self._severity_threshold]

    async def _analyze_image_async(
        self, image: Dict[str, Any]
    ) -> List[ContentModerationEvaluation]:
        """Call the asynchronous image analysis API."""
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
        evaluations = self.get_evaluation_response(response)
        return [v for v in evaluations if v.severity >= self._severity_threshold]

    @staticmethod
    def _images_from_message(msg: Any) -> List[Dict[str, Any]]:
        """Extract image descriptors from a LangChain message's content.

        Supported block shapes:

        * ``{"type": "image_url", "image_url": "data:image/png;base64,<b64>"}``
        * ``{"type": "image_url", "image_url": "https://..."}``
        * ``{"type": "image_url", "image_url": {"url": "data:..." | "https://..."}}``

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
