"""Conversation PII redaction middleware for Azure AI Language Service."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.content import NonStandardAnnotation

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai._resources import _get_base_url_from_endpoint
from langchain_azure_ai.agents.middleware.content_safety._base import (
    ContentSafetyAnnotationPayload,
    ContentSafetyViolationError,
    PIIEntityEvaluation,
    _PIIRedactionState,
)

try:
    from azure.ai.language.conversations import ConversationAnalysisClient
    from azure.ai.language.conversations.aio import (
        ConversationAnalysisClient as AsyncConversationAnalysisClient,
    )
except ImportError:
    ConversationAnalysisClient = None  # type: ignore[assignment, misc]
    AsyncConversationAnalysisClient = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


@experimental()
class AzureConversationPIIRedaction(AgentMiddleware[AgentState[Any], Any]):
    """AgentMiddleware that redacts PII from conversations using Azure AI Language.

    This middleware uses the Azure AI Language Service's Conversation PII
    Detection feature to identify and redact Personally Identifiable
    Information (PII) from messages flowing in and out of an agent.

    PII categories include names, phone numbers, email addresses, social
    security numbers, credit card numbers, and many more.  When PII is
    detected, the middleware can either replace the message content with
    the redacted version (``"replace"`` mode) or annotate the message with
    detected entities without altering the content (``"continue"`` mode).

    Pass this class in the ``middleware`` parameter of any LangChain
    ``create_agent`` call:

    .. code-block:: python

        from langchain.agents import create_agent
        from langchain_azure_ai.agents.middleware import (
            AzureConversationPIIRedaction,
        )

        agent = create_agent(
            model="azure_ai:gpt-4.1",
            middleware=[
                # Redact PII from user input before the agent sees it
                AzureConversationPIIRedaction(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    apply_to_input=True,
                    apply_to_output=False,
                ),
            ],
        )

    You can also redact PII from agent output before the user sees it:

    .. code-block:: python

        agent = create_agent(
            model="azure_ai:gpt-4.1",
            middleware=[
                AzureConversationPIIRedaction(
                    endpoint="https://my-resource.cognitiveservices.azure.com/",
                    apply_to_input=True,
                    apply_to_output=True,
                ),
            ],
        )

    In ``"replace"`` mode (default), the message content is replaced with
    the redacted text returned by the Azure service (e.g. ``"My name is
    [PERSON]"``).  In ``"continue"`` mode, the original content is preserved
    and the detected PII entities are added as annotations on the message.
    In ``"error"`` mode, a :exc:`ContentSafetyViolationError` is raised
    whenever PII is detected.

    Both synchronous (``before_agent`` / ``after_agent``) and asynchronous
    (``abefore_agent`` / ``aafter_agent``) hooks are implemented.

    Note:
        This middleware requires the ``azure-ai-language-conversations``
        package.  Install it with:
        ``pip install "langchain-azure-ai[language]"``

    Args:
        endpoint: Azure AI Language resource endpoint URL.  Falls back to
            the ``AZURE_LANGUAGE_ENDPOINT`` environment variable.
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
        pii_categories: List of PII entity categories to detect and redact.
            When ``None`` (default), all supported PII categories are
            detected.  Examples: ``["Person", "PhoneNumber", "Email"]``.
        language: BCP-47 language code for the conversation language.
            Defaults to ``"en"`` (English).
        model_version: Model version to use for analysis.  Defaults to
            ``"latest"``.
        exit_behavior: What to do when PII is detected.  One of
            ``"replace"`` (default) – replaces the message with the
            redacted text from the service; ``"continue"`` – annotates the
            message with detected entities and lets execution proceed;
            ``"error"`` – raises :exc:`ContentSafetyViolationError`.
        apply_to_input: Whether to redact PII from the agent's input (last
            ``HumanMessage``).  Defaults to ``True``.
        apply_to_output: Whether to redact PII from the agent's output
            (last ``AIMessage``).  Defaults to ``True``.
        name: Node-name prefix used when wiring this middleware into a
            LangGraph.  Defaults to ``"azure_conversation_pii_redaction"``.
    """

    #: State schema contributed by this middleware.
    state_schema: type = _PIIRedactionState

    #: Extra LangGraph tools contributed by this middleware (none by default).
    tools: list = []

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        project_endpoint: Optional[str] = None,
        pii_categories: Optional[List[str]] = None,
        language: str = "en",
        model_version: str = "latest",
        exit_behavior: Literal["replace", "continue", "error"] = "replace",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        name: str = "azure_conversation_pii_redaction",
    ) -> None:
        """Initialise the Conversation PII redaction middleware.

        Args:
            endpoint: Azure AI Language resource endpoint URL.
            credential: Azure credential (TokenCredential, AzureKeyCredential,
                or API-key string).  Defaults to DefaultAzureCredential.
            project_endpoint: Azure AI Foundry project endpoint URL.
                Mutually exclusive with ``endpoint``.
            pii_categories: PII entity categories to detect.  ``None`` means
                all categories.
            language: BCP-47 language code.  Defaults to ``"en"``.
            model_version: Model version for the Language service.
            exit_behavior: ``"replace"`` (default), ``"continue"``, or
                ``"error"``.
            apply_to_input: Redact PII from the last HumanMessage.
            apply_to_output: Redact PII from the last AIMessage.
            name: Node-name prefix for LangGraph wiring.
        """
        # Check SDK availability at instantiation time so that mocking
        # sys.modules after the module has been imported still works.
        try:
            from azure.ai.language.conversations import (  # noqa: F401
                ConversationAnalysisClient as _CAC,
            )
        except ImportError:
            raise ImportError(
                "The 'azure-ai-language-conversations' package is required to use "
                "AzureConversationPIIRedaction middleware.  Install it with:\n"
                "  `pip install azure-ai-language-conversations`"
            )

        # Validate mutual exclusivity before falling back to env vars.
        if endpoint and project_endpoint:
            raise ValueError(
                "'endpoint' and 'project_endpoint' are mutually exclusive. "
                "Provide only one."
            )

        # Resolve from environment variables when not explicitly provided.
        if not endpoint and not project_endpoint:
            endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
        if not endpoint and not project_endpoint:
            project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")

        if not endpoint and not project_endpoint:
            raise ValueError(
                "An endpoint is required.  Pass 'endpoint' or "
                "'project_endpoint', or set the "
                "AZURE_LANGUAGE_ENDPOINT / AZURE_AI_PROJECT_ENDPOINT "
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

        self._pii_categories = pii_categories
        self._language = language
        self._model_version = model_version
        self.exit_behavior = exit_behavior
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
    # Synchronous hooks
    # ------------------------------------------------------------------

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Redact PII from the last HumanMessage before the agent runs.

        Args:
            state: Current LangGraph state dict.
            runtime: The runtime context.

        Returns:
            ``None``, or a state-patch with ``pii_redaction_result`` when
            PII was detected.

        Raises:
            ContentSafetyViolationError: When ``exit_behavior="error"`` and
                PII is detected.
        """
        if not self.apply_to_input:
            return None
        msg = self._get_human_message(state)
        text = self._get_text(msg)
        if not text:
            logger.debug("[%s] before_agent: no HumanMessage text found", self.name)
            return None
        logger.debug(
            "[%s] before_agent: redacting PII from input (%d chars)",
            self.name,
            len(text),
        )
        redacted_text, entities = self._redact_sync(text)
        return self._apply_result(
            entities, redacted_text, "agent.input", msg
        )

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Redact PII from the last AIMessage after the agent runs.

        Args:
            state: Current LangGraph state dict.
            runtime: The runtime context.

        Returns:
            ``None``, or a state-patch with ``pii_redaction_result`` when
            PII was detected.

        Raises:
            ContentSafetyViolationError: When ``exit_behavior="error"`` and
                PII is detected.
        """
        if not self.apply_to_output:
            return None
        msg = self._get_ai_message(state)
        text = self._get_text(msg)
        if not text:
            logger.debug("[%s] after_agent: no AIMessage text found", self.name)
            return None
        logger.debug(
            "[%s] after_agent: redacting PII from output (%d chars)",
            self.name,
            len(text),
        )
        redacted_text, entities = self._redact_sync(text)
        return self._apply_result(
            entities, redacted_text, "agent.output", msg
        )

    # ------------------------------------------------------------------
    # Asynchronous hooks
    # ------------------------------------------------------------------

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_agent`."""
        if not self.apply_to_input:
            return None
        msg = self._get_human_message(state)
        text = self._get_text(msg)
        if not text:
            logger.debug(
                "[%s] abefore_agent: no HumanMessage text found", self.name
            )
            return None
        logger.debug(
            "[%s] abefore_agent: redacting PII from input (%d chars)",
            self.name,
            len(text),
        )
        redacted_text, entities = await self._redact_async(text)
        return self._apply_result(
            entities, redacted_text, "agent.input", msg
        )

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_agent`."""
        if not self.apply_to_output:
            return None
        msg = self._get_ai_message(state)
        text = self._get_text(msg)
        if not text:
            logger.debug(
                "[%s] aafter_agent: no AIMessage text found", self.name
            )
            return None
        logger.debug(
            "[%s] aafter_agent: redacting PII from output (%d chars)",
            self.name,
            len(text),
        )
        redacted_text, entities = await self._redact_async(text)
        return self._apply_result(
            entities, redacted_text, "agent.output", msg
        )

    # ------------------------------------------------------------------
    # Internal helpers – result application
    # ------------------------------------------------------------------

    def _apply_result(
        self,
        entities: List[PIIEntityEvaluation],
        redacted_text: str,
        context: Literal["agent.input", "agent.output"],
        msg: Optional[BaseMessage],
    ) -> dict[str, Any] | None:
        """Apply the PII redaction result according to ``exit_behavior``.

        Args:
            entities: PII entities detected in the message.
            redacted_text: The redacted text returned by the service.
            context: Human-readable context label.
            msg: The original message, or ``None``.

        Returns:
            A state-patch dict with ``pii_redaction_result``, or ``None``.

        Raises:
            ContentSafetyViolationError: When ``exit_behavior="error"`` and
                PII entities are present.
        """
        if not entities:
            logger.debug("[%s] No PII detected in %s", self.name, context)
            return None

        categories = ", ".join(sorted({e.category for e in entities}))
        logger.info(
            "[%s] %d PII entity(s) detected in %s: %s",
            self.name,
            len(entities),
            context,
            categories,
        )

        if self.exit_behavior == "error":
            raise ContentSafetyViolationError(
                f"PII detected in {context}: {categories}",
                entities,
            )

        if msg is not None:
            if self.exit_behavior == "replace":
                msg.content = redacted_text
            else:
                # "continue" – annotate the message with entity info.
                annotation = self._build_annotation(entities)
                sanitized_content = msg.content
                if isinstance(sanitized_content, str):
                    msg.content = [
                        {"type": "text", "text": sanitized_content},
                        dict(annotation),
                    ]
                else:
                    msg.content = list(sanitized_content) + [  # type: ignore[assignment]
                        dict(annotation)
                    ]

        return {
            "pii_redaction_result": {
                "context": context,
                "entities_detected": len(entities),
                "categories": list({e.category for e in entities}),
            }
        }

    def _build_annotation(
        self, entities: List[PIIEntityEvaluation]
    ) -> NonStandardAnnotation:
        """Build a ``NonStandardAnnotation`` for PII entities."""
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="conversation_pii_redaction",
                violations=[asdict(e) for e in entities],
            ).to_dict(),
        )

    # ------------------------------------------------------------------
    # Internal helpers – SDK calls
    # ------------------------------------------------------------------

    def _build_request_body(self, text: str) -> Dict[str, Any]:
        """Build the ``begin_analyze_conversation_job`` request body.

        Args:
            text: The text to analyse.

        Returns:
            A JSON-serialisable dict for the ``AnalyzeConversationOperationInput``.
        """
        action_content: Dict[str, Any] = {"modelVersion": self._model_version}
        if self._pii_categories:
            action_content["piiCategories"] = self._pii_categories

        return {
            "analysisInput": {
                "conversations": [
                    {
                        "id": "1",
                        "language": self._language,
                        "modality": "text",
                        "conversationItems": [
                            {
                                "id": "1",
                                "participantId": "participant",
                                "text": text,
                            }
                        ],
                    }
                ]
            },
            "tasks": [
                {
                    "kind": "ConversationPII",
                    "parameters": action_content,
                }
            ],
        }

    @staticmethod
    def _extract_entities_from_result(
        result: Dict[str, Any],
    ) -> tuple[str, List[PIIEntityEvaluation]]:
        """Parse the LRO job result into (redacted_text, entities).

        Args:
            result: The deserialized job result from the Language service.

        Returns:
            A tuple of the redacted text (first conversation item) and a
            list of :class:`PIIEntityEvaluation` instances for all detected
            entities.
        """
        entities: List[PIIEntityEvaluation] = []
        redacted_text = ""

        tasks = result.get("tasks", {})
        items = tasks.get("items", [])
        for task in items:
            task_result = task.get("results", {})
            for conversation in task_result.get("conversations", []):
                for item in conversation.get("conversationItems", []):
                    # Capture the first item's redacted content.
                    if not redacted_text:
                        redacted_text = item.get("redactedContent", {}).get(
                            "text", ""
                        )
                    for entity in item.get("entities", []):
                        entities.append(
                            PIIEntityEvaluation(
                                category=entity.get("category", "Unknown"),
                                text=entity.get("text", ""),
                                offset=entity.get("offset", 0),
                                length=entity.get("length", 0),
                                confidence_score=entity.get("confidenceScore", 0.0),
                                subcategory=entity.get("subcategory", ""),
                            )
                        )
        return redacted_text, entities

    def _redact_sync(self, text: str) -> tuple[str, List[PIIEntityEvaluation]]:
        """Call the synchronous Conversation PII API.

        Submits a job via ``begin_analyze_conversation_job`` (an LRO),
        waits for completion, then extracts the redacted text and entity
        list from the response.

        Args:
            text: Input text to analyse and redact.

        Returns:
            A tuple of ``(redacted_text, entities)``.
        """
        from azure.core.rest import HttpRequest

        body = self._build_request_body(text)
        logger.debug(
            "[%s] Submitting sync Conversation PII job (language=%s)",
            self.name,
            self._language,
        )
        endpoint = self._endpoint.rstrip("/")
        url = f"{endpoint}/language/analyze-conversations/jobs"
        request = HttpRequest(
            method="POST",
            url=url,
            params={"api-version": "2024-11-01"},
            json=body,
            headers={"Content-Type": "application/json"},
        )
        client = self._get_sync_client()
        response = client.send_request(request)
        response.raise_for_status()

        # Poll the operation URL from the Location header.
        operation_url = response.headers.get("Operation-Location") or response.headers.get(
            "operation-location"
        )
        if not operation_url:
            logger.warning(
                "[%s] No Operation-Location header in response; returning empty result",
                self.name,
            )
            return "", []

        result = self._poll_job_sync(client, operation_url)
        redacted_text, entities = self._extract_entities_from_result(result)
        logger.debug(
            "[%s] PII job complete: %d entity(s) found", self.name, len(entities)
        )
        return redacted_text, entities

    def _poll_job_sync(
        self, client: Any, operation_url: str
    ) -> Dict[str, Any]:
        """Poll a Language Service LRO job URL until completion.

        Args:
            client: The synchronous ``ConversationAnalysisClient``.
            operation_url: The job polling URL from ``Operation-Location``.

        Returns:
            Parsed JSON response body once the job has succeeded.

        Raises:
            RuntimeError: If the job fails or reaches an unexpected status.
        """
        import time

        from azure.core.rest import HttpRequest

        for _ in range(60):  # poll up to 60 times (~60s)
            poll_req = HttpRequest(method="GET", url=operation_url)
            poll_resp = client.send_request(poll_req)
            poll_resp.raise_for_status()
            data = poll_resp.json()
            status = data.get("status", "").lower()
            if status == "succeeded":
                return data
            if status in ("failed", "cancelled", "cancelling"):
                errors = data.get("errors", [])
                raise RuntimeError(
                    f"Conversation PII job {status}: {errors}"
                )
            time.sleep(1)

        raise RuntimeError("Conversation PII job did not complete within the timeout")

    async def _redact_async(
        self, text: str
    ) -> tuple[str, List[PIIEntityEvaluation]]:
        """Call the asynchronous Conversation PII API.

        Args:
            text: Input text to analyse and redact.

        Returns:
            A tuple of ``(redacted_text, entities)``.
        """
        import asyncio

        from azure.core.rest import HttpRequest

        body = self._build_request_body(text)
        logger.debug(
            "[%s] Submitting async Conversation PII job (language=%s)",
            self.name,
            self._language,
        )
        endpoint = self._endpoint.rstrip("/")
        url = f"{endpoint}/language/analyze-conversations/jobs"
        request = HttpRequest(
            method="POST",
            url=url,
            params={"api-version": "2024-11-01"},
            json=body,
            headers={"Content-Type": "application/json"},
        )
        client = self._get_async_client()
        response = await client.send_request(request)
        response.raise_for_status()

        operation_url = response.headers.get("Operation-Location") or response.headers.get(
            "operation-location"
        )
        if not operation_url:
            logger.warning(
                "[%s] No Operation-Location header in response; returning empty result",
                self.name,
            )
            return "", []

        result = await self._poll_job_async(client, operation_url, asyncio)
        redacted_text, entities = self._extract_entities_from_result(result)
        logger.debug(
            "[%s] Async PII job complete: %d entity(s) found",
            self.name,
            len(entities),
        )
        return redacted_text, entities

    @staticmethod
    async def _poll_job_async(
        client: Any, operation_url: str, asyncio_module: Any
    ) -> Dict[str, Any]:
        """Poll a Language Service LRO job URL asynchronously until completion.

        Args:
            client: The async ``ConversationAnalysisClient``.
            operation_url: The job polling URL from ``Operation-Location``.
            asyncio_module: The ``asyncio`` module (injected for testability).

        Returns:
            Parsed JSON response body once the job has succeeded.

        Raises:
            RuntimeError: If the job fails or reaches an unexpected status.
        """
        from azure.core.rest import HttpRequest

        for _ in range(60):
            poll_req = HttpRequest(method="GET", url=operation_url)
            poll_resp = await client.send_request(poll_req)
            poll_resp.raise_for_status()
            data = poll_resp.json()
            status = data.get("status", "").lower()
            if status == "succeeded":
                return data
            if status in ("failed", "cancelled", "cancelling"):
                errors = data.get("errors", [])
                raise RuntimeError(
                    f"Conversation PII job {status}: {errors}"
                )
            await asyncio_module.sleep(1)

        raise RuntimeError("Conversation PII job did not complete within the timeout")

    # ------------------------------------------------------------------
    # Client accessors (lazy construction)
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> Any:
        """Return (creating if necessary) the synchronous ConversationAnalysisClient."""
        if self.__sync_client is None:
            self.__sync_client = ConversationAnalysisClient(
                self._endpoint, self._credential
            )
        return self.__sync_client

    def _get_async_client(self) -> Any:
        """Return (creating if necessary) the async ConversationAnalysisClient."""
        if self.__async_client is None:
            self.__async_client = AsyncConversationAnalysisClient(
                self._endpoint, self._credential
            )
        return self.__async_client

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _get_human_message(
        self, state: AgentState[Any]
    ) -> Optional[HumanMessage]:
        """Extract the most recent HumanMessage from state."""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return msg
        return None

    def _get_ai_message(self, state: AgentState[Any]) -> Optional[AIMessage]:
        """Extract the most recent AIMessage from state."""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg
        return None

    def _get_text(self, msg: Optional[BaseMessage]) -> Optional[str]:
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


def _collect_pii_evaluations(
    result: Dict[str, Any],
) -> tuple[str, List[PIIEntityEvaluation]]:
    """Extract redacted text and PII entities from a Conversation PII job result.

    This is a module-level convenience wrapper around
    :meth:`AzureConversationPIIRedaction._extract_entities_from_result`.

    Args:
        result: The deserialized job result from the Azure Language Service.

    Returns:
        A tuple of ``(redacted_text, entities)``.
    """
    return AzureConversationPIIRedaction._extract_entities_from_result(result)
