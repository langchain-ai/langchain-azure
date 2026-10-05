"""Policy-based agent moderation using Azure AI Content Safety."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from azure.ai.contentsafety.models import (
    UnifiedModerateContext,
    UnifiedModerateOptions,
    UnifiedModerateResult,
)
from langchain.agents.middleware import AgentState, Runtime, ToolCallRequest
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.content import NonStandardAnnotation
from langgraph.types import Command

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.middleware.content_safety._base import (
    ContentSafetyAnnotationPayload,
    ContentSafetyEvaluation,
    ContentSafetyViolationError,
    _AzureContentSafetyBaseMiddleware,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedModerationEvaluation(ContentSafetyEvaluation):
    """Policy evaluation included in annotations and blocked-verdict errors.

    Attributes:
        category: Evaluation category, always ``"UnifiedModeration"``.
        source: Service intervention point.
        verdict: Enforced caller-facing verdict.
        reason: Optional policy reason.
        acs_verdict: Full Agent Control Specification verdict, without the
            moderated content.
    """

    category: str = "UnifiedModeration"
    source: str = ""
    verdict: str = ""
    reason: Optional[str] = None
    acs_verdict: Dict[str, Any] = field(default_factory=dict)


@experimental()
class AzureContentSafetyPolicyMiddleware(_AzureContentSafetyBaseMiddleware):
    """Enforce a Content Safety guardrail on a LangChain ``create_agent`` agent.

    Calls the Azure Content Safety Unified Moderate preview operation at the
    agent and model input/output hooks and around tool calls. All six hooks
    are submitted to the service, which controls the policy at each
    intervention point. A message may be evaluated at both the agent and
    model boundaries. Text-only message content is screened at these
    boundaries, while tool arguments and results are submitted as
    JSON-encoded content.

    A ``blocked`` service verdict raises :class:`ContentSafetyViolationError`
    before the content can reach the next stage. An ``allowed`` verdict applies
    returned content (including policy transforms). Non-blocking findings and
    transforms are attached as content-safety annotations to the corresponding
    message. Requests and malformed service responses fail explicitly.

    Example:
        .. code-block:: python

            from langchain.agents import create_agent
            from langchain_azure_ai.agents.middleware import (
                AzureContentSafetyPolicyMiddleware,
            )

            agent = create_agent(
                model=my_model,
                tools=my_tools,
                middleware=[
                    AzureContentSafetyPolicyMiddleware(
                        policy_id="my-guardrail",
                        endpoint="https://my-resource.cognitiveservices.azure.com",
                    )
                ],
            )

    Args:
        policy_id: ID of a guardrail policy configured on the Content Safety
            resource. Must be nonempty.
        endpoint: Content Safety resource URL; falls back to
            ``AZURE_CONTENT_SAFETY_ENDPOINT``. Mutually exclusive with
            ``project_endpoint``.
        credential: Token credential, Azure key credential, or key string.
            Defaults to ``DefaultAzureCredential``.
        project_endpoint: Foundry project endpoint; falls back to
            ``AZURE_AI_PROJECT_ENDPOINT`` when no resource URL is available.
        context: Optional SDK context (agent/session/correlation metadata)
            included with each moderation request.
        name: Middleware node-name prefix.

    Raises:
        ValueError: If the policy ID is empty, an endpoint is invalid, or the
            service response cannot be enforced safely.
        ContentSafetyViolationError: If the service blocks content.
    """

    def __init__(
        self,
        policy_id: str,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        *,
        project_endpoint: Optional[str] = None,
        context: Optional[UnifiedModerateContext] = None,
        name: str = "azure_unified_moderation",
    ) -> None:
        """Initialize the middleware with a policy enforced at all hooks.

        Args:
            policy_id: Required Content Safety guardrail policy ID.
            endpoint: Content Safety resource URL or environment fallback.
            credential: Azure credential or key; defaults to Azure identity.
            project_endpoint: Alternative Foundry project endpoint.
            context: Optional SDK agent/session context for all requests.
            name: Middleware node-name prefix.

        Raises:
            ValueError: If the policy ID is empty or endpoints are invalid.
        """
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("'policy_id' must be a nonempty string.")
        super().__init__(
            endpoint=endpoint,
            credential=credential,
            project_endpoint=project_endpoint,
            name=name,
        )
        self.policy_id = policy_id
        self.context = context

    def get_annotation_from_evaluations(
        self, evaluations: Sequence[ContentSafetyEvaluation]
    ) -> NonStandardAnnotation:
        """Build a standard annotation from policy evaluations.

        Args:
            evaluations: Evaluations returned by Unified Moderate.

        Returns:
            A content-safety annotation containing their policy verdicts.
        """
        return NonStandardAnnotation(
            type="non_standard_annotation",
            value=ContentSafetyAnnotationPayload(
                detection_type="unified_moderation",
                violations=[evaluation.to_dict() for evaluation in evaluations],
            ).to_dict(),
        )

    def get_evaluation_response(
        self, response: UnifiedModerateResult
    ) -> List[UnifiedModerationEvaluation]:
        """Extract a typed evaluation from the SDK response.

        Args:
            response: Result returned by the Unified Moderate SDK operation.

        Returns:
            A single evaluation with the complete ACS verdict.

        Raises:
            ValueError: If the service omitted a required verdict.
        """
        if (
            response is None
            or response.verdict not in ("allowed", "blocked")
            or response.acs_verdict is None
            or response.acs_verdict.decision not in ("allow", "deny", "transform")
        ):
            raise ValueError("Unified Moderate returned an invalid policy verdict.")
        return [
            UnifiedModerationEvaluation(
                verdict=str(response.verdict),
                reason=response.reason,
                acs_verdict=dict(response.acs_verdict),
            )
        ]

    def _enforce(
        self,
        response: UnifiedModerateResult,
        source: str,
        original: str,
    ) -> tuple[str, Optional[UnifiedModerationEvaluation]]:
        evaluation = self.get_evaluation_response(response)[0]
        evaluation = UnifiedModerationEvaluation(
            source=source,
            verdict=evaluation.verdict,
            reason=evaluation.reason,
            acs_verdict=evaluation.acs_verdict,
        )
        if response.verdict == "blocked":
            logger.info("[%s] Unified Moderate blocked %s", self.name, source)
            raise ContentSafetyViolationError(
                f"Unified moderation blocked {source}: "
                f"{response.reason or response.acs_verdict.reason or 'policy denied'}",
                [evaluation],
            )
        if response.acs_verdict.decision == "deny":
            raise ValueError("Unified Moderate allowed content with a deny decision.")
        if response.acs_verdict.decision == "transform" and response.content is None:
            raise ValueError("Unified Moderate omitted required transformed content.")
        content = response.content if response.content is not None else original
        if not isinstance(content, str):
            raise ValueError("Unified Moderate returned non-text content.")
        harm_results = response.acs_verdict.harm_results or {}
        annotate = (
            content != original
            or bool(response.acs_verdict.warnings)
            or any(harm.detected for harm in harm_results.values())
        )
        return content, evaluation if annotate else None

    def _options(
        self, source: str, content: str, **kwargs: Any
    ) -> UnifiedModerateOptions:
        return UnifiedModerateOptions(
            policy_id=self.policy_id,
            source=source,
            content=content,
            context=self.context,
            **kwargs,
        )

    def _moderate_sync(
        self, source: str, content: str, **kwargs: Any
    ) -> tuple[str, Optional[UnifiedModerationEvaluation]]:
        response = self._get_sync_client().unified_moderate(
            self._options(source, content, **kwargs)
        )
        return self._enforce(response, source, content)

    async def _moderate_async(
        self, source: str, content: str, **kwargs: Any
    ) -> tuple[str, Optional[UnifiedModerationEvaluation]]:
        response = await self._get_async_client().unified_moderate(
            self._options(source, content, **kwargs)
        )
        return self._enforce(response, source, content)

    def _message(
        self, state: AgentState[Any], *, is_input: bool, model: bool
    ) -> Optional[BaseMessage]:
        if model and is_input:
            return next(
                (
                    msg
                    for msg in reversed(state.get("messages", []))
                    if self.get_text_from_message(msg)
                ),
                None,
            )
        if is_input:
            return self.get_human_message_from_state(state)
        return self.get_ai_message_from_state(state)

    def _update_message(
        self,
        message: BaseMessage,
        original: str,
        content: str,
        evaluation: Optional[UnifiedModerationEvaluation],
    ) -> None:
        if content != original:
            if not isinstance(message.content, str) and any(
                not isinstance(block, dict) or block.get("type") != "text"
                for block in message.content
            ):
                raise ValueError(
                    "Cannot transform a message containing non-text blocks."
                )
            message.content = content
        if evaluation is not None:
            annotation = self.get_annotation_from_evaluations([evaluation])
            if isinstance(message.content, str):
                message.content = [
                    {"type": "text", "text": message.content},
                    dict(annotation),
                ]
            else:
                message.content = [*message.content, dict(annotation)]

    def _screen_message_sync(
        self, state: AgentState[Any], *, is_input: bool, model: bool
    ) -> None:
        message = self._message(state, is_input=is_input, model=model)
        content = self.get_text_from_message(message)
        if message is None or not content:
            return
        source = "input" if is_input else "output"
        moderated, evaluation = self._moderate_sync(source, content)
        self._update_message(message, content, moderated, evaluation)

    async def _screen_message_async(
        self, state: AgentState[Any], *, is_input: bool, model: bool
    ) -> None:
        message = self._message(state, is_input=is_input, model=model)
        content = self.get_text_from_message(message)
        if message is None or not content:
            return
        source = "input" if is_input else "output"
        moderated, evaluation = await self._moderate_async(source, content)
        self._update_message(message, content, moderated, evaluation)

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Moderate agent input before execution.

        Args:
            state: Current agent state containing the input message.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the input.
        """
        self._screen_message_sync(state, is_input=True, model=False)
        return None

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Asynchronously moderate agent input.

        Args:
            state: Current agent state containing the input message.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the input.
        """
        await self._screen_message_async(state, is_input=True, model=False)
        return None

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Moderate final agent output.

        Args:
            state: Current agent state containing the final response.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the output.
        """
        self._screen_message_sync(state, is_input=False, model=False)
        return None

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Asynchronously moderate final agent output.

        Args:
            state: Current agent state containing the final response.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the output.
        """
        await self._screen_message_async(state, is_input=False, model=False)
        return None

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Moderate the latest textual message sent to the model.

        Args:
            state: Current agent state containing model input messages.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the input.
        """
        self._screen_message_sync(state, is_input=True, model=True)
        return None

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Asynchronously moderate the latest textual model input.

        Args:
            state: Current agent state containing model input messages.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the input.
        """
        await self._screen_message_async(state, is_input=True, model=True)
        return None

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Moderate each model response.

        Args:
            state: Current agent state containing the model response.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the output.
        """
        self._screen_message_sync(state, is_input=False, model=True)
        return None

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Asynchronously moderate each model response.

        Args:
            state: Current agent state containing the model response.
            runtime: Current agent runtime.

        Returns:
            ``None``; allowed content is applied to the message.

        Raises:
            ContentSafetyViolationError: If the policy blocks the output.
        """
        await self._screen_message_async(state, is_input=False, model=True)
        return None

    @staticmethod
    def _tool_message(
        result: ToolMessage | Command[Any], tool_call_id: Optional[str]
    ) -> ToolMessage:
        if isinstance(result, ToolMessage):
            if tool_call_id is not None and result.tool_call_id != tool_call_id:
                raise ValueError(
                    "Tool result ID does not match the moderated tool call."
                )
            return result
        if tool_call_id is None:
            raise ValueError("Tool call ID is required to moderate a Command result.")
        update = result.update
        messages = update.get("messages", []) if isinstance(update, dict) else []
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        if not tool_messages:
            raise ValueError(
                "Unified moderation requires a ToolMessage for tool results."
            )
        if len(tool_messages) != 1 or tool_messages[0].tool_call_id != tool_call_id:
            raise ValueError(
                "Unified moderation cannot screen unrelated tool results in a Command."
            )
        return tool_messages[0]

    def _tool_result(
        self,
        result: ToolMessage | Command[Any],
        request: ToolCallRequest,
        pre_evaluation: Optional[UnifiedModerationEvaluation],
        *,
        moderated: Optional[str] = None,
        post_evaluation: Optional[UnifiedModerationEvaluation] = None,
    ) -> ToolMessage | Command[Any]:
        message = self._tool_message(result, request.tool_call["id"])
        if moderated is not None:
            try:
                replacement = json.loads(moderated)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Unified Moderate returned invalid JSON for a tool result."
                ) from exc
            if isinstance(replacement, (dict, int, float, bool)) or replacement is None:
                message.content = json.dumps(replacement)
            elif isinstance(replacement, str):
                message.content = replacement
            elif isinstance(replacement, list) and all(
                isinstance(item, (str, dict)) for item in replacement
            ):
                message.content = replacement
            else:
                raise ValueError(
                    "Unified Moderate returned an unsupported tool result."
                )
        for evaluation in (pre_evaluation, post_evaluation):
            if evaluation is not None:
                self._update_message(
                    message,
                    self.get_text_from_message(message) or "",
                    self.get_text_from_message(message) or "",
                    evaluation,
                )
        return result

    def _tool_pre(
        self, request: ToolCallRequest, content: str
    ) -> tuple[ToolCallRequest, Optional[UnifiedModerationEvaluation]]:
        moderated, evaluation = self._moderate_sync(
            "pre_tool_call",
            content,
            tool_name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )
        return self._apply_tool_arguments(request, content, moderated), evaluation

    async def _atool_pre(
        self, request: ToolCallRequest, content: str
    ) -> tuple[ToolCallRequest, Optional[UnifiedModerationEvaluation]]:
        moderated, evaluation = await self._moderate_async(
            "pre_tool_call",
            content,
            tool_name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )
        return self._apply_tool_arguments(request, content, moderated), evaluation

    @staticmethod
    def _apply_tool_arguments(
        request: ToolCallRequest, original: str, moderated: str
    ) -> ToolCallRequest:
        if moderated == original:
            return request
        try:
            arguments = json.loads(moderated)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Unified Moderate returned invalid JSON tool arguments."
            ) from exc
        if not isinstance(arguments, dict):
            raise ValueError("Unified Moderate tool arguments must be a JSON object.")
        return request.override(tool_call={**request.tool_call, "args": arguments})

    @staticmethod
    def _encode_tool_result(message: ToolMessage) -> str:
        if isinstance(message.content, str):
            try:
                structured = json.loads(message.content)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(structured, (dict, list)):
                    return message.content
        return json.dumps(message.content)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Moderate proposed arguments and completed tool results synchronously.

        Args:
            request: Tool call request, including structured arguments.
            handler: Executes the allowed, possibly transformed tool call.

        Returns:
            The tool message or Command, with allowed results applied.

        Raises:
            ContentSafetyViolationError: If the policy blocks either stage.
            ValueError: If a transformation or result cannot be enforced.
        """
        request, pre_evaluation = self._tool_pre(
            request, json.dumps(request.tool_call["args"])
        )
        started = perf_counter()
        result = handler(request)
        message = self._tool_message(result, request.tool_call["id"])
        original_result = self._encode_tool_result(message)
        moderated, post_evaluation = self._moderate_sync(
            "post_tool_call",
            original_result,
            tool_name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            tool_arguments=json.dumps(request.tool_call["args"]),
            tool_result_is_error=message.status == "error",
            tool_duration_ms=(perf_counter() - started) * 1000,
        )
        return self._tool_result(
            result,
            request,
            pre_evaluation,
            moderated=moderated if moderated != original_result else None,
            post_evaluation=post_evaluation,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Moderate proposed arguments and completed tool results asynchronously.

        Args:
            request: Tool call request, including structured arguments.
            handler: Executes the allowed, possibly transformed tool call.

        Returns:
            The tool message or Command, with allowed results applied.

        Raises:
            ContentSafetyViolationError: If the policy blocks either stage.
            ValueError: If a transformation or result cannot be enforced.
        """
        request, pre_evaluation = await self._atool_pre(
            request, json.dumps(request.tool_call["args"])
        )
        started = perf_counter()
        result = await handler(request)
        message = self._tool_message(result, request.tool_call["id"])
        original_result = self._encode_tool_result(message)
        moderated, post_evaluation = await self._moderate_async(
            "post_tool_call",
            original_result,
            tool_name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            tool_arguments=json.dumps(request.tool_call["args"]),
            tool_result_is_error=message.status == "error",
            tool_duration_ms=(perf_counter() - started) * 1000,
        )
        return self._tool_result(
            result,
            request,
            pre_evaluation,
            moderated=moderated if moderated != original_result else None,
            post_evaluation=post_evaluation,
        )
