"""Tests for policy-based Azure Content Safety middleware."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.ai.contentsafety.models import (
    AcsHarmResult,
    AcsVerdict,
    UnifiedModerateResult,
)
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain_azure_ai.agents.middleware import (
    AzureContentSafetyPolicyMiddleware,
    UnifiedModerationEvaluation,
)
from langchain_azure_ai.agents.middleware.content_safety import (
    ContentSafetyViolationError,
    get_content_safety_annotations,
)

ENDPOINT = "https://test.cognitiveservices.azure.com"
RUNTIME: Runtime[Any] = Runtime(context=None)


def _result(
    verdict: str = "allowed",
    *,
    decision: str = "allow",
    content: str | None = None,
    warnings: list[str] | None = None,
    detected: bool = False,
) -> UnifiedModerateResult:
    return UnifiedModerateResult(
        verdict=verdict,
        reason="policy_reason" if verdict == "blocked" else None,
        content=content,
        acs_verdict=AcsVerdict(
            decision=decision,
            warnings=warnings,
            harm_results={
                "CustomHarm": AcsHarmResult(blocked=False, detected=detected)
            },
        ),
    )


@pytest.fixture
def middleware() -> AzureContentSafetyPolicyMiddleware:
    return AzureContentSafetyPolicyMiddleware(
        policy_id="my-guardrail", endpoint=ENDPOINT, credential="test-key"
    )


@pytest.mark.parametrize(
    ("hook", "message", "source"),
    [
        ("before_agent", HumanMessage(content="input"), "input"),
        ("after_agent", AIMessage(content="output"), "output"),
        (
            "before_model",
            ToolMessage(content="tool data", tool_call_id="call"),
            "input",
        ),
        (
            "after_model",
            AIMessage(content="model output"),
            "output",
        ),
    ],
)
def test_message_hooks(
    middleware: AzureContentSafetyPolicyMiddleware,
    hook: str,
    message: HumanMessage | AIMessage | ToolMessage,
    source: str,
) -> None:
    client = MagicMock()
    client.unified_moderate.return_value = _result()
    with patch.object(middleware, "_get_sync_client", return_value=client):
        assert (
            getattr(middleware, hook)({"messages": [message]}, runtime=RUNTIME) is None
        )
        options = client.unified_moderate.call_args.args[0]
        assert dict(options)["policyId"] == "my-guardrail"
        assert options.source == source
        assert options.content == message.content
        assert options.tool_name is None
        client.unified_moderate.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hook", "message", "source"),
    [
        ("abefore_agent", HumanMessage(content="input"), "input"),
        ("aafter_agent", AIMessage(content="output"), "output"),
        (
            "abefore_model",
            ToolMessage(content="tool data", tool_call_id="call"),
            "input",
        ),
        (
            "aafter_model",
            AIMessage(content="output"),
            "output",
        ),
    ],
)
async def test_async_message_hooks(
    middleware: AzureContentSafetyPolicyMiddleware,
    hook: str,
    message: HumanMessage | AIMessage | ToolMessage,
    source: str,
) -> None:
    client = MagicMock(unified_moderate=AsyncMock(return_value=_result()))
    with patch.object(middleware, "_get_async_client", return_value=client):
        assert (
            await getattr(middleware, hook)({"messages": [message]}, runtime=RUNTIME)
            is None
        )
        assert client.unified_moderate.call_args.args[0].source == source
        client.unified_moderate.assert_awaited_once()


@pytest.mark.parametrize(
    "hook", ["before_agent", "after_agent", "before_model", "after_model"]
)
def test_blocked_message_stops_execution(
    middleware: AzureContentSafetyPolicyMiddleware, hook: str
) -> None:
    message = (
        HumanMessage(content="request")
        if "before" in hook
        else AIMessage(content="reply")
    )
    client = MagicMock(
        unified_moderate=MagicMock(return_value=_result("blocked", decision="deny"))
    )
    with patch.object(middleware, "_get_sync_client", return_value=client):
        with pytest.raises(ContentSafetyViolationError) as exc:
            getattr(middleware, hook)({"messages": [message]}, runtime=RUNTIME)
    assert exc.value.violations[0].category == "UnifiedModeration"
    assert "policy_reason" in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook", ["abefore_agent", "aafter_agent", "abefore_model", "aafter_model"]
)
async def test_async_blocked_message_stops_execution(
    middleware: AzureContentSafetyPolicyMiddleware, hook: str
) -> None:
    message = (
        HumanMessage(content="request")
        if "before" in hook
        else AIMessage(content="reply")
    )
    client = MagicMock(
        unified_moderate=AsyncMock(return_value=_result("blocked", decision="deny"))
    )
    with patch.object(middleware, "_get_async_client", return_value=client):
        with pytest.raises(ContentSafetyViolationError):
            await getattr(middleware, hook)({"messages": [message]}, runtime=RUNTIME)


def test_allowed_transform_and_annotation(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    message = HumanMessage(content="original")
    client = MagicMock(
        unified_moderate=MagicMock(
            return_value=_result(
                content="safe replacement",
                decision="transform",
                warnings=["reviewed"],
                detected=True,
            )
        )
    )
    with patch.object(middleware, "_get_sync_client", return_value=client):
        middleware.before_agent({"messages": [message]}, runtime=RUNTIME)
    assert message.text == "safe replacement"
    annotation = get_content_safety_annotations(message)[0]
    evaluation = annotation.violations[0]
    assert evaluation["source"] == "input"
    assert evaluation["acs_verdict"]["harmResults"]["CustomHarm"]["detected"]
    assert evaluation["acs_verdict"]["warnings"] == ["reviewed"]
    assert isinstance(
        middleware.get_evaluation_response(_result())[0], UnifiedModerationEvaluation
    )


def test_allowed_detection_is_not_blocked(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    message = AIMessage(content="allowed with warning")
    client = MagicMock(unified_moderate=MagicMock(return_value=_result(detected=True)))
    with patch.object(middleware, "_get_sync_client", return_value=client):
        middleware.after_model({"messages": [message]}, runtime=RUNTIME)
    assert len(get_content_safety_annotations(message)) == 1


@pytest.mark.parametrize(
    "result",
    [
        _result("unknown"),
        _result("allowed", decision="deny"),
        _result("allowed", decision="transform"),
        MagicMock(
            verdict="allowed", content=123, acs_verdict=AcsVerdict(decision="allow")
        ),
    ],
)
def test_invalid_verdict_fails_closed(
    middleware: AzureContentSafetyPolicyMiddleware, result: UnifiedModerateResult
) -> None:
    with patch.object(
        middleware,
        "_get_sync_client",
        return_value=MagicMock(unified_moderate=MagicMock(return_value=result)),
    ):
        with pytest.raises(ValueError):
            middleware.before_agent(
                {"messages": [HumanMessage(content="request")]}, runtime=RUNTIME
            )


def _request(args: dict[str, object] | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "lookup", "args": args or {"query": "unsafe"}, "id": "call"},
        tool=None,
        state={"messages": []},
        runtime=None,  # type: ignore[arg-type]
    )


def test_tool_hooks_transform_and_annotate(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock()
    client.unified_moderate.side_effect = [
        _result(content='{"query":"safe"}', decision="transform"),
        _result(content='"redacted"', decision="transform", warnings=["redacted"]),
    ]
    handler = MagicMock(
        side_effect=lambda req: ToolMessage(
            content=f"result for {req.tool_call['args']['query']}", tool_call_id="call"
        )
    )
    with patch.object(middleware, "_get_sync_client", return_value=client):
        result = middleware.wrap_tool_call(_request(), handler)
    assert isinstance(result, ToolMessage)
    assert result.text == "redacted"
    assert len(get_content_safety_annotations(result)) == 2
    assert handler.call_args.args[0].tool_call["args"] == {"query": "safe"}
    pre, post = (call.args[0] for call in client.unified_moderate.call_args_list)
    assert pre.source == "pre_tool_call"
    assert json.loads(pre.content) == {"query": "unsafe"}
    assert post.source == "post_tool_call"
    assert json.loads(post.content) == "result for safe"
    assert json.loads(post.tool_arguments) == {"query": "safe"}
    assert post.tool_result_is_error is False
    assert post.tool_duration_ms >= 0


@pytest.mark.asyncio
async def test_async_tool_hooks_and_command(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock()
    client.unified_moderate = AsyncMock(
        side_effect=[
            _result(warnings=["pre"]),
            _result(content='"clean"', decision="transform"),
        ]
    )
    tool_message = ToolMessage(content="unclean", tool_call_id="call", status="error")
    command = Command(update={"messages": [tool_message], "saved": 1}, goto="model")
    handler = AsyncMock(return_value=command)
    with patch.object(middleware, "_get_async_client", return_value=client):
        assert await middleware.awrap_tool_call(_request(), handler) is command
    assert tool_message.text == "clean"
    assert isinstance(command.update, dict)
    assert command.update["saved"] == 1
    assert command.goto == "model"
    assert len(get_content_safety_annotations(tool_message)) == 2
    post = client.unified_moderate.call_args_list[1].args[0]
    assert post.tool_result_is_error is True


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_pre_tool_block_prevents_handler(
    middleware: AzureContentSafetyPolicyMiddleware, async_call: bool
) -> None:
    client = MagicMock(
        unified_moderate=MagicMock(return_value=_result("blocked", decision="deny"))
    )
    async_client = MagicMock(
        unified_moderate=AsyncMock(return_value=_result("blocked", decision="deny"))
    )
    handler = MagicMock()
    async_handler = AsyncMock()
    with (
        patch.object(middleware, "_get_sync_client", return_value=client),
        patch.object(middleware, "_get_async_client", return_value=async_client),
    ):
        with pytest.raises(ContentSafetyViolationError):
            if async_call:
                await middleware.awrap_tool_call(_request(), async_handler)
            else:
                middleware.wrap_tool_call(_request(), handler)
    handler.assert_not_called()
    async_handler.assert_not_awaited()


def test_post_tool_block_prevents_result(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock(
        unified_moderate=MagicMock(
            side_effect=[_result(), _result("blocked", decision="deny")]
        )
    )
    with patch.object(middleware, "_get_sync_client", return_value=client):
        with pytest.raises(ContentSafetyViolationError):
            middleware.wrap_tool_call(
                _request(),
                lambda _: ToolMessage(content="unsafe result", tool_call_id="call"),
            )


@pytest.mark.asyncio
async def test_async_post_tool_block_prevents_result(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock(
        unified_moderate=AsyncMock(
            side_effect=[_result(), _result("blocked", decision="deny")]
        )
    )
    with patch.object(middleware, "_get_async_client", return_value=client):
        with pytest.raises(ContentSafetyViolationError):
            await middleware.awrap_tool_call(
                _request(),
                AsyncMock(
                    return_value=ToolMessage(content="unsafe", tool_call_id="call")
                ),
            )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"safe":true}', '{"safe": true}'),
        ('["safe"]', ["safe"]),
        ("invalid", None),
        ("[1]", None),
    ],
)
def test_transformed_tool_results_validate_json(
    middleware: AzureContentSafetyPolicyMiddleware,
    content: str,
    expected: str | list[str] | None,
) -> None:
    client = MagicMock(
        unified_moderate=MagicMock(
            side_effect=[_result(), _result(content=content, decision="transform")]
        )
    )
    with patch.object(middleware, "_get_sync_client", return_value=client):
        if expected is None:
            with pytest.raises(ValueError):
                middleware.wrap_tool_call(
                    _request(),
                    lambda _: ToolMessage(content="original", tool_call_id="call"),
                )
        else:
            result = middleware.wrap_tool_call(
                _request(),
                lambda _: ToolMessage(content="original", tool_call_id="call"),
            )
            assert isinstance(result, ToolMessage)
            if isinstance(expected, str):
                assert result.text == expected
            else:
                assert result.content[0] == "safe"


def test_invalid_json_tool_arguments(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    with patch.object(
        middleware,
        "_get_sync_client",
        return_value=MagicMock(
            unified_moderate=MagicMock(
                return_value=_result(content="{invalid", decision="transform")
            )
        ),
    ):
        with pytest.raises(ValueError, match="invalid JSON tool arguments"):
            middleware.wrap_tool_call(_request(), MagicMock())


def test_tool_command_without_result_fails_closed(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock(unified_moderate=MagicMock(return_value=_result()))
    with patch.object(middleware, "_get_sync_client", return_value=client):
        with pytest.raises(ValueError, match="requires a ToolMessage"):
            middleware.wrap_tool_call(
                _request(), lambda _: Command(update={"saved": 1})
            )


@pytest.mark.parametrize(
    "result",
    [
        ToolMessage(content="wrong", tool_call_id="different"),
        Command(
            update={
                "messages": [
                    ToolMessage(content="first", tool_call_id="call"),
                    ToolMessage(content="second", tool_call_id="different"),
                ]
            }
        ),
    ],
)
def test_unrelated_tool_results_fail_closed(
    middleware: AzureContentSafetyPolicyMiddleware,
    result: ToolMessage | Command[Any],
) -> None:
    client = MagicMock(unified_moderate=MagicMock(return_value=_result()))
    with patch.object(middleware, "_get_sync_client", return_value=client):
        with pytest.raises(ValueError, match="[Tt]ool results?"):
            middleware.wrap_tool_call(_request(), lambda _: result)


def test_service_error_propagates(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock(
        unified_moderate=MagicMock(side_effect=RuntimeError("unavailable"))
    )
    with patch.object(middleware, "_get_sync_client", return_value=client):
        with pytest.raises(RuntimeError, match="unavailable"):
            middleware.before_agent(
                {"messages": [HumanMessage(content="request")]}, runtime=RUNTIME
            )


def test_structured_tool_result_is_not_double_encoded(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    client = MagicMock(unified_moderate=MagicMock(return_value=_result()))
    with patch.object(middleware, "_get_sync_client", return_value=client):
        result = middleware.wrap_tool_call(
            _request(),
            lambda _: ToolMessage(content='{"records":[1,2]}', tool_call_id="call"),
        )
    assert isinstance(result, ToolMessage)
    assert result.content == '{"records":[1,2]}'
    assert [call.args[0].source for call in client.unified_moderate.call_args_list] == [
        "pre_tool_call",
        "post_tool_call",
    ]
    options = client.unified_moderate.call_args.args[0]
    assert json.loads(options.content) == {"records": [1, 2]}


def test_requires_policy_id() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        AzureContentSafetyPolicyMiddleware("", endpoint=ENDPOINT)


@pytest.mark.parametrize(
    "hook_option",
    [
        "apply_to_input",
        "apply_to_output",
        "apply_to_model_input",
        "apply_to_model_output",
        "apply_to_tool_call",
        "apply_to_tool_result",
    ],
)
def test_hook_selection_is_not_configurable(hook_option: str) -> None:
    invalid_kwargs: dict[str, Any] = {hook_option: False}
    with pytest.raises(TypeError, match=hook_option):
        AzureContentSafetyPolicyMiddleware(
            policy_id="my-guardrail",
            endpoint=ENDPOINT,
            **invalid_kwargs,
        )


def test_create_agent_calls_all_applicable_hooks(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    client = MagicMock(unified_moderate=MagicMock(return_value=_result()))
    with patch.object(middleware, "_get_sync_client", return_value=client):
        agent = create_agent(
            model=FakeListChatModel(responses=["ok"]), middleware=[middleware]
        )
        result = agent.invoke({"messages": [HumanMessage(content="hello")]})
    assert result["messages"][-1].text == "ok"
    assert [call.args[0].source for call in client.unified_moderate.call_args_list] == [
        "input",
        "input",
        "output",
        "output",
    ]


def test_create_agent_executes_tool_hooks(
    middleware: AzureContentSafetyPolicyMiddleware,
) -> None:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.tools import tool

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(
            self, tools: object, **kwargs: object
        ) -> FakeMessagesListChatModel:
            return self

    @tool
    def lookup(query: str) -> str:
        """Return a sample lookup result."""
        return query

    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"query": "hello"}, "id": "call"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    client = MagicMock(unified_moderate=MagicMock(return_value=_result()))
    with patch.object(middleware, "_get_sync_client", return_value=client):
        result = create_agent(
            model=model, tools=[lookup], middleware=[middleware]
        ).invoke({"messages": [HumanMessage(content="hello")]})
    assert result["messages"][-1].text == "done"
    assert [call.args[0].source for call in client.unified_moderate.call_args_list] == [
        "input",
        "input",
        "pre_tool_call",
        "post_tool_call",
        "input",
        "output",
        "output",
    ]
