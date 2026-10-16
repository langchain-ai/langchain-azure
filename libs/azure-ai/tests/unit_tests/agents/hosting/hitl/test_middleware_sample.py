"""Exercise the middleware sample through the unmodified Responses host."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, Mock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
from openai import OpenAI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    FoundryCheckpointSaver,
    ResponsesHostServer,
)

from ..test_foundry_checkpoint_saver import _credential, _FakeStateStore  # noqa: E402
from .conftest import assistant_text, sentinels  # noqa: E402

SAMPLE_DIRECTORY = (
    Path(__file__).resolve().parents[7]
    / "samples/hosting/langgraph-hosted-agents/responses/11_middleware"
)
ORIGINAL_ARGUMENTS = {
    "recipient": "alice@example.com",
    "subject": "Project update",
    "body": "The review starts at 10:00.",
}


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingModel:
        return self


def _load_sample(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    directory: Path = SAMPLE_DIRECTORY,
) -> ModuleType:
    path = directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"middleware_sample_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample_agent(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_sample("agent", monkeypatch)


@pytest.fixture
def sample_client(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_sample("client", monkeypatch)


@pytest.fixture
def sample_e2e_runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_sample(
        "run_samples_e2e",
        monkeypatch,
        directory=SAMPLE_DIRECTORY.parents[1] / "tests",
    )


@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def _record_send(sample_agent: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Mock:
    send = Mock(wraps=sample_agent.send_email.func)

    def recorded_send(recipient: str, subject: str, body: str) -> str:
        return str(send(recipient=recipient, subject=subject, body=body))

    monkeypatch.setattr(sample_agent.send_email, "func", recorded_send)
    return send


def _model(*, actions: int = 1) -> ToolCallingModel:
    return ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-email-{index}",
                        "name": "send_email",
                        "args": {
                            **ORIGINAL_ARGUMENTS,
                            "subject": f"Project update {index}",
                        },
                    }
                    for index in range(actions)
                ],
            ),
            AIMessage(content="Review finished."),
        ]
    )


def _snapshot(graph: Any, thread_id: str) -> Any:
    return asyncio.run(graph.aget_state({"configurable": {"thread_id": thread_id}}))


@pytest.fixture
def sample_main(
    sample_agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    monkeypatch.syspath_prepend(str(SAMPLE_DIRECTORY))
    monkeypatch.setitem(sys.modules, "agent", sample_agent)
    return _load_sample("main", monkeypatch)


@pytest.mark.parametrize("capture", [None, "false", "true"])
def test_content_capture_default_precedes_disabled_tracing(
    sample_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capture: str | None,
) -> None:
    variable = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    if capture is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, capture)
    sample_main._configure_tracing(AsyncExitStack(), Mock())
    assert sample_main.os.environ.get(variable) == (capture or "false")


def test_readme_environment_template_is_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", ".env.example"],
        cwd=SAMPLE_DIRECTORY,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("is_hosted", [False, True])
@pytest.mark.parametrize(
    "failure",
    [None, "tracing", "endpoint", "model", "checkpoint", "graph", "server", "cancel"],
)
async def test_startup_owns_resource_lifetimes(
    sample_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    is_hosted: bool,
    failure: str | None,
) -> None:
    resources = {
        name: MagicMock()
        for name in (
            "credential",
            "project",
            "endpoint",
            "sync_http",
            "async_http",
            "saver",
        )
    }
    for resource in resources.values():
        resource.__enter__.return_value = resource
        resource.__aenter__.return_value = resource
    resources["endpoint"].base_url = "https://test/openai/v1/"
    resources["project"].get_openai_client.return_value = resources["endpoint"]
    factories = {
        "DefaultAzureCredential": "credential",
        "AIProjectClient": "project",
        "DefaultHttpxClient": "sync_http",
        "DefaultAsyncHttpxClient": "async_http",
        "FoundryCheckpointSaver": "saver",
    }
    for factory, resource_name in factories.items():
        monkeypatch.setattr(
            sample_main, factory, Mock(return_value=resources[resource_name])
        )
    monkeypatch.setattr(sample_main, "load_dotenv", Mock())
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://test/api/projects/sample")
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "sample-model")
    monkeypatch.setenv("PORT", "8099")
    monkeypatch.setattr(
        sample_main.AgentConfig,
        "from_env",
        lambda: SimpleNamespace(is_hosted=is_hosted),
    )
    model = _model()
    model_factory = Mock(return_value=model)
    graph = object()
    graph_factory = Mock(return_value=graph)
    monkeypatch.setattr(sample_main, "ChatOpenAI", model_factory)
    monkeypatch.setattr(sample_main, "build_agent", graph_factory)
    tracing = Mock()
    if failure == "tracing":
        tracing.side_effect = RuntimeError("tracing failure")
    monkeypatch.setattr(sample_main, "_configure_tracing", tracing)
    server = Mock()

    async def serve(**_: Any) -> None:
        resources["credential"].__exit__.assert_not_called()
        resources["sync_http"].__exit__.assert_not_called()
        resources["async_http"].__aexit__.assert_not_awaited()
        resources["endpoint"].__exit__.assert_called_once()
        resources["project"].__exit__.assert_called_once()
        if is_hosted:
            resources["saver"].__aexit__.assert_not_awaited()
        if failure == "cancel":
            raise asyncio.CancelledError
        if failure == "server":
            raise RuntimeError("server failure")

    server.run_async = AsyncMock(side_effect=serve)
    host_factory = Mock(return_value=server)
    monkeypatch.setattr(sample_main, "ResponsesHostServer", host_factory)
    if failure == "endpoint":
        resources["project"].get_openai_client.side_effect = RuntimeError(
            "endpoint failure"
        )
    elif failure == "model":
        model_factory.side_effect = RuntimeError("model failure")
    elif failure == "checkpoint":
        if is_hosted:
            resources["saver"].__aenter__.side_effect = RuntimeError(
                "checkpoint failure"
            )
        else:
            monkeypatch.setattr(
                sample_main,
                "InMemorySaver",
                Mock(side_effect=RuntimeError("checkpoint failure")),
            )
    elif failure == "graph":
        graph_factory.side_effect = RuntimeError("graph failure")
    if failure is None:
        await sample_main.main()
    else:
        expected = asyncio.CancelledError if failure == "cancel" else RuntimeError
        with pytest.raises(expected):
            await sample_main.main()
    resources["credential"].__exit__.assert_called_once()
    tracing.assert_called_once_with(ANY, resources["credential"])
    if failure == "tracing":
        resources["project"].__exit__.assert_not_called()
        resources["sync_http"].__exit__.assert_not_called()
        resources["async_http"].__aexit__.assert_not_awaited()
        return
    resources["project"].__exit__.assert_called_once()
    if failure != "endpoint":
        resources["endpoint"].__exit__.assert_called_once()
        resources["sync_http"].__exit__.assert_called_once()
        resources["async_http"].__aexit__.assert_awaited_once()
    if is_hosted and failure not in {"endpoint", "model", "checkpoint"}:
        resources["saver"].__aexit__.assert_awaited_once()
    if failure not in {"endpoint", "model", "checkpoint", "graph"}:
        host_factory.assert_called_once_with(graph)
        server.run_async.assert_awaited_once_with(port=8099)
        assert model_factory.call_args.kwargs["model"] == "sample-model"
        assert model_factory.call_args.kwargs["use_responses_api"] is True
        assert model_factory.call_args.kwargs["http_client"] is resources["sync_http"]
        assert (
            model_factory.call_args.kwargs["http_async_client"]
            is resources["async_http"]
        )
        if is_hosted:
            assert graph_factory.call_args.args[1] is resources["saver"]
        else:
            assert isinstance(graph_factory.call_args.args[1], InMemorySaver)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "destination",
    [
        "disabled",
        "azure",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    ],
)
@pytest.mark.parametrize("failure", [False, True])
async def test_tracing_setup_respects_destination_and_ownership(
    sample_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
    failure: bool,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "OTEL_SDK_DISABLED", "true" if destination == "disabled" else "false"
    )
    if destination.startswith("OTEL_EXPORTER"):
        monkeypatch.setenv(destination, "http://localhost:4318")
    provider = Mock()
    credential = Mock()
    monkeypatch.setattr(
        sample_main.trace,
        "get_tracer_provider",
        lambda: sample_main.trace.ProxyTracerProvider(),
    )
    exporter = Mock()
    processor = Mock()
    enable = Mock()
    if failure and destination != "disabled":
        enable.side_effect = RuntimeError("tracing setup failure")
    monkeypatch.setattr(sample_main, "TracerProvider", Mock(return_value=provider))
    monkeypatch.setattr(sample_main, "OTLPSpanExporter", Mock(return_value=exporter))
    monkeypatch.setattr(sample_main, "BatchSpanProcessor", Mock(return_value=processor))
    monkeypatch.setattr(sample_main, "enable_auto_tracing", enable)
    set_provider = Mock()
    monkeypatch.setattr(sample_main.trace, "set_tracer_provider", set_provider)
    if failure and destination != "disabled":
        with pytest.raises(RuntimeError, match="tracing setup"):
            async with AsyncExitStack() as resources:
                sample_main._configure_tracing(resources, credential)
    else:
        async with AsyncExitStack() as resources:
            sample_main._configure_tracing(resources, credential)
            provider.shutdown.assert_not_called()
    if destination == "disabled":
        enable.assert_not_called()
        set_provider.assert_not_called()
        provider.shutdown.assert_not_called()
    elif destination == "azure":
        enable.assert_called_once_with(
            credential=credential, auto_configure_azure_monitor=True
        )
        set_provider.assert_not_called()
        provider.shutdown.assert_not_called()
    else:
        provider.add_span_processor.assert_called_once_with(processor)
        set_provider.assert_called_once_with(provider)
        enable.assert_called_once_with()
        provider.shutdown.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_platform_tracing_provider_is_not_replaced_or_closed(
    sample_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    provider = Mock()
    monkeypatch.setattr(sample_main.trace, "get_tracer_provider", lambda: provider)
    factory = Mock()
    monkeypatch.setattr(sample_main, "TracerProvider", factory)
    set_provider = Mock()
    monkeypatch.setattr(sample_main.trace, "set_tracer_provider", set_provider)
    enable = Mock(side_effect=RuntimeError("tracing setup") if failure else None)
    monkeypatch.setattr(sample_main, "enable_auto_tracing", enable)
    if failure:
        with pytest.raises(RuntimeError, match="tracing setup"):
            async with AsyncExitStack() as resources:
                sample_main._configure_tracing(resources, Mock())
    else:
        async with AsyncExitStack() as resources:
            sample_main._configure_tracing(resources, Mock())
    enable.assert_called_once_with(auto_configure_azure_monitor=False)
    factory.assert_not_called()
    set_provider.assert_not_called()
    provider.shutdown.assert_not_called()


@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
def test_native_agent_without_host(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    send = _record_send(sample_agent, monkeypatch)
    graph = sample_agent.build_agent(_model(), InMemorySaver())
    config = {"configurable": {"thread_id": "native"}}
    paused = graph.invoke(
        {"messages": [{"role": "user", "content": "Send it"}]}, config
    )
    assert send.call_count == 0
    interrupt = paused["__interrupt__"][0]
    pending = {
        "output": [
            {
                "type": "function_call",
                "name": "__hosted_agent_adapter_interrupt__",
                "call_id": interrupt.id,
                "arguments": json.dumps({"value": interrupt.value}),
            }
        ]
    }
    replies = sample_client.build_resume_items(pending, decision, "bob@example.com")
    resumed = graph.invoke(Command(**json.loads(replies[0]["output"])), config)
    assert resumed["messages"][-1].content == "Review finished."
    assert send.call_count == (0 if decision == "reject" else 1)
    if decision != "reject":
        assert send.call_args.kwargs["recipient"] == (
            "bob@example.com" if decision == "edit" else "alice@example.com"
        )


@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("continuation", ["conversation", "previous_response_id"])
def test_middleware_survives_responses_transport(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    stream: bool,
    continuation: str,
) -> None:
    send = _record_send(sample_agent, monkeypatch)
    graph = sample_agent.build_agent(_model(), InMemorySaver())
    options: dict[str, Any] = {}
    if continuation == "conversation":
        options["conversation"] = {"id": "middleware-conversation"}
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            first = sample_client.create_response(
                client, input="Send it", stream=stream, **options
            )
            pending = sentinels(first)
            assert len(pending) == 1, first
            assert send.call_count == 0
            review = json.loads(pending[0]["arguments"])["value"]
            assert review["action_requests"][0]["name"] == "send_email"
            assert review["review_configs"][0]["allowed_decisions"] == [
                "approve",
                "edit",
                "reject",
            ]
            thread_id = options.get("conversation", {}).get("id", first["id"])
            if continuation == "previous_response_id":
                options["previous_response_id"] = first["id"]
            finished = sample_client.create_response(
                client,
                input=sample_client.build_resume_items(
                    first, decision, "bob@example.com"
                ),
                stream=stream,
                **options,
            )
    assert finished["status"] == "completed", finished
    assert not sentinels(finished)
    assert "Review finished." in assistant_text(finished)
    assert send.call_count == (0 if decision == "reject" else 1)
    state = _snapshot(graph, thread_id).values
    results = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(results) == 1
    if decision == "reject":
        assert results[0].status == "error"
        assert "Rejected by the reviewer" in results[0].content
    else:
        recipient = "bob@example.com" if decision == "edit" else "alice@example.com"
        assert send.call_args.kwargs["recipient"] == recipient
        assert recipient in results[0].content
    assert all(
        "__hosted_agent_adapter_interrupt__" not in str(message.content)
        for message in state["messages"]
    )


@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
@pytest.mark.parametrize("stream", [False, True])
def test_middleware_shared_e2e_runner(
    sample_agent: ModuleType,
    sample_e2e_runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    stream: bool,
) -> None:
    checks = sample_e2e_runner._filter(
        sample_e2e_runner._build_registry(), ["responses/11"]
    )
    assert {check.name for check in checks} == {
        f"{choice}_{mode}"
        for choice in ("approve", "edit", "reject")
        for mode in ("json", "sse")
    }
    name = f"{decision}_{'sse' if stream else 'json'}"
    check = next(check for check in checks if check.name == name)
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://foundry.test")
    send = _record_send(sample_agent, monkeypatch)
    graph = sample_agent.build_agent(_model(), InMemorySaver())
    server = SimpleNamespace(url="http://testserver", terminate=Mock())
    start = Mock(return_value=server)
    monkeypatch.setattr(sample_e2e_runner, "start_sample", start)
    monkeypatch.setattr(sample_agent.logger, "disabled", True)
    with TestClient(ResponsesHostServer(graph).app) as transport:
        monkeypatch.setattr(sample_e2e_runner.httpx, "post", transport.post)
        monkeypatch.setattr(sample_e2e_runner.httpx, "stream", transport.stream)
        check.fn()
    start.assert_called_once_with("responses/11_middleware/main.py")
    server.terminate.assert_called_once_with()
    exchanges = sample_e2e_runner._RAW_API_EXCHANGES
    assert len(exchanges) == 2
    assert all(exchange.stream is stream for exchange in exchanges)
    assert "previous_response_id" in exchanges[1].request_json
    assert send.call_count == (0 if decision == "reject" else 1)
    if decision != "reject":
        assert send.call_args.kwargs["recipient"] == (
            "bob@example.com" if decision == "edit" else "alice@example.com"
        )


@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
def test_client_answers_every_action_in_order(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    send = _record_send(sample_agent, monkeypatch)
    graph = sample_agent.build_agent(_model(actions=2), InMemorySaver())
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            first = sample_client.create_response(client, input="Send two emails")
            assert send.call_count == 0
            replies = sample_client.build_resume_items(
                first, decision, "bob@example.com"
            )
            call_id = replies[0]["call_id"]
            decisions = json.loads(replies[0]["output"])["resume"][call_id]["decisions"]
            assert len(decisions) == 2
            finished = sample_client.create_response(
                client, input=replies, previous_response_id=first["id"]
            )
    assert not sentinels(finished), finished
    assert send.call_count == (0 if decision == "reject" else 2)
    if decision == "edit":
        assert [item["edited_action"]["args"]["subject"] for item in decisions] == [
            "Project update 0",
            "Project update 1",
        ]
        assert all(
            call.kwargs["recipient"] == "bob@example.com"
            for call in send.call_args_list
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
def test_reviewer_prompts_for_pending_review(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stream: bool,
    decision: str,
) -> None:
    send = _record_send(sample_agent, monkeypatch)
    graph = sample_agent.build_agent(_model(), InMemorySaver())
    ask = Mock(side_effect=[decision, "bob@example.com"])
    monkeypatch.setattr("builtins.input", ask)
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            sample_client.review_agent(
                client,
                SimpleNamespace(
                    prompt="Send it", stream=stream, decision=None, recipient=None
                ),
            )
    assert ask.call_count == (2 if decision == "edit" else 1)
    assert send.call_count == (0 if decision == "reject" else 1)
    if decision != "reject":
        assert send.call_args.kwargs["recipient"] == (
            "bob@example.com" if decision == "edit" else "alice@example.com"
        )
    assert "Review finished." in capsys.readouterr().out


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("decision", [None, "approve"])
@pytest.mark.parametrize("scenario", ["direct_answer", "message_limit"])
def test_reviewer_finishes_without_pending_review(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stream: bool,
    decision: str | None,
    scenario: str,
) -> None:
    send = _record_send(sample_agent, monkeypatch)
    model = (
        _model()
        if scenario == "message_limit"
        else ToolCallingModel(responses=[AIMessage(content="No action needed.")])
    )
    prompt: Any = (
        [{"role": "user", "content": f"Message {index}"} for index in range(50)]
        if scenario == "message_limit"
        else "Hello"
    )
    graph = sample_agent.build_agent(model, InMemorySaver())
    create = Mock(wraps=sample_client.create_response)
    monkeypatch.setattr(sample_client, "create_response", create)
    ask = Mock(side_effect=AssertionError("No reviewer prompt expected"))
    monkeypatch.setattr("builtins.input", ask)
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            sample_client.review_agent(
                client,
                SimpleNamespace(
                    prompt=prompt, stream=stream, decision=decision, recipient=None
                ),
            )
    ask.assert_not_called()
    assert create.call_count == 1
    send.assert_not_called()
    expected = (
        "Conversation limit reached."
        if scenario == "message_limit"
        else "No action needed."
    )
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("message_count", [48, 49, 50])
def test_custom_before_model_respects_message_limit(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    message_count: int,
) -> None:
    send = _record_send(sample_agent, monkeypatch)
    model = _model()
    graph = sample_agent.build_agent(model, InMemorySaver())
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            response = sample_client.create_response(
                client,
                input=[
                    {"role": "user", "content": f"Message {index}"}
                    for index in range(message_count)
                ],
                stream=stream,
            )
            assert response["status"] == "completed", response
            limited = message_count == 50
            assert (
                "Conversation limit reached." in assistant_text(response)
            ) is limited
            assert len(sentinels(response)) == (0 if limited else 1)
            send.assert_not_called()
            if not limited:
                response = sample_client.create_response(
                    client,
                    input=sample_client.build_resume_items(response, "approve"),
                    previous_response_id=response["id"],
                    stream=stream,
                )
    assert "Conversation limit reached." in assistant_text(response)
    assert not sentinels(response)
    assert send.call_count == (0 if limited else 1)
    assert model.i == (0 if limited else 1)


def test_client_does_not_resume_unrelated_function_calls(
    sample_client: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="No middleware review"):
        sample_client.build_resume_items(
            {
                "output": [
                    {"type": "function_call", "name": "send_email", "call_id": "tool"}
                ]
            },
            "approve",
        )


@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
def test_middleware_state_survives_host_recreation(
    sample_agent: ModuleType,
    sample_client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    store = _FakeStateStore()
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._foundry_checkpoint_saver."
        "FoundryStateStore.get_or_create",
        AsyncMock(return_value=store),
    )
    send = _record_send(sample_agent, monkeypatch)
    graph = sample_agent.build_agent(
        _model(), FoundryCheckpointSaver(credential=_credential())
    )
    conversation = {"id": "middleware-restart"}
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            paused = sample_client.create_response(
                client, input="Send it", conversation=conversation
            )
    assert len(sentinels(paused)) == 1, paused
    assert send.call_count == 0
    assert store.items
    resumed_model = ToolCallingModel(responses=[AIMessage(content="Restored review.")])
    restored_graph = sample_agent.build_agent(
        resumed_model, FoundryCheckpointSaver(credential=_credential())
    )
    with TestClient(ResponsesHostServer(restored_graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            finished = sample_client.create_response(
                client,
                input=sample_client.build_resume_items(
                    paused, decision, "bob@example.com"
                ),
                conversation=conversation,
            )
    assert "Restored review." in assistant_text(finished), finished
    assert not sentinels(finished), finished
    assert send.call_count == (0 if decision == "reject" else 1)
    if decision != "reject":
        assert send.call_args.kwargs["recipient"] == (
            "bob@example.com" if decision == "edit" else "alice@example.com"
        )


def test_client_rejects_unconfigured_decision(sample_client: ModuleType) -> None:
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "__hosted_agent_adapter_interrupt__",
                "call_id": "interrupt-1",
                "arguments": json.dumps(
                    {
                        "value": {
                            "action_requests": [
                                {"name": "send_email", "args": ORIGINAL_ARGUMENTS}
                            ],
                            "review_configs": [{"allowed_decisions": ["approve"]}],
                        }
                    }
                ),
            }
        ]
    }
    with pytest.raises(ValueError, match="not allowed"):
        sample_client.build_resume_items(response, "reject")
