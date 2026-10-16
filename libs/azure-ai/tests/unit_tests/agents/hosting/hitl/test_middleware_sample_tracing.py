"""Check real middleware-sample tracers in isolated processes."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("microsoft.opentelemetry")
pytest.importorskip("opentelemetry.sdk")

SAMPLE_DIRECTORY = (
    Path(__file__).resolve().parents[7]
    / "samples/hosting/langgraph-hosted-agents/responses/11_middleware"
)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    ("sdk_capture", "host_capture", "expect_content"),
    [
        pytest.param(None, None, False, id="default"),
        pytest.param(None, "false", False, id="disabled"),
        pytest.param("true", "false", True, id="sdk-opt-in"),
        pytest.param("false", "true", True, id="host-opt-in"),
    ],
)
def test_real_tracers_require_content_opt_in(
    tmp_path: Path,
    sdk_capture: str | None,
    host_capture: str | None,
    expect_content: bool,
    stream: bool,
) -> None:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(
            (
                "OTEL_",
                "AZURE_TRACING_",
                "APPLICATIONINSIGHTS_",
                "APPLICATION_INSIGHTS_",
                "FOUNDRY_",
                "AZURE_AI_",
                "AGENTSERVER_",
                "MICROSOFT_OTEL_",
            )
        )
    }
    env.update(
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        MICROSOFT_OTEL_SDKSTATS_DISABLED="true",
        APPLICATIONINSIGHTS_STATSBEAT_DISABLED_ALL="true",
    )
    if sdk_capture is not None:
        env["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = sdk_capture
    if host_capture is not None:
        env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = host_capture
    source = """
        import asyncio
        import os
        import sys
        from contextlib import AsyncExitStack

        from azure.identity import DefaultAzureCredential
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel,
        )
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from openai import OpenAI
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from starlette.testclient import TestClient

        sys.path.insert(0, sys.argv[1])
        import main as sample_main
        from agent import build_agent
        from client import build_resume_items, create_response

        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        class MemoryConversationStore:
            def __init__(self):
                self.records = {}

            async def get(self, conversation_chain_id, key):
                return self.records.get((conversation_chain_id, key))

            async def set(self, conversation_chain_id, key, data):
                self.records[(conversation_chain_id, key)] = dict(data)

        user_message = "private-user-message-314159"
        arguments = {
            "recipient": "private-recipient-314159@example.com",
            "subject": "private-subject-314159",
            "body": "private-email-body-314159",
        }
        model_result = "private-model-result-314159"
        markers = [
            user_message, *arguments.values(), model_result, "Simulated email to",
        ]
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        async def exercise():
            async with AsyncExitStack() as resources:
                credential = resources.enter_context(DefaultAzureCredential())
                sample_main._configure_tracing(resources, credential)
                model = ToolCallingModel(responses=[
                    AIMessage(content="", tool_calls=[{
                        "id": "call-private-email",
                        "name": "send_email",
                        "args": arguments,
                    }]),
                    AIMessage(content=model_result),
                ])
                graph = build_agent(model, InMemorySaver())
                host = sample_main.ResponsesHostServer(
                    graph, conversation_chain_store=MemoryConversationStore(),
                )
                with TestClient(host.app) as transport:
                    with OpenAI(
                        base_url="http://testserver/", api_key="local",
                        http_client=transport,
                    ) as client:
                        stream = sys.argv[3] == "true"
                        first = create_response(
                            client, input=user_message, stream=stream,
                        )
                        replies = build_resume_items(first, "approve")
                        assert len(replies) == 1, first
                        state = await graph.aget_state({
                            "configurable": {"thread_id": first["id"]},
                        })
                        assert not any(
                            isinstance(message, ToolMessage)
                            for message in state.values["messages"]
                        )
                        finished = create_response(
                            client, input=replies, stream=stream,
                            previous_response_id=first["id"],
                        )
                        assert finished["status"] == "completed", finished
                        state = await graph.aget_state({
                            "configurable": {"thread_id": first["id"]},
                        })
                        results = [
                            message for message in state.values["messages"]
                            if isinstance(message, ToolMessage)
                        ]
                        assert len(results) == 1
                        assert results[0].content.startswith("Simulated email to")
                provider.force_flush()
                spans = exporter.get_finished_spans()
                scopes = {span.instrumentation_scope.name for span in spans}
                assert any(
                    scope.startswith("microsoft.opentelemetry")
                    and "langchain" in scope for scope in scopes
                ), scopes
                assert "AzureAIOpenTelemetryTracer" in scopes, scopes
                payload = "\\n".join(span.to_json() for span in spans)
                captured = {marker: marker in payload for marker in markers}
                if sys.argv[2] == "true":
                    assert all(captured.values()), captured
                else:
                    assert not any(captured.values()), captured
                for scope in scopes:
                    variable = (
                        "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"
                        if scope == "AzureAIOpenTelemetryTracer"
                        else "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
                    )
                    scope_payload = "\\n".join(
                        span.to_json() for span in spans
                        if span.instrumentation_scope.name == scope
                    )
                    captured = {
                        marker: marker in scope_payload for marker in markers
                    }
                    if os.environ.get(variable) == "true":
                        assert all(captured.values()), (scope, captured)
                    else:
                        assert not any(captured.values()), (scope, captured)
                print(f"Verified {len(spans)} real spans across {len(scopes)} scopes")

        try:
            asyncio.run(exercise())
        finally:
            provider.shutdown()
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(source),
            str(SAMPLE_DIRECTORY),
            str(expect_content).lower(),
            str(stream).lower(),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]
