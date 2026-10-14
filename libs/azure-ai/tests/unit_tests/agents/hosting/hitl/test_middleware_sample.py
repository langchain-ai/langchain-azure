"""Exercise the middleware sample through the unmodified Responses host."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

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
    run,
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


def _load_sample(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = SAMPLE_DIRECTORY / f"{name}.py"
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


@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
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


@pytest.mark.parametrize("is_hosted", [False, True])
def test_configuration_entrypoint_preserves_agent_composition(
    sample_agent: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    is_hosted: bool,
) -> None:
    monkeypatch.syspath_prepend(str(SAMPLE_DIRECTORY))
    monkeypatch.setitem(sys.modules, "agent", sample_agent)
    monkeypatch.setitem(
        sys.modules, "_langchain_azure_hosting_main", ModuleType("test_graph_factory")
    )
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://test.services.ai.azure.com/api/projects/sample",
    )
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "sample-model")
    model_factory = Mock(return_value=_model())
    credential = Mock()
    project = Mock()
    project.get_openai_client.return_value.base_url = "https://test/openai/v1/"
    monkeypatch.setattr(
        "azure.identity.DefaultAzureCredential", Mock(return_value=credential)
    )
    monkeypatch.setattr("azure.ai.projects.AIProjectClient", Mock(return_value=project))
    monkeypatch.setattr(
        "azure.identity.get_bearer_token_provider", Mock(return_value="token")
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", model_factory)
    monkeypatch.setattr(
        "azure.ai.agentserver.core.AgentConfig.from_env",
        lambda: SimpleNamespace(is_hosted=is_hosted),
    )
    foundry_saver = InMemorySaver()
    saver_factory = Mock(return_value=foundry_saver)
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting.FoundryCheckpointSaver", saver_factory
    )
    targets = run._read_graphs(SAMPLE_DIRECTORY / "langgraph.json")
    graph = run._load_graph(targets["agent"], SAMPLE_DIRECTORY)
    assert "model_call_count" in graph.channels
    if is_hosted:
        assert graph.checkpointer is foundry_saver
        saver_factory.assert_called_once_with()
    else:
        assert isinstance(graph.checkpointer, InMemorySaver)
        saver_factory.assert_not_called()
    model_factory.assert_called_once()
    assert model_factory.call_args.kwargs["model"] == "sample-model"
    assert model_factory.call_args.kwargs["use_responses_api"] is True


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
    assert resumed["model_call_count"] == 2
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
            assert _snapshot(graph, thread_id).values["model_call_count"] == 1
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
    assert state["model_call_count"] == 2
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
def test_custom_before_model_short_circuits_hosted_agent(
    sample_agent: ModuleType, sample_client: ModuleType, stream: bool
) -> None:
    model = _model()
    graph = sample_agent.build_agent(model, InMemorySaver(), max_messages=1)
    with TestClient(ResponsesHostServer(graph).app) as transport:
        with OpenAI(
            base_url="http://testserver/", api_key="local", http_client=transport
        ) as client:
            response = sample_client.create_response(
                client, input="Send it", stream=stream
            )
    assert response["status"] == "completed", response
    assert "Conversation limit reached." in assistant_text(response)
    assert not sentinels(response)
    assert model.i == 0


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
    state = _snapshot(restored_graph, conversation["id"]).values
    assert state["model_call_count"] == 2
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
