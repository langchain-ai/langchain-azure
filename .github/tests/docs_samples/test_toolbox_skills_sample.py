"""Exercise standalone Toolbox skills through the sample's native Responses host."""

import asyncio
import importlib.util
import json
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from mcp.types import TextResourceContents
from pydantic import Field
from starlette.testclient import TestClient

# Windows asyncio uses loopback socket pairs; external connections remain blocked.
pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])

SAMPLE = (
    Path(__file__).resolve().parents[3]
    / "samples/hosting/langgraph-hosted-agents/responses/14_foundry_toolbox_skills"
)
SKILL_URI = "skill://release-notes/SKILL.md"
INDEX_URI = "skill://index.json"
SKILL = (SAMPLE / "skill-src/release-notes/SKILL.md").read_text(encoding="utf-8")
CATALOG = json.dumps(
    {
        "skills": [
            {
                "name": "release-notes",
                "type": "skill-md",
                "description": (
                    "Turn raw commits, tickets, or change summaries into "
                    "structured customer-facing release notes."
                ),
                "url": SKILL_URI,
            }
        ]
    }
)


class ScriptedModel(GenericFakeChatModel):
    disable_streaming: bool = True
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    tool_names: set[str] = Field(default_factory=set)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.tool_names = {tool.name for tool in tools}
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def call(name: str, **arguments: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": name}],
    )


@pytest.fixture
def sample(docs_samples_dirs: list[Path], monkeypatch: pytest.MonkeyPatch) -> Any:
    if not any(SAMPLE.is_relative_to(directory) for directory in docs_samples_dirs):
        pytest.skip("This smoke test covers the Foundry Toolbox skills sample.")
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT", "https://example.test/api/projects/p"
    )
    monkeypatch.setenv("TOOLBOX_NAME", "agent-skills")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "toolbox_skills_sample", SAMPLE / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mcp_boundary(sample: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    class Client:
        def __init__(self) -> None:
            self.resources: dict[str, list[str]] = {
                INDEX_URI: [CATALOG],
                SKILL_URI: [SKILL],
            }
            self.resource_reads: list[str] = []
            self.fail_uris: set[str] = set()

        async def get_tools(self) -> list[Any]:
            raise AssertionError("Standalone skills sample must not load Toolbox tools")

        @asynccontextmanager
        async def session(self, server_name: str) -> Any:
            assert server_name == "toolbox"
            yield self

        async def read_resource(self, uri: str) -> Any:
            self.resource_reads.append(uri)
            if uri in self.fail_uris:
                raise RuntimeError("transient toolbox failure")
            return SimpleNamespace(
                contents=[
                    TextResourceContents(uri=uri, text=text)
                    for text in self.resources[uri]
                ]
            )

    client = Client()
    monkeypatch.setattr(
        sample.AzureAIProjectToolbox, "_build_mcp_client", lambda _: client
    )
    return SimpleNamespace(client=client)


def test_model_replays_tool_turn_without_provider_storage(
    sample: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["store"] is False
        assert payload["input"][1]["type"] == "function_call"
        assert payload["input"][2]["type"] == "function_call_output"
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": 0,
                "model": "gpt-4o",
                "object": "response",
                "status": "completed",
                "output": [],
            },
        )

    monkeypatch.setattr(sample, "DefaultAzureCredential", Mock())
    project = Mock()
    project.get_openai_client.return_value.base_url = "https://example.test/openai/v1/"
    monkeypatch.setattr(sample, "AIProjectClient", Mock(return_value=project))
    monkeypatch.setattr(
        sample, "get_bearer_token_provider", lambda *_: lambda: "test-token"
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            sample, "ChatOpenAI", partial(sample.ChatOpenAI, http_client=client)
        )
        model = sample._build_chat_model()
        result = model.invoke(
            [
                HumanMessage(content="Draft release notes."),
                AIMessage(
                    content=[
                        {
                            "type": "function_call",
                            "id": "fc_test",
                            "call_id": "call_test",
                            "name": "read_skill_resource",
                            "arguments": json.dumps({"uri": SKILL_URI}),
                        }
                    ]
                ),
                ToolMessage(content=SKILL, tool_call_id="call_test"),
            ]
        )
    assert result.response_metadata["status"] == "completed"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "exporter",
    [None, "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"],
)
def test_entrypoint_reads_skill_on_demand_without_toolbox_tools(
    sample: Any,
    mcp_boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream: bool,
    exporter: str | None,
) -> None:
    final = (
        "Highlights: added --timeout flag (default 30s). "
        "Fixes: retry transient 429 responses. "
        "Breaking Changes: dropped Python 3.10 support. "
        "Upgrade Steps: 1. Upgrade to Python 3.11+."
    )
    model = ScriptedModel(
        messages=iter(
            [
                call("read_skill_resource", uri=SKILL_URI),
                AIMessage(content=final),
            ]
        )
    )
    tracing = Mock()
    monkeypatch.setattr(sample, "_build_chat_model", lambda: model)
    monkeypatch.setattr(sample, "enable_auto_tracing", tracing)
    if exporter:
        monkeypatch.setenv(exporter, "http://127.0.0.1:4318/v1/traces")
        monkeypatch.setattr(sample, "TracerProvider", Mock())
        monkeypatch.setattr(sample, "BatchSpanProcessor", Mock())
        monkeypatch.setattr(sample, "OTLPSpanExporter", Mock())
        monkeypatch.setattr(sample.trace, "set_tracer_provider", Mock())
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path / "state"))

    def run(host: Any, **kwargs: Any) -> None:
        assert mcp_boundary.client.resource_reads == [INDEX_URI]
        with TestClient(host.app) as client:
            response = client.post(
                "/responses",
                json={
                    "input": (
                        "Draft customer release notes for v2.0 from these commits: "
                        "feat: add --timeout flag (default 30s); "
                        "fix: retry transient 429 responses; "
                        "breaking: drop Python 3.10 support."
                    ),
                    "stream": stream,
                    "store": False,
                },
            )
        assert response.status_code == 200, response.text
        if stream:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert not any(
                event["type"] in {"error", "response.failed"} for event in events
            )
            payload = next(
                event["response"]
                for event in events
                if event["type"] == "response.completed"
            )
        else:
            payload = response.json()
        assert payload["status"] == "completed"
        names = [
            item["name"]
            for item in payload["output"]
            if item["type"] == "function_call"
        ]
        assert names == ["read_skill_resource"]
        text = "".join(
            part.get("text", "")
            for item in payload["output"]
            if item["type"] == "message"
            for part in item["content"]
        )
        assert text == final

    monkeypatch.setattr(sample.ResponsesHostServer, "run", run)
    sample.main()
    if exporter:
        tracing.assert_called_once_with()
        sample.trace.set_tracer_provider.assert_called_once()
    else:
        tracing.assert_called_once_with(auto_configure_azure_monitor=True)
    assert mcp_boundary.client.resource_reads == [INDEX_URI, SKILL_URI]
    assert model.tool_names == {"read_skill_resource"}
    first_prompt = str(model.requests[0][0].content)
    assert CATALOG in first_prompt
    assert "Rewrite internal commit jargon" not in first_prompt
    assert model.requests[1][-1].content == SKILL


def test_unrelated_request_does_not_read_skill(sample: Any, mcp_boundary: Any) -> None:
    read_skill_resource, catalog = asyncio.run(sample._load_toolbox_skills())
    model = ScriptedModel(messages=iter([AIMessage(content="Hello!")]))
    graph = sample.create_agent(
        model, tools=[read_skill_resource], system_prompt=catalog
    )
    result = asyncio.run(graph.ainvoke({"messages": [("user", "Hello")]}))
    assert result["messages"][-1].content == "Hello!"
    assert mcp_boundary.client.resource_reads == [INDEX_URI]
    assert SKILL not in str(model.requests)


def test_supporting_resource_is_read_on_demand(sample: Any, mcp_boundary: Any) -> None:
    reference_uri = "skill://release-notes/references/style.md"
    mcp_boundary.client.resources[reference_uri] = ["Reference text."]
    read_skill_resource, _ = asyncio.run(sample._load_toolbox_skills())
    assert mcp_boundary.client.resource_reads == [INDEX_URI]
    result = asyncio.run(read_skill_resource.ainvoke({"uri": reference_uri}))
    assert result == "Reference text."
    assert mcp_boundary.client.resource_reads == [INDEX_URI, reference_uri]


@pytest.mark.parametrize("uri", ["file:///secret.txt", "https://example.test/skill"])
def test_reader_rejects_non_skill_uri(sample: Any, mcp_boundary: Any, uri: str) -> None:
    read_skill_resource, _ = asyncio.run(sample._load_toolbox_skills())
    result = asyncio.run(read_skill_resource.ainvoke({"uri": uri}))
    assert "Use a skill:// URI" in result
    assert mcp_boundary.client.resource_reads == [INDEX_URI]


def test_reader_returns_sanitized_error_on_read_failure(
    sample: Any, mcp_boundary: Any
) -> None:
    mcp_boundary.client.fail_uris.add(SKILL_URI)
    read_skill_resource, _ = asyncio.run(sample._load_toolbox_skills())
    result = asyncio.run(read_skill_resource.ainvoke({"uri": SKILL_URI}))
    assert "Could not read the skill resource" in result
    assert "transient toolbox failure" not in result


@pytest.mark.parametrize("contents", [[], ["   "]])
def test_empty_skill_returns_tool_error(
    sample: Any,
    mcp_boundary: Any,
    contents: list[str],
) -> None:
    mcp_boundary.client.resources[SKILL_URI] = contents
    read_skill_resource, _ = asyncio.run(sample._load_toolbox_skills())
    result = asyncio.run(read_skill_resource.ainvoke({"uri": SKILL_URI}))
    assert "No skill content returned" in result


@pytest.mark.parametrize("contents", [[], ["   "]])
def test_empty_index_fails_at_startup(
    sample: Any, mcp_boundary: Any, contents: list[str]
) -> None:
    mcp_boundary.client.resources[INDEX_URI] = contents
    with pytest.raises(ValueError, match="must provide skill://index.json"):
        asyncio.run(sample._load_toolbox_skills())
