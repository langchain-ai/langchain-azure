"""Exercise Toolbox skills and tools through the sample's native Responses host."""

import asyncio
import importlib.util
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from mcp.types import TextResourceContents
from pydantic import Field
from starlette.testclient import TestClient

# Windows asyncio uses loopback socket pairs; external connections remain blocked.
pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])

SAMPLE = (
    Path(__file__).resolve().parents[3]
    / "samples/hosting/langgraph-hosted-agents/responses/04_foundry_toolbox"
)
SKILL_URI = "skill://source-brief/SKILL.md"
SKILL = (SAMPLE / "skill-src/source-brief/SKILL.md").read_text(encoding="utf-8")


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
        pytest.skip("This smoke test covers the Foundry Toolbox sample.")
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT", "https://example.test/api/projects/p"
    )
    monkeypatch.setenv("TOOLBOX_NAME", "agent-tools")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("toolbox_sample", SAMPLE / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mcp_boundary(sample: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    calls = []

    @tool
    def web_search(query: str) -> str:
        """Search primary sources (simulated service boundary)."""
        calls.append(("web_search", query))
        return (
            "Python 3.14 release notes: https://docs.python.org/3.14/whatsnew/3.14.html"
        )

    @tool
    def code_interpreter(code: str) -> str:
        """Calculate from the user's measurements (simulated service boundary)."""
        calls.append(("code_interpreter", code))
        assert code == "(250 - 200) / 250 * 100"
        return "20.0"

    class Client:
        contents = [SKILL]
        resource_reads: list[str] = []

        async def get_tools(self) -> list[Any]:
            return [web_search, code_interpreter]

        @asynccontextmanager
        async def session(self, server_name: str) -> Any:
            assert server_name == "toolbox"
            yield self

        async def read_resource(self, uri: str) -> Any:
            assert uri == SKILL_URI
            self.resource_reads.append(uri)
            return SimpleNamespace(
                contents=[
                    TextResourceContents(uri=uri, text=text) for text in self.contents
                ]
            )

    client = Client()
    monkeypatch.setattr(
        sample.AzureAIProjectToolbox, "_build_mcp_client", lambda _: client
    )
    return SimpleNamespace(client=client, calls=calls)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize(
    "exporter",
    [None, "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"],
)
def test_entrypoint_loads_skill_prompt_and_calls_tools(
    sample: Any,
    mcp_boundary: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream: bool,
    exporter: str | None,
) -> None:
    final = (
        "Recommendation: test the upgrade. Evidence: our latency fell 20%; "
        "release guidance https://docs.python.org/3.14/whatsnew/3.14.html. "
        "Open questions: production workload coverage."
    )
    model = ScriptedModel(
        messages=iter(
            [
                call("web_search", query="Python 3.14 release notes"),
                call("code_interpreter", code="(250 - 200) / 250 * 100"),
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
        with TestClient(host.app) as client:
            response = client.post(
                "/responses",
                json={
                    "input": "Use source-brief to research Python 3.14 and calculate "
                    "the latency reduction from our 250 ms and 200 ms benchmarks.",
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
        assert names == ["web_search", "code_interpreter"]
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
    assert mcp_boundary.client.resource_reads == [SKILL_URI]
    assert [name for name, _ in mcp_boundary.calls] == [
        "web_search",
        "code_interpreter",
    ]
    assert model.tool_names == {"web_search", "code_interpreter"}
    first_prompt = str(model.requests[0][0].content)
    assert SKILL in first_prompt


@pytest.mark.parametrize("contents", [[], ["   "]])
def test_empty_skill_fails_at_startup(
    sample: Any,
    mcp_boundary: Any,
    contents: list[str],
) -> None:
    mcp_boundary.client.contents = contents
    with pytest.raises(ValueError, match="must provide source-brief/SKILL.md"):
        asyncio.run(sample._load_toolbox())
