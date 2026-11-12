"""Azure configuration and HTTP contract tests for Decisions."""

import json
import time
from types import TracebackType
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from azure.core.credentials import AccessToken, AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.runnables import RunnableLambda
from langchain_openai.decisions import OpenAIDecisions
from pydantic import SecretStr

from langchain_azure_ai._api.base import ExperimentalWarning
from langchain_azure_ai.decisions import (
    AzureAIOpenAIApiDecisions,
    Choice,
    DecisionRequest,
    Predicate,
    Score,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::langchain_azure_ai._api.base.ExperimentalWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
    ),
]

ENDPOINT = "https://test.services.ai.azure.com/openai/v1"
PROJECT = "https://test.services.ai.azure.com/api/projects/test-project"
REQUEST: DecisionRequest = {
    "input": "Checkout is broken. Please refund my payment.",
    "questions": {
        "refund": Predicate(instructions="Is a refund requested?"),
        "team": Choice(
            instructions="Who should handle this?",
            choices={"billing": "Payments", "engineering": "Bugs"},
        ),
        "severity": Score(instructions="How severe?", levels=["Minor", "Major"]),
    },
}
BODY = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "refund", "probability": 0.9},
        {
            "type": "choice",
            "name": "team",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": [
                {"value": "billing", "probability": 0.9},
                {"value": "engineering", "probability": 0.1},
            ],
        },
        {
            "type": "score",
            "name": "severity",
            "score": 0.8,
            "confidence": 0.7,
            "probabilities": [
                {"value": 0, "label": "Minor", "probability": 0.2},
                {"value": 1, "label": "Major", "probability": 0.8},
            ],
        },
    ],
    "usage": {"input_tokens": 30, "output_tokens": 0},
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AZURE_AI_PROJECT_ENDPOINT",
        "FOUNDRY_PROJECT_ENDPOINT",
        "AZURE_AI_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "LANGSMITH_GATEWAY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=BODY, headers={"x-request-id": "test-id"})

    return httpx.MockTransport(handle)


class SyncCredential:
    def __init__(self) -> None:
        self.scopes: list[tuple[str, ...]] = []

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        self.scopes.append(scopes)
        return AccessToken(f"token-{len(self.scopes)}", int(time.time()) + 30)


class AsyncCredential(AsyncTokenCredential):
    def __init__(self) -> None:
        self.scopes: list[tuple[str, ...]] = []

    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        self.scopes.append(scopes)
        return AccessToken(f"token-{len(self.scopes)}", int(time.time()) + 30)

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "AsyncCredential":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        pass


@pytest.mark.parametrize("credential", ["test-key", AzureKeyCredential("test-key")])
def test_sync_contract(
    credential: Any, transport: httpx.MockTransport, requests: list[httpx.Request]
) -> None:
    with httpx.Client(transport=transport) as client:
        decisions = AzureAIOpenAIApiDecisions(
            endpoint=ENDPOINT,
            credential=credential,
            model="custom-deployment",
            http_client=client,
        )
        answer = decisions.invoke(REQUEST)
    assert str(requests[0].url) == ENDPOINT + "/decisions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert requests[0].headers["x-ms-useragent"] == "langchain-azure-ai"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "custom-deployment"
    assert [q["type"] for q in payload["questions"]] == ["predicate", "choice", "score"]
    assert answer.predicates["refund"].probability == 0.9
    assert answer.choices["team"].choice == "billing"
    assert answer.scores["severity"].legend == {0: "Minor", 1: "Major"}
    assert answer.usage.input_tokens == 30
    assert answer.request_id == "test-id"


@pytest.mark.parametrize("credential_type", [SyncCredential, AsyncCredential])
async def test_async_token_refresh(
    credential_type: Any, transport: httpx.MockTransport, requests: list[httpx.Request]
) -> None:
    credential = credential_type()
    async with httpx.AsyncClient(transport=transport) as client:
        decisions = AzureAIOpenAIApiDecisions(
            endpoint=ENDPOINT,
            credential=credential,
            model="custom-deployment",
            http_async_client=client,
        )
        for _ in range(2):
            answer = await decisions.ainvoke(REQUEST)
            assert answer.predicates["refund"].probability == 0.9
    assert [r.headers["authorization"] for r in requests] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert credential.scopes == [("https://ai.azure.com/.default",)] * 2


def test_sync_token_refresh(
    transport: httpx.MockTransport, requests: list[httpx.Request]
) -> None:
    credential = SyncCredential()
    with httpx.Client(transport=transport) as client:
        decisions = AzureAIOpenAIApiDecisions(
            project_endpoint=PROJECT,
            credential=credential,
            model="custom-deployment",
            http_client=client,
        )
        decisions.batch([REQUEST, REQUEST], config={"max_concurrency": 1})
    assert [str(r.url) for r in requests] == [ENDPOINT + "/decisions"] * 2
    assert [r.headers["authorization"] for r in requests] == [
        "Bearer token-1",
        "Bearer token-2",
    ]


def test_async_credential_rejects_sync() -> None:
    decisions = AzureAIOpenAIApiDecisions(
        endpoint=ENDPOINT, credential=AsyncCredential(), model="deployment"
    )
    with pytest.raises(ValueError, match="Sync invocation"):
        decisions.invoke(REQUEST)


@pytest.mark.parametrize(
    "project_env", ["AZURE_AI_PROJECT_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT"]
)
def test_project_environment(project_env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(project_env, PROJECT)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment")
    monkeypatch.setenv(
        "AZURE_AI_OPENAI_ENDPOINT", "https://other.example.com/openai/v1"
    )
    with patch(
        "langchain_azure_ai.decisions.openai.DefaultAzureCredential",
        return_value=SyncCredential(),
    ):
        decisions = AzureAIOpenAIApiDecisions()
    assert decisions.base_url == ENDPOINT
    assert decisions.project_endpoint == PROJECT
    assert decisions.model == "env-deployment"


def test_azure_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.services.ai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    decisions = AzureAIOpenAIApiDecisions()
    assert decisions.base_url == ENDPOINT
    assert isinstance(decisions.api_key, SecretStr)
    assert decisions.api_key.get_secret_value() == "env-key"
    monkeypatch.setenv(
        "AZURE_AI_OPENAI_ENDPOINT", "https://preferred.example.com/openai/v1"
    )
    assert (
        AzureAIOpenAIApiDecisions().base_url
        == "https://preferred.example.com/openai/v1"
    )


def test_explicit_config_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", PROJECT)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    decisions = AzureAIOpenAIApiDecisions(
        endpoint="https://explicit.example.com/openai/v1",
        credential="explicit-key",
        model="explicit-deployment",
    )
    assert decisions.base_url == "https://explicit.example.com/openai/v1"
    assert decisions.model == "explicit-deployment"
    assert isinstance(decisions.api_key, SecretStr)
    assert decisions.api_key.get_secret_value() == "explicit-key"


@pytest.mark.parametrize("credential", [SyncCredential(), AsyncCredential()])
def test_default_credential(credential: Any) -> None:
    with patch(
        "langchain_azure_ai.decisions.openai.DefaultAzureCredential",
        return_value=credential,
    ):
        decisions = AzureAIOpenAIApiDecisions(endpoint=ENDPOINT, model="deployment")
    assert callable(decisions.api_key)
    assert not credential.scopes


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "Provide an Azure"),
        ({"endpoint": ENDPOINT, "project_endpoint": PROJECT}, "only one"),
        ({"endpoint": ENDPOINT, "base_url": ENDPOINT}, "only one"),
        ({"endpoint": ENDPOINT, "credential": "key", "api_key": "key"}, "only one"),
        ({"project_endpoint": PROJECT, "credential": "key"}, "token credential"),
        ({"project_endpoint": PROJECT, "api_key": "key"}, "token credential"),
    ],
)
def test_configuration_errors(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AzureAIOpenAIApiDecisions(model="deployment", **kwargs)


def test_native_options_and_runnable_composition(
    transport: httpx.MockTransport, requests: list[httpx.Request]
) -> None:
    with httpx.Client(transport=transport) as client:
        decisions = AzureAIOpenAIApiDecisions(
            base_url=ENDPOINT,
            api_key=SecretStr("test-key"),
            model="deployment",
            http_client=client,
            timeout=12,
            max_retries=0,
            default_headers={"x-extra": "value"},
            default_query={"test": "value"},
        )
        chain = decisions | RunnableLambda(
            lambda answer: answer.predicates["refund"].probability
        )
        assert chain.invoke(REQUEST) == 0.9
    assert requests[0].url.params["test"] == "value"
    assert requests[0].headers["x-extra"] == "value"
    assert decisions._client.timeout == 12


def test_refusals_and_api_errors() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [{"type": "refusal", "name": "refund"}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        decisions = AzureAIOpenAIApiDecisions(
            endpoint=ENDPOINT,
            credential="test-key",
            model="deployment",
            http_client=client,
            max_retries=0,
        )
        assert "refund" in decisions.invoke(REQUEST).refusals
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, json={"error": {"message": "unauthorized"}})
        )
    ) as client:
        decisions = AzureAIOpenAIApiDecisions(
            endpoint=ENDPOINT,
            credential="test-key",
            model="deployment",
            http_client=client,
            max_retries=0,
        )
        with pytest.raises(openai.AuthenticationError):
            decisions.invoke(REQUEST)


@pytest.mark.parametrize(
    "credential", ["private-test-key", AzureKeyCredential("private-test-key")]
)
def test_no_credential_leak(credential: Any) -> None:
    decisions = AzureAIOpenAIApiDecisions(
        endpoint=ENDPOINT, credential=credential, model="deployment"
    )
    assert "private-test-key" not in repr(decisions)
    assert "private-test-key" not in decisions.model_dump_json()
    assert "private-test-key" not in json.dumps(decisions.to_json())


def test_experimental_warning_and_exports() -> None:
    import langchain_openai.decisions as upstream

    import langchain_azure_ai.decisions as azure

    for name in azure.__all__:
        if name != "AzureAIOpenAIApiDecisions":
            assert getattr(azure, name) is getattr(upstream, name)
    assert issubclass(AzureAIOpenAIApiDecisions, OpenAIDecisions)
    with pytest.warns(ExperimentalWarning, match="is experimental"):
        AzureAIOpenAIApiDecisions(
            endpoint=ENDPOINT, credential="test-key", model="deployment"
        )
