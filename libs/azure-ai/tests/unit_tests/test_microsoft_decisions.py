import json
import time
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from azure.core.credentials import AccessToken, AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.decisions import BaseDecisionModel, Box
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from langchain_azure_ai.decisions import (
    AzureAIMicrosoftDecisions,
    Choice,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulCriteria,
    Score,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_azure_ai._api.base.ExperimentalWarning"
)

ENDPOINT = "https://test.services.ai.azure.com"
PROJECT_ENDPOINT = ENDPOINT + "/api/projects/test-project"
REQUEST: ClassifierRequest = {
    "state": {"ticket": "Checkout stopped working.", "attempts": 2},
    "questions": {
        "outage": Noul(
            instructions="Is there an outage?",
            criteria=NoulCriteria(true="The service stopped working"),
        ),
        "severity": Score(instructions="How severe?", criteria=["Minor", "Major"]),
        "team": Choice(
            instructions="Which team?", criteria={"billing": "Costs", "tech": "Bugs"}
        ),
    },
}
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


@pytest.mark.parametrize(
    ("model_id", "deployment", "use_project", "credential"),
    [
        (
            "microsoft-decision-1",
            "first-deployment",
            False,
            "test-key",
        ),
        (
            "microsoft-decision-1.1",
            "second-deployment",
            True,
            AzureKeyCredential("test-key"),
        ),
    ],
)
def test_systemone_contract(
    model_id: str,
    deployment: str,
    use_project: bool,
    credential: str | AzureKeyCredential,
) -> None:
    answers = {
        "outage": {"type": "noul", "noul": 0.9},
        "severity": {
            "type": "score",
            "score": 0.8,
            "confidence": 0.7,
            "probabilities": {"0": 0.2, "1": 0.8},
        },
        "team": {
            "type": "choice",
            "choice": "tech",
            "probabilities": {"billing": 0.1, "tech": 0.9},
            "confidence": 0.8,
        },
    }

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == ENDPOINT + "/providers/microsoft/v1/systemone"
        assert request.headers["api-key"] == "test-key"
        assert "authorization" not in request.headers
        expected = {
            "outage": {
                "type": "noul",
                "instructions": "Is there an outage?",
                "criteria": {"true": "The service stopped working"},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["Minor", "Major"],
            },
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Costs", "tech": "Bugs"},
            },
        }
        assert json.loads(request.content) == {
            "model": deployment,
            "state": REQUEST["state"],
            "questions": expected,
        }
        return httpx.Response(
            200,
            json={
                "model": model_id,
                "answers": answers,
                "usage": {"input_tokens": 30, "output_tokens": 0},
            },
            headers={"apim-request-id": "test-id"},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        decisions = AzureAIMicrosoftDecisions(
            endpoint=None if use_project else ENDPOINT + "/",
            project_endpoint=PROJECT_ENDPOINT + "/" if use_project else None,
            model=deployment,
            credential=credential,
            http_client=client,
        )
        assert isinstance(decisions, BaseDecisionModel)
        answer = decisions.invoke(REQUEST)
        assert not client.is_closed

    assert "test-key" not in repr(decisions)
    if isinstance(credential, str):
        assert isinstance(decisions.credential, SecretStr)
    assert "credential" not in decisions.model_dump()
    assert isinstance(answer, ClassifierResponse)
    assert answer.model == model_id
    assert answer.nouls["outage"].noul == 0.9
    assert answer.scores["severity"].legend == {0: "Minor", 1: "Major"}
    assert answer.scores["severity"].probabilities == {0: 0.2, 1: 0.8}
    assert answer.scores["severity"].confidence == 0.7
    assert answer.choices["team"].choice == "tech"
    assert answer.choices["team"].confidence == 0.8
    assert answer.usage == {"input_tokens": 30, "output_tokens": 0, "total_tokens": 30}
    assert answer.request_id == "test-id"


@pytest.mark.parametrize("async_credential", [False, True])
async def test_entra_auth(async_credential: bool) -> None:
    token = AccessToken("test-token", int(time.time()) + 600)
    if async_credential:
        credential = Mock(spec=AsyncTokenCredential)
        credential.get_token = AsyncMock(return_value=token)
    else:
        credential = Mock(spec=TokenCredential)
        credential.get_token.return_value = token
    request: ClassifierRequest = {
        "state": "Checkout stopped working.",
        "questions": {"outage": Noul(instructions="Is there an outage?")},
    }

    def handle(http_request: httpx.Request) -> httpx.Response:
        assert http_request.headers["authorization"] == "Bearer test-token"
        assert "api-key" not in http_request.headers
        assert json.loads(http_request.content)["state"] == request["state"]
        return httpx.Response(
            200,
            json={
                "model": "microsoft-decision-1",
                "answers": {"outage": {"type": "noul", "noul": 0.9}},
            },
        )

    transport = httpx.MockTransport(handle)
    with httpx.Client(transport=transport) as client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            decisions = AzureAIMicrosoftDecisions(
                endpoint=ENDPOINT,
                model="deployment",
                credential=credential,
                http_client=client,
                http_async_client=async_client,
            )
            result = await decisions.ainvoke(request)
            assert result.nouls["outage"].noul == 0.9
            assert not async_client.is_closed
            if async_credential:
                credential.get_token.assert_awaited_once_with(TOKEN_SCOPE)
                with pytest.raises(ValueError, match="requires ainvoke"):
                    decisions.invoke(request)
            else:
                assert decisions.invoke(request) == result
                assert all(
                    call.args == (TOKEN_SCOPE,)
                    for call in credential.get_token.call_args_list
                )


@pytest.mark.parametrize(
    ("classifier_input", "message"),
    [
        ({"state": "text", "questions": {}}, "at least one"),
        (
            {"state": "text", "questions": {"location": Box(instructions="Find it")}},
            "supports Noul, Choice, and Score",
        ),
        (
            {**REQUEST, "state": {"message": HumanMessage("hello")}},
            "State must be text",
        ),
    ],
)
def test_invalid_request(classifier_input: ClassifierRequest, message: str) -> None:
    decisions = AzureAIMicrosoftDecisions(
        endpoint=ENDPOINT, model="deployment", credential="test-key"
    )
    with pytest.raises(ValueError, match=message):
        decisions.invoke(classifier_input)


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (307, {}, httpx.HTTPStatusError),
        (429, {}, httpx.HTTPStatusError),
        (200, {"model": "microsoft-decision-1", "answers": {}}, ValueError),
        (200, {"model": "microsoft-decision-1", "answers": []}, ValueError),
    ],
)
def test_api_errors(status: int, body: dict, error: type[Exception]) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "test.services.ai.azure.com"
        return httpx.Response(
            status, json=body, headers={"location": "https://other.test"}
        )

    with httpx.Client(
        transport=httpx.MockTransport(handle), follow_redirects=True
    ) as client:
        decisions = AzureAIMicrosoftDecisions(
            endpoint=ENDPOINT,
            model="deployment",
            credential="test-key",
            http_client=client,
        )
        with pytest.raises(error):
            decisions.invoke(REQUEST)


@pytest.mark.parametrize(
    "suffix", ["/api/projects/example", "/openai/v1", "?key=secret"]
)
def test_requires_resource_endpoint(suffix: str) -> None:
    with pytest.raises(ValueError, match="HTTPS Foundry.*endpoint"):
        AzureAIMicrosoftDecisions(
            endpoint=ENDPOINT + suffix, model="deployment", credential="test-key"
        )


def test_project_endpoint_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", PROJECT_ENDPOINT)
    decisions = AzureAIMicrosoftDecisions(model="deployment", credential="test-key")
    assert decisions.project_endpoint == PROJECT_ENDPOINT


def test_rejects_conflicting_endpoints() -> None:
    with pytest.raises(ValueError, match="Provide only one"):
        AzureAIMicrosoftDecisions(
            project_endpoint=PROJECT_ENDPOINT,
            endpoint=ENDPOINT,
            model="deployment",
            credential="test-key",
        )
