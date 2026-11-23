"""LangChain decision model for Microsoft Decision deployments in Foundry."""

import math
import re
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

import httpx
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.decisions import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    ClassifierResponse,
    Noul,
    Question,
    Score,
    ScoreAnswer,
    State,
)
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from langchain_azure_ai._resources import (
    _get_base_url_from_endpoint,
    _make_async_token_provider,
    _make_token_provider,
)
from langchain_azure_ai.utils.env import get_project_endpoint

_DECISION_PATH = "/providers/microsoft/v1/systemone"
_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
_JSON_STATE = TypeAdapter(str | dict[str, JsonValue] | list[JsonValue])


class AzureAIMicrosoftDecisionsClassifier(BaseDecisionModel):
    """Ask typed questions with a Microsoft Decision deployment in Foundry.

    The deployment must use a compatible System One request and response format.
    Pass the deployment name as ``model``.

    Args:
        model: Foundry deployment name, not the underlying model name.
        project_endpoint: HTTPS Foundry project URL. Defaults to
            ``AZURE_AI_PROJECT_ENDPOINT``, then ``FOUNDRY_PROJECT_ENDPOINT``,
            when neither endpoint is supplied. The adapter sends requests to
            the resource that hosts the project.
        endpoint: Direct HTTPS Foundry resource URL as an alternative to
            ``project_endpoint``.
        credential: API key string, ``AzureKeyCredential``, or an Entra token
            credential. Defaults to ``DefaultAzureCredential``. An async-only
            credential requires ``ainvoke`` or ``abatch``.
        timeout: HTTP timeout in seconds. Defaults to 60.
        http_client: Optional caller-owned synchronous HTTP client.
        http_async_client: Optional caller-owned asynchronous HTTP client.

    Raises:
        ValueError: If configuration, input, or response data is invalid.
        httpx.HTTPError: If invocation fails at the HTTP layer.

    Note:
        State must be text, a JSON object, or a JSON array. This adapter does not
        convert LangChain messages or send image assets. ``Box`` questions are
        unsupported. ``Choice`` requires at least two labels.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True
    )

    model: str = Field(min_length=1)
    project_endpoint: str | None = None
    endpoint: str | None = None
    credential: (
        str | SecretStr | AzureKeyCredential | TokenCredential | AsyncTokenCredential
    ) = Field(default_factory=DefaultAzureCredential, exclude=True, repr=False)
    timeout: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    http_client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    http_async_client: httpx.AsyncClient | None = Field(
        default=None, exclude=True, repr=False
    )

    _sync_token: Callable[[], str] | None = PrivateAttr(default=None)
    _async_token: Callable[[], Awaitable[str]] | None = PrivateAttr(default=None)
    _url: str = PrivateAttr()

    @field_validator("credential", mode="before")
    @classmethod
    def _mask_key(cls, credential: Any) -> Any:
        return SecretStr(credential) if isinstance(credential, str) else credential

    @model_validator(mode="after")
    def _configure_endpoint(self) -> "AzureAIMicrosoftDecisionsClassifier":
        if self.endpoint is not None and self.project_endpoint is not None:
            raise ValueError("Provide only one of `endpoint` and `project_endpoint`.")
        if self.endpoint is None and self.project_endpoint is None:
            self.project_endpoint = get_project_endpoint()
        try:
            url = httpx.URL(self.endpoint or self.project_endpoint or "")
        except httpx.InvalidURL:
            raise ValueError("Provide a valid HTTPS Foundry endpoint.") from None
        if (
            url.scheme != "https"
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Provide a valid HTTPS Foundry endpoint.")
        if self.project_endpoint is not None:
            if not re.fullmatch(r"/api/projects/[^/]+/?", url.path):
                raise ValueError(
                    "Provide a Foundry project URL ending in /api/projects/<name>."
                )
        elif url.path != "/":
            raise ValueError("Provide the HTTPS Foundry resource endpoint.")
        self._url = _get_base_url_from_endpoint(str(url)) + _DECISION_PATH
        return self

    @model_validator(mode="after")
    def _configure_auth(self) -> "AzureAIMicrosoftDecisionsClassifier":
        credential = self.credential
        if isinstance(credential, (TokenCredential, AsyncTokenCredential)):
            self._async_token = _make_async_token_provider(credential, _TOKEN_SCOPE)
            if not isinstance(credential, AsyncTokenCredential):
                self._sync_token = _make_token_provider(credential, _TOKEN_SCOPE)
        elif not self._key().strip():
            raise ValueError("The API key must not be empty.")
        return self

    @property
    def _provider(self) -> str:
        return "microsoft_foundry"

    def _key(self) -> str:
        credential = self.credential
        if isinstance(credential, SecretStr):
            return credential.get_secret_value()
        if isinstance(credential, AzureKeyCredential):
            return credential.key
        raise ValueError("An async credential requires ainvoke or abatch.")

    def _body(self, state: State, questions: dict[str, Question]) -> dict[str, Any]:
        if not questions:
            raise ValueError("Provide at least one decision question.")
        try:
            json_state = _JSON_STATE.validate_python(state, strict=True)
        except ValidationError:
            raise ValueError(
                "State must be text, a JSON object, or a JSON array."
            ) from None
        for question in questions.values():
            if not isinstance(question, (Noul, Choice, Score)):
                raise ValueError(
                    "Microsoft Decisions supports Noul, Choice, and Score."
                )
            if isinstance(question, Choice) and len(question.criteria) < 2:
                raise ValueError("Choice requires at least two labels.")
        return {
            "model": self.model,
            "state": json_state,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in questions.items()
            },
        }

    @staticmethod
    def _response(
        response: httpx.Response, questions: dict[str, Question]
    ) -> ClassifierResponse:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(
            payload.get("answers"), dict
        ):
            raise ValueError("The Microsoft Decisions response needs named answers.")
        answers = payload["answers"]
        if answers.keys() != questions.keys():
            raise ValueError("Microsoft Decisions returned the wrong set of answers.")
        for name, question in questions.items():
            answer = answers[name]
            if not isinstance(answer, dict) or answer.get("type") != question.type:
                raise ValueError("A Microsoft Decisions answer has the wrong type.")
            if isinstance(question, Score):
                answer.setdefault("legend", dict(enumerate(question.criteria)))
        payload["request_id"] = (
            response.headers.get("x-request-id")
            or response.headers.get("apim-request-id")
            or response.headers.get("x-ms-request-id")
        )
        result = ClassifierResponse.model_validate(payload)

        for name, question in questions.items():
            answer = result.answers[name]
            if isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
                if (
                    answer.choice not in question.criteria
                    or answer.probabilities.keys() != question.criteria.keys()
                ):
                    raise ValueError(
                        "A Microsoft Decisions choice has unexpected labels."
                    )
            if isinstance(question, Score) and isinstance(answer, ScoreAnswer):
                levels = set(range(len(question.criteria)))
                if (
                    answer.probabilities.keys() != levels
                    or answer.legend.keys() != levels
                    or not math.isfinite(answer.score)
                ):
                    raise ValueError("A Microsoft Decisions score answer is invalid.")
            if isinstance(answer, (ChoiceAnswer, ScoreAnswer)) and any(
                not 0 <= probability <= 1
                for probability in answer.probabilities.values()
            ):
                raise ValueError("A Microsoft Decisions probability is outside [0, 1].")
        return result

    def _classify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        body = self._body(state, questions)
        headers = (
            {"Authorization": f"Bearer {self._sync_token()}"}
            if self._sync_token is not None
            else {"api-key": self._key()}
        )
        client_context = (
            nullcontext(self.http_client) if self.http_client else httpx.Client()
        )
        with client_context as client:
            response = client.post(
                self._url,
                json=body,
                headers={"x-ms-useragent": "langchain-azure-ai", **headers},
                timeout=self.timeout,
                follow_redirects=False,
            )
        return self._response(response, questions)

    async def _aclassify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        body = self._body(state, questions)
        headers = (
            {"Authorization": f"Bearer {await self._async_token()}"}
            if self._async_token is not None
            else {"api-key": self._key()}
        )
        client_context = (
            nullcontext(self.http_async_client)
            if self.http_async_client
            else httpx.AsyncClient()
        )
        async with client_context as client:
            response = await client.post(
                self._url,
                json=body,
                headers={"x-ms-useragent": "langchain-azure-ai", **headers},
                timeout=self.timeout,
                follow_redirects=False,
            )
        return self._response(response, questions)
