from __future__ import annotations

from typing import Self
from unittest.mock import AsyncMock

import client as client_module
import httpx
import pytest
from client import _openai_base_url, _openai_default_query, build_parser
from openai import APIStatusError, AsyncOpenAI


def test_responses_arguments_are_accepted() -> None:
    args = build_parser().parse_args(
        [
            "--url",
            "https://example.test/openai/responses",
            "--auth",
            "--reconnect-timeout",
            "300",
        ]
    )

    assert args.url == "https://example.test/openai/responses"
    assert args.auth is True
    assert args.reconnect_timeout == 300.0


def test_responses_argument_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.url == "http://127.0.0.1:8088"
    assert args.auth is False
    assert args.reconnect_timeout == 120.0


def test_responses_url_accepts_host_or_full_endpoint() -> None:
    assert _openai_base_url("http://127.0.0.1:8088") == "http://127.0.0.1:8088"
    assert _openai_base_url("https://example.test/responses") == (
        "https://example.test"
    )
    hosted_url = (
        "https://example.test/endpoint/protocols/openai/responses?api-version=v1"
    )
    assert _openai_base_url(hosted_url) == (
        "https://example.test/endpoint/protocols/openai"
    )
    assert _openai_default_query(hosted_url, authenticated=True) == {
        "api-version": "v1"
    }
    assert _openai_default_query(
        "https://example.test/endpoint/protocols/openai/responses",
        authenticated=True,
    ) == {"api-version": "v1"}


@pytest.mark.asyncio
async def test_hosted_endpoint_builds_expected_openai_request() -> None:
    endpoint = (
        "https://example.test/endpoint/protocols/openai/responses?api-version=v1"
    )
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400,
            request=request,
            json={"error": {"message": "stop"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(capture)) as http:
        async with AsyncOpenAI(
            base_url=_openai_base_url(endpoint),
            api_key="unused",
            default_query=_openai_default_query(endpoint, authenticated=True),
            http_client=http,
            max_retries=0,
        ) as client:
            with pytest.raises(APIStatusError):
                await client.responses.create(input="hello")

    assert str(requests[0].url) == endpoint


@pytest.mark.asyncio
async def test_local_mode_constructs_direct_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_options: dict[str, object] = {}
    conversation_options: dict[str, object] = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs: object) -> None:
            client_options.update(kwargs)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    def create_conversation(*args: object, **kwargs: object) -> object:
        conversation_options.update(kwargs)
        return conversation

    run_app = AsyncMock()
    conversation = object()
    monkeypatch.setattr(client_module, "uuid4", lambda: "generated-conversation")
    monkeypatch.setattr(client_module, "AsyncOpenAI", FakeOpenAIClient)
    monkeypatch.setattr(client_module, "Conversation", create_conversation)
    monkeypatch.setattr(client_module, "ResponsesCuiApp", lambda value: run_app)

    args = build_parser().parse_args([])
    await client_module.amain(args)

    assert client_options == {
        "base_url": "http://127.0.0.1:8088",
        "api_key": "local",
        "default_query": {},
        "max_retries": 0,
    }
    assert conversation_options == {
        "conversation_id": "generated-conversation",
        "reconnect_timeout": 120.0,
    }
    run_app.run_async.assert_awaited_once()
