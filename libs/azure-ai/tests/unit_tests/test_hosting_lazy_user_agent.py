"""Fresh-process checks for lazy model imports and outgoing User-Agent headers."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _run_fresh_process(source: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    env.pop("LANGCHAIN_AZURE_AI_USER_AGENT_DISABLED", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=PACKAGE_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("import_order", ["hosting_first", "sdk_first"])
def test_user_agent_installed_for_either_import_order(
    provider: str, import_order: str
) -> None:
    _run_fresh_process(
        """
        import importlib
        import sys

        provider, order = sys.argv[1:]
        if order == "sdk_first":
            sdk = importlib.import_module(provider)

        import langchain_azure_ai.agents.hosting as hosting
        if order == "hosting_first":
            assert "openai" not in sys.modules
            assert "anthropic" not in sys.modules
            sdk = importlib.import_module(provider)
        other = "anthropic" if provider == "openai" else "openai"
        assert other not in sys.modules

        cls = sdk.OpenAI if provider == "openai" else sdk.Anthropic
        with cls(api_key="test-key") as client:
            ua = client.default_headers["User-Agent"]
            assert ua.count(hosting.HOSTING_USER_AGENT) == 1
        """,
        provider,
        import_order,
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_langchain_client_created_later_sends_dynamic_user_agent(
    provider: str, mode: str
) -> None:
    _run_fresh_process(
        """
        import asyncio
        import sys
        from unittest.mock import patch

        import langchain_azure_ai.agents.hosting as hosting

        assert "openai" not in sys.modules
        assert "anthropic" not in sys.modules

        import httpx
        from langchain_azure_ai._user_agent import BASE_USER_AGENT

        provider, mode = sys.argv[1:]
        sent_headers = []

        def respond(request):
            sent_headers.append(dict(request.headers))
            if provider == "openai":
                body = {
                    "id": "test", "object": "chat.completion", "created": 0,
                    "model": "test", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                    },
                }
            else:
                body = {
                    "id": "test", "type": "message", "role": "assistant",
                    "model": "test", "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            return httpx.Response(200, json=body, request=request)

        sync_http = httpx.Client(transport=httpx.MockTransport(respond))
        async_http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        options = dict(
            model="test", api_key="test-key",
            default_headers={"User-Agent": "test-app/1.0", "X-Test": "preserved"},
        )

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(**options, http_client=sync_http,
                               http_async_client=async_http, use_responses_api=False)
            assert "anthropic" not in sys.modules
        else:
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(**options)
            # Replace transports only; LangChain constructs the actual SDK clients.
            with patch("langchain_anthropic.chat_models._get_default_httpx_client",
                       return_value=sync_http):
                assert model._client
            with patch(
                "langchain_anthropic.chat_models._get_default_async_httpx_client",
                return_value=async_http,
            ):
                assert model._async_client
            assert "openai" not in sys.modules

        async def invoke():
            if mode == "async":
                return await model.ainvoke("hello")
            return model.invoke("hello")

        async def check_requests():
            try:
                assert (await invoke()).content == "ok"
                with hosting._hosting_feature_scope(hosting.HostingFeature.RESPONSES):
                    hosting._add_request_hosting_features(hosting.HostingFeature.HITL)
                    assert (await invoke()).content == "ok"
                assert len(sent_headers) == 2
                for headers, mask in zip(sent_headers, ("0", "5")):
                    ua = headers["user-agent"]
                    assert ua.count(hosting.HOSTING_USER_AGENT) == 1, ua
                    assert BASE_USER_AGENT in ua, ua
                    assert "(features=" + mask + ")" in ua, ua
                    assert ua.endswith("test-app/1.0"), ua
                    assert headers["x-test"] == "preserved"
            finally:
                sync_http.close()
                await async_http.aclose()

        asyncio.run(check_requests())
        """,
        provider,
        mode,
    )
