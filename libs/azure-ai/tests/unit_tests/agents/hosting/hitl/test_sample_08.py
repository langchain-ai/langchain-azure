"""Regression tests against the actual approval-style HITL sample graph."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_azure_ai.agents.hosting import ResponsesHostServer

from .conftest import (
    REAL_INTERRUPT_ASYNC_XFAIL,
    approval_requests,
    assistant_text,
    client_for,
    resume_item,
    sentinels,
)


@pytest.fixture
def hitl_sample() -> ModuleType:
    sample_path = (
        Path(__file__).resolve().parents[7]
        / "samples/hosting/langgraph-hosted-agents/responses/08_hitl/main.py"
    )
    spec = importlib.util.spec_from_file_location("hitl_sample_08", sample_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("locations", [["Seattle"], ["Seattle", "Paris"]])
@pytest.mark.parametrize("override", [False, True])
def test_approves_every_tool_call_across_turns(
    hitl_sample: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    locations: list[str],
    override: bool,
) -> None:
    model = Mock()
    model.bind_tools.return_value = model
    weather_tool = Mock(wraps=hitl_sample.get_weather)
    monkeypatch.setattr(hitl_sample, "_build_chat_model", lambda: model)
    monkeypatch.setitem(hitl_sample._TOOLS_BY_NAME, "get_weather", weather_tool)
    graph = hitl_sample._build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "sample-multiturn"}}

    for turn in range(2):
        model.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"weather-{turn}-{location}",
                        "name": "get_weather",
                        "args": {"location": location},
                    }
                    for location in locations
                ],
            ),
            AIMessage(content=f"Forecasts for turn {turn}."),
        ]
        weather_tool.invoke.reset_mock()
        model.invoke.reset_mock()
        prompt = f"Turn {turn}: weather in {', '.join(locations)}?"
        result = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        assert model.invoke.call_args.args[0][-1].content == prompt

        for location in locations:
            pending = result.get("__interrupt__", ())
            assert len(pending) == 1, result
            assert pending[0].value == {
                "tool": "get_weather",
                "arguments": {"location": location},
            }
            weather_tool.invoke.assert_not_called()
            assert model.invoke.call_count == 1
            approved = pending[0].value
            if override and location == locations[-1]:
                approved = {
                    "tool": "get_weather",
                    "arguments": {"location": "Vancouver"},
                }
            result = graph.invoke(Command(resume=approved), config)

        assert not result.get("__interrupt__")
        assert model.invoke.call_count == 2
        assert weather_tool.invoke.call_count == len(locations)
        tool_messages = [
            message
            for message in model.invoke.call_args.args[0]
            if isinstance(message, ToolMessage)
        ][-len(locations) :]
        assert [message.tool_call_id for message in tool_messages] == [
            f"weather-{turn}-{location}" for location in locations
        ]
        expected_locations = locations[:-1] + [
            "Vancouver" if override else locations[-1]
        ]
        assert [message.content for message in tool_messages] == [
            f"It's sunny and 22C in {location}." for location in expected_locations
        ]


@REAL_INTERRUPT_ASYNC_XFAIL
@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
def test_host_resumes_later_calls_and_new_turns(
    hitl_sample: ModuleType, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"weather-{location}",
                    "name": "get_weather",
                    "args": {"location": location},
                }
                for location in ["Seattle", "Paris"]
            ],
        ),
        AIMessage(content="Forecasts ready."),
        AIMessage(content="Latest turn received."),
    ]
    weather_tool = Mock(wraps=hitl_sample.get_weather)
    monkeypatch.setattr(hitl_sample, "_build_chat_model", lambda: model)
    monkeypatch.setitem(hitl_sample._TOOLS_BY_NAME, "get_weather", weather_tool)
    host = ResponsesHostServer(hitl_sample._build_graph(InMemorySaver()))
    conversation = {"id": "sample-host-multiturn"}
    with client_for(host) as client:
        response = client.post(
            "/responses",
            json={
                "input": "Weather in Seattle and Paris?",
                "conversation": conversation,
            },
        )
        for location in ["Seattle", "Paris"]:
            assert response.status_code == 200, response.text
            pending = approval_requests(response.json())
            assert len(pending) == 1, response.text
            assert json.loads(pending[0]["arguments"])["value"] == {
                "tool": "get_weather",
                "arguments": {"location": location},
            }
            weather_tool.invoke.assert_not_called()
            if decision == "edit" and location == "Paris":
                resume = resume_item(
                    sentinels(response.json())[0]["call_id"],
                    {"tool": "get_weather", "arguments": {"location": "Vancouver"}},
                )
            else:
                resume = {
                    "type": "mcp_approval_response",
                    "approval_request_id": pending[0]["id"],
                    "approve": not (decision == "reject" and location == "Paris"),
                }
            response = client.post(
                "/responses", json={"input": [resume], "conversation": conversation}
            )

        assert response.status_code == 200, response.text
        payload = response.json()
        if decision == "reject":
            assert payload["status"] == "failed", payload
            assert payload["error"]["code"] == "interrupt_rejected", payload
            weather_tool.invoke.assert_not_called()
            assert model.invoke.call_count == 1
            return

        assert payload["status"] == "completed", payload
        assert weather_tool.invoke.call_count == 2
        assert weather_tool.invoke.call_args.args[0] == {
            "location": "Vancouver" if decision == "edit" else "Paris"
        }
        response = client.post(
            "/responses", json={"input": "Now Tokyo?", "conversation": conversation}
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed", response.text
        assert "Latest turn received." in assistant_text(response.json())
        assert model.invoke.call_args.args[0][-1].content == "Now Tokyo?"
