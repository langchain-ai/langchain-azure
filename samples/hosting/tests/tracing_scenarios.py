# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inputs for the tracing contract; keep sample graphs and startup code intact."""

from __future__ import annotations

from langchain_core.messages import AIMessage

# Every discovered hosted sample must have a scenario. New samples fail the
# coverage check until their protocol and representative tool path are specified.
CASES = {
    f"hosting/langgraph-hosted-agents/{sample}": (protocol, scenario)
    for sample, protocol, scenario in [
        ("responses/01_basic", "responses", "basic"),
        ("responses/02_tools", "responses", "weather"),
        ("responses/03_mcp", "responses", "remote_tool"),
        ("responses/04_foundry_toolbox", "responses", "remote_tool"),
        ("responses/05_workflows", "responses", "weather"),
        ("responses/06_files", "responses", "basic"),
        ("responses/07_observability", "responses", "basic"),
        ("responses/08_hitl", "responses", "approval"),
        ("responses/09_run", "responses", "basic"),
        ("responses/10_resilient", "responses", "resilient"),
        ("responses/11_middleware", "responses", "email"),
        ("responses/12_deep_agents", "responses", "remote_tool"),
        ("responses/13_branching", "responses", "basic"),
        ("invocations/01_basic", "invocations", "basic"),
        ("invocations/02_tools", "invocations", "weather"),
        ("invocations/03_run", "invocations", "basic"),
        ("invocations/04_resilient", "invocations", "resilient"),
    ]
}
FINAL_TEXT = "Tracing test completed"


def model_responses(scenario: str) -> list[AIMessage]:
    """Return deterministic model decisions while executing the real tools."""
    calls = {
        "basic": [],
        "weather": [[("get_weather", {"location": "Seattle"})]],
        "approval": [[("get_weather", {"location": "Seattle"})]],
        "email": [
            [
                (
                    "send_email",
                    {
                        "recipient": "test@example.invalid",
                        "subject": "Tracing",
                        "body": "Test",
                    },
                )
            ]
        ],
        "remote_tool": [[("web_search", {"query": "distributed tracing"})]],
        "resilient": [
            [
                ("search_flights", {"city": "Paris"}),
                ("search_hotels", {"city": "Paris"}),
            ],
            [("book_trip", {"city": "Paris", "nights": 2})],
        ],
    }[scenario]
    return [
        *[
            AIMessage(
                content="Calling tools",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call_{step}_{index}"}
                    for index, (name, args) in enumerate(batch)
                ],
            )
            for step, batch in enumerate(calls)
        ],
        AIMessage(content=FINAL_TEXT),
    ]
