# Copyright (c) Microsoft. All rights reserved.

"""Keep executable README and sample regressions active in docs/samples CI."""

import __future__

import ast
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "libs" / "azure-compute" / "tests" / "unit_tests" / "graph"
spec = importlib.util.spec_from_file_location(
    "durable_graph_docs_regressions", PATH / "test_review_tdd.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# Reuse the same parametrized assertions, including their pytest fixtures.
test_refund_gate = module.test_review_readme_refund_gate
test_install_commands = module.test_review_sample_root_install_includes_requirements
test_readiness_timeout = module.test_review_worker_readiness_timeout_reaps_silent_child
test_readiness_cleanup = module.test_review_worker_readiness_success_and_failure_cleanup
test_local_identity = module.test_review_sample_local_identity_requires_explicit_opt_in


def test_root_readme_durable_execution_quickstart():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    overview = text.split("## LangGraph on Azure\n", 1)[1].split("## Samples\n", 1)[0]
    assert len([line for line in overview.splitlines() if line.startswith("- **")]) == 3
    assert "**Durable Execution (Azure Durable Task Scheduler):**" in overview
    samples = text.split("## Samples\n", 1)[1].split("# Quick Start", 1)[0]
    assert "[Durable Task Scheduler (DTS) LangGraph samples]" in samples

    section = text.split(
        "### Durable Execution with Azure Durable Task Scheduler\n", 1
    )[1].split("# Contributing", 1)[0]
    assert 'pip install -U "langchain-azure-compute[durabletask]"' in section
    source = section.split("```python\n", 1)[1].split("```", 1)[0]
    tree = ast.parse(source)
    compile(tree, "<root-readme-durable-execution>", "exec")
    calls = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "create_agent",
        "sdk_options",
        "DurableTaskSchedulerWorker",
        "DurableTaskSchedulerClient",
    } <= calls.keys()
    registration = calls["attach_durability"]
    assert isinstance(registration.args[0], ast.Name)
    assert registration.args[0].id == "agent"
    assert any(
        keyword.arg == "name"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "support_agent"
        for keyword in registration.keywords
    )
    assert any(
        keyword.arg == "retries"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == 3
        for keyword in calls["durable_tool"].keywords
    )
    for parts in (
        ("libs", "azure-compute", "README.md"),
        ("samples", "langgraph-durabletask", "README.md"),
    ):
        assert ROOT.joinpath(*parts).is_file()


@pytest.mark.parametrize(
    "relative",
    [
        ("libs", "azure-compute", "README.md"),
        ("samples", "langgraph-durabletask", "05_multi_turn_conversation", "README.md"),
    ],
)
def test_persistent_checkpoint_docs_do_not_hide_durable_activity_payloads(relative):
    text = ROOT.joinpath(*relative).read_text()
    assert "step transitions pass" in text.lower()
    assert "Durable tool/node inputs and outputs" in text
    assert "final output" in text and "DTS payloads" in text
    assert "DTS carries only" not in text
    assert "history carries only" not in text


def test_sample_requirements_match_supported_minor_series():
    requirements = sorted(
        (ROOT / "samples" / "langgraph-durabletask").glob("*/requirements.txt")
    )
    assert len(requirements) == 10
    for path in requirements:
        lines = path.read_text().splitlines()
        assert "langgraph>=1.2.0,<1.3" in lines
        assert "durabletask>=1.13.0,<1.14" in lines
        assert "durabletask-azuremanaged>=1.13.0,<1.14" in lines
        assert len([line for line in lines if line.startswith("azure-functions")]) <= 1


def test_memory_step_retries_deduplicate_by_message_id():
    path = ROOT / "samples" / "langgraph-durabletask" / "05_multi_turn_conversation"
    tree = ast.parse((path / "sample.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "assistant_node"
    )
    namespace = {"AIMessage": AIMessage, "HumanMessage": HumanMessage}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )

    class Store:
        def __init__(self):
            self.items = {}

        def put(self, ns, key, value):
            self.items[ns, key] = value

        def search(self, ns, limit):
            return [
                SimpleNamespace(value=v) for (n, _), v in self.items.items() if n == ns
            ][:limit]

    store = Store()
    runtime = SimpleNamespace(store=store, context=SimpleNamespace(tenant_id="tenant"))
    node = namespace["assistant_node"]
    first = {"messages": [HumanMessage(content="fact", id="turn-1")], "user_id": "u"}
    output1 = node(first, {}, runtime)
    output2 = node(first, {}, runtime)
    assert output1 == output2
    assert "Turn #1" in output2["messages"][0].content
    second = {"messages": [HumanMessage(content="fact", id="turn-2")], "user_id": "u"}
    assert "Turn #2" in node(second, {}, runtime)["messages"][0].content
    with pytest.raises(ValueError, match="stable"):
        node({"messages": [HumanMessage(content="fact")], "user_id": "u"}, {}, runtime)


def test_booking_retry_overwrites_the_same_durable_call_id():
    from langchain_azure_compute.graph import attach_durability
    from langchain_azure_compute.graph.runner import pack, run_step, run_tool, unpack

    path = (
        ROOT / "samples" / "langgraph-durabletask" / "02_vacation_booking" / "sample.py"
    )
    spec = importlib.util.spec_from_file_location("docs_vacation", path)
    sample = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sample
    spec.loader.exec_module(sample)
    log = Path.cwd() / f".validation-bookings-{uuid.uuid4().hex}.jsonl"
    try:
        reg = attach_durability(
            sample.build_vacation_graph(log, fail_car_once=False), name="docs_bookings"
        )
        report = run_step(
            reg.graph,
            reg.limits,
            {
                "thread": "t",
                "kind": "start",
                "expect": None,
                "input": pack({"messages": [], "destination": "Kyoto"}),
            },
        )
        call = report["tool_calls"][0]
        request = {
            "thread": "t",
            "ckpt": report["ckpt"],
            "node": report["node"],
            **call,
        }
        first = run_tool(reg.graph, request)
        second = run_tool(reg.graph, request)
        assert unpack(first["output"]).content == unpack(second["output"]).content
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(records) == 1 and records[0]["call_id"] == call["call_id"]
    finally:
        log.unlink(missing_ok=True)
