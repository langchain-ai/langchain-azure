# Copyright (c) Microsoft. All rights reserved.

"""Deterministic orchestration protocol contracts for errors and live upgrades."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from langchain_azure_compute.graph import runner as r


class Context:
    def __init__(self):
        self.statuses, self.rollovers = [], []

    def call_activity(self, name, input, retry_policy):
        return NS(name=name, input=input, retry=retry_policy)

    def wait_for_external_event(self, name):
        return NS(event=name)

    def set_custom_status(self, value):
        self.statuses.append(value)

    def continue_as_new(self, payload, save_events=False):
        self.rollovers.append(payload)


def initial():
    return {
        "thread": "t",
        "req": {"thread": "t", "kind": "start", "expect": None, "input": r.pack({})},
        "step_name": "langgraph.g.step",
        "tool_name": "langgraph.g.tool",
        "node_name": "langgraph.g.node",
        "version": "v1",
        "owner": "alice",
    }


def activity_report(kind, mixed=False, two=False):
    specs = []
    for index in range(2 if mixed or two else 1):
        common = {
            "index": index,
            "call_id": f"call-{index}",
            "task_id": f"task-{index}",
            "name": "node",
            "durable": index == 0 or two,
            "retries": 1,
        }
        if kind == "tools":
            common.update(
                call=r.pack({"name": "tool", "args": {}, "id": str(index)}),
                input_type="dict",
                handle_errors=True,
            )
        else:
            common["input"] = r.pack({"n": index})
        specs.append(common)
    report = {"status": kind, "ckpt": "c", "max_concurrency": 1}
    if kind == "tools":
        report.update(node="tools", tool_calls=specs, tool_ns="sub", tool_ckpt="inner")
    else:
        report["node_tasks"] = specs
    return report


@pytest.mark.parametrize("kind", ["tools", "nodes"])
@pytest.mark.parametrize("jump", [False, True])
@pytest.mark.parametrize("rollover", [False, True])
def test_subtask_upgrade_rewrites_dispatch_or_commits_without_reexecuting(
    kind, jump, rollover
):
    ctx = Context()
    inp = initial()
    inp.update(max_concurrency=1, node_max_concurrency=1)
    limits = r.Limits(steps_per_generation=1 if rollover else 100)
    gen = r.make_orchestrator(limits)(ctx, inp)
    next(gen)
    work = gen.send(activity_report(kind, mixed=True))
    assert work.input["call_id"] == "call-0" and work.input["owner"] == "alice"
    event = gen.send(
        {"status": "interrupted", "interrupt": {"id": "i", "value": r.pack(1)}}
    )
    assert event.event == "resume:i"
    work = gen.send(
        {
            "value": r.pack(True),
            "version": "v2",
            "as_node": "new",
            "migrate_values": r.pack({"n": 3}),
        }
    )
    assert work.input["kind"] == "migrate"
    assert work.input["as_node"] == "new" and work.input["migrate_from"] == "v1"
    migration = {
        "status": "running",
        "ckpt": "new",
        "jumped": jump,
        "checkpoint_bundle": {"v": 1},
        "target_version": "v3",
    }
    if jump:
        if rollover:
            with pytest.raises(StopIteration):
                gen.send(migration)
            assert ctx.rollovers[0]["version"] == "v3"
        else:
            next_work = gen.send(migration)
            assert next_work.input["kind"] == "step"
            with pytest.raises(StopIteration) as completed:
                gen.send({"status": "done", "ckpt": "done"})
            assert completed.value.value["version"] == "v2"
    else:
        resumed = gen.send(migration)
        assert resumed.input["resumes"] == [r.pack(True)]
        assert resumed.input["ckpt"] == "new"
        commit = gen.send({"status": "ok", "output": r.pack({"n": 3})})
        assert commit.input["kind"] == kind
        assert commit.input["outputs"] == {"0": r.pack({"n": 3})}
        with pytest.raises(StopIteration) as completed:
            gen.send(
                {
                    "status": "done",
                    "ckpt": "done",
                    "result": r.pack(3),
                    "target_version": "v3",
                    "checkpoint_bundle": {"v": 1},
                }
            )
        assert completed.value.value["result"] == r.pack(3)


@pytest.mark.parametrize("kind", ["tools", "nodes"])
def test_subtask_migration_error_fails_closed(kind):
    ctx = Context()
    gen = r.make_orchestrator(r.Limits())(ctx, initial())
    next(gen)
    gen.send(activity_report(kind))
    gen.send({"status": "interrupted", "interrupt": {"id": "i", "value": r.pack(1)}})
    gen.send({"value": r.pack(1), "version": "v2"})
    with pytest.raises(r.RunError, match="migration failed"):
        gen.send({"status": "error", "error": "migration failed"})


@pytest.mark.parametrize(
    "stage", ["first", "subtask", "commit", "inner_limit", "outer_limit"]
)
@pytest.mark.parametrize("kind", ["tools", "nodes"])
def test_orchestration_error_envelopes_and_step_limits(stage, kind):
    ctx = Context()
    inp = initial()
    if stage == "outer_limit":
        inp["steps"] = 1
    gen = r.make_orchestrator(
        r.Limits(max_steps=1 if stage.endswith("limit") else 100)
    )(ctx, inp)
    if stage == "outer_limit":
        with pytest.raises(r.RunError, match="step limit"):
            next(gen)
        return
    next(gen)
    if stage == "first":
        with pytest.raises(r.RunError, match="failed"):
            gen.send({"status": "error", "error": "failed"})
        return
    if stage == "inner_limit":
        with pytest.raises(r.RunError, match="step limit"):
            gen.send(activity_report(kind))
        return
    gen.send(activity_report(kind))
    if stage == "subtask":
        with pytest.raises(r.RunError, match="failed"):
            gen.send({"status": "error", "error": "failed"})
    else:
        gen.send({"status": "ok", "output": r.pack({})})
        with pytest.raises(r.RunError, match="failed"):
            gen.send({"status": "error", "error": "failed"})


def test_tool_error_policy_transports_exhausted_failures_and_unhandled_errors():
    for handled in (False, True):
        report = activity_report("tools")
        report["tool_calls"][0]["handle_errors"] = handled
        ctx = Context()
        gen = r.make_orchestrator(r.Limits())(ctx, initial())
        next(gen)
        gen.send(report)
        if not handled:
            with pytest.raises(ValueError, match="tool error"):
                gen.throw(ValueError("tool error"))
        else:
            commit = gen.throw(ValueError("tool error"))
            error = r.unpack(commit.input["outputs"][0])[r.DURABLE_TOOL_ERROR_KEY]
            assert error["error_type"] == "ValueError"
            assert error["message"] == "tool error"
            gen.close()


@pytest.mark.parametrize("partial", [False, True])
def test_node_failure_commits_only_already_completed_siblings(partial):
    ctx = Context()
    gen = r.make_orchestrator(r.Limits())(ctx, initial())
    next(gen)
    report = activity_report("nodes", two=True)
    report["max_concurrency"] = 2
    gen.send(report)
    if partial:
        gen.send({"status": "ok", "output": r.pack({"n": 1})})
        commit = gen.throw(ValueError("node error"))
        assert commit.input["partial"] is True
        assert commit.input["outputs"] == {"0": r.pack({"n": 1})}
        with pytest.raises(ValueError, match="node error"):
            gen.send({"status": "nodes", "ckpt": "c"})
    else:
        with pytest.raises(ValueError, match="node error"):
            gen.throw(ValueError("node error"))


def test_event_upgrade_formats_and_packed_map_validation():
    assert r._extract_event_upgrade(None) == (None, None, None)
    assert r._extract_event_upgrade({"upgrade": "v2"}) == ("v2", None, None)
    assert r._extract_event_upgrade(
        {
            "upgrade": {
                "version": "v2",
                "as_node": "node",
                "values": r.pack(1),
            }
        }
    ) == ("v2", "node", r.pack(1))
    assert r._extract_event_upgrade({"upgrade": {"migrate_values": r.pack(2)}}) == (
        None,
        None,
        r.pack(2),
    )
    assert r._is_valid_packed_blob({"__resume_map__": {"i": r.pack(1)}})
    assert not r._is_valid_packed_blob({"__resume_map__": {"i": {}}})
    assert r._extract_failure_info(
        NS(details=NS(message="bad", error_type="ValueError"))
    ) == {
        "error_type": "ValueError",
        "message": "bad",
    }
    assert r._cap_interrupt_for_status({"id": "i", "value": r.pack("x" * 100)}, 1) == {
        "id": "i",
        "oversized": True,
    }
    payload = r._build_can_payload(
        thread="t",
        generation=1,
        steps=3,
        request={"thread": "t", "kind": "inspect", "expect": "c"},
        step_name="step",
        tool_name="tool",
        node_name="node",
    )
    assert payload["gen"] == 1 and payload["steps"] == 3


def test_stale_and_malformed_events_are_ignored_until_valid_resume():
    ctx = Context()
    gen = r._wait_for_resume_event(ctx, "i", "expected", 0)
    next(gen)
    for malformed in (None, {}, {"value": {}}, {"value": r.pack(1), "ckpt": "stale"}):
        assert gen.send(malformed).event == "resume:i"
    with pytest.raises(StopIteration) as result:
        gen.send({"value": r.pack(True), "ckpt": "expected"})
    assert result.value.value[0]["value"] == r.pack(True)
