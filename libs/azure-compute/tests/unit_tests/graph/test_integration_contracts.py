# Copyright (c) Microsoft. All rights reserved.

"""Offline contract tests for the emulator integration harness."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def integration_module():
    path = Path(__file__).resolve().parents[2] / "integration_tests" / "graph"
    spec = importlib.util.spec_from_file_location(
        "emulator_contract", path / "test_emulator_samples.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("https://scheduler.example", ("scheduler.example", 443)),
        ("http://127.0.0.1:8080", ("127.0.0.1", 8080)),
        ("true", ("127.0.0.1", 8080)),
        ("127.0.0.1:8080", ("127.0.0.1", 8080)),
    ],
)
def test_emulator_endpoint_ports(endpoint, expected, monkeypatch):
    mod = integration_module()
    monkeypatch.setenv("DTS_ENDPOINT", endpoint)
    seen = []

    class Connection:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    def connect(address, **kwargs):
        seen.append(address)
        return Connection()

    monkeypatch.setattr(mod.socket, "create_connection", connect)
    assert mod._emulator_endpoint()
    assert seen == [expected]


def test_explicit_unreachable_emulator_fails(monkeypatch):
    mod = integration_module()
    monkeypatch.setenv("DTS_ENDPOINT", "127.0.0.1:1")

    def fail(*args, **kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(mod.socket, "create_connection", fail)
    with pytest.raises(pytest.fail.Exception, match="explicit|Explicit"):
        mod._emulator_endpoint()
    monkeypatch.delenv("DTS_ENDPOINT")
    monkeypatch.delenv("DTS_EMULATOR", raising=False)
    assert mod._emulator_endpoint() is None


@pytest.mark.parametrize("fail", [False, True])
def test_roundtrip_closes_client_on_success_and_failure(monkeypatch, fail):
    mod = integration_module()
    closed = []

    class Worker:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def start(self):
            pass

    class Sdk:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append(True)

    def wait_result(**kwargs):
        if fail:
            raise RuntimeError("run failed")
        return {"item": "laptop", "approved": True}

    run = SimpleNamespace(
        wait_pending=lambda **kwargs: SimpleNamespace(value={"item": "laptop"}),
        resume=lambda *args: None,
        wait_result=wait_result,
    )
    reg = SimpleNamespace(
        register=lambda worker: None,
        client=lambda sdk: SimpleNamespace(start=lambda inp: run),
    )
    monkeypatch.setattr(mod, "_emulator_endpoint", lambda: "127.0.0.1:8080")
    monkeypatch.setattr(mod, "DurableTaskSchedulerWorker", Worker)
    monkeypatch.setattr(mod, "DurableTaskSchedulerClient", Sdk)
    monkeypatch.setattr(mod, "attach_durability", lambda *args, **kwargs: reg)
    if fail:
        with pytest.raises(RuntimeError, match="run failed"):
            mod.test_emulator_durable_graph_and_hitl_roundtrip()
    else:
        mod.test_emulator_durable_graph_and_hitl_roundtrip()
    assert closed == [True]
