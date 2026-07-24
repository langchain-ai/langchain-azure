# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from langchain_azure_ai.agents.hosting import run  # noqa: E402


def _write_config(directory: Path, graphs: dict[str, object]) -> None:
    (directory / "langgraph.json").write_text(
        json.dumps({"graphs": graphs}), encoding="utf-8"
    )


def test_main_hosts_only_graph_with_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    _write_config(tmp_path, {"agent": "./graph.py:graph"})
    run_server = MagicMock()
    options = object()
    options_type = MagicMock(return_value=options)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "_run_server", run_server)
    monkeypatch.setattr(run, "ResponsesServerOptions", options_type)

    run.main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--option",
            "resilient_background=true",
            "--option",
            "default_fetch_history_count=25",
        ]
    )

    loaded_graph = run_server.call_args.args[0]
    assert loaded_graph is not None
    options_type.assert_called_once_with(
        resilient_background=True, default_fetch_history_count=25
    )
    run_server.assert_called_once_with(
        loaded_graph, options, host="127.0.0.1", port=9000
    )


def test_main_selects_named_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "graphs.py").write_text(
        "first = object()\nsecond = object()\n", encoding="utf-8"
    )
    _write_config(
        tmp_path,
        {
            "first": "./graphs.py:first",
            "second": "./graphs.py:second",
        },
    )
    run_server = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "_run_server", run_server)

    run.main(["second"])

    assert run_server.call_args.args[0] is not None
    assert run_server.call_args.kwargs == {"host": "0.0.0.0", "port": None}


def test_main_loads_structured_graph_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    _write_config(
        tmp_path,
        {"agent": {"path": "./graph.py:graph", "description": "Test graph"}},
    )
    run_server = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "_run_server", run_server)

    run.main([])

    assert run_server.call_args.args[0] is not None


def test_main_loads_graph_from_custom_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    (config_directory / "graph.py").write_text(
        "graph = object()\n", encoding="utf-8"
    )
    (config_directory / "custom.json").write_text(
        json.dumps({"graphs": {"agent": "./graph.py:graph"}}), encoding="utf-8"
    )
    run_server = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "_run_server", run_server)

    run.main(["--config", "config/custom.json"])

    assert run_server.call_args.args[0] is not None


def test_load_graph_invokes_sync_factory(tmp_path: Path) -> None:
    (tmp_path / "graph.py").write_text(
        "def make_graph():\n    return {'kind': 'graph'}\n",
        encoding="utf-8",
    )

    graph = run._load_graph("./graph.py:make_graph", tmp_path)

    assert graph == {"kind": "graph"}


def test_load_graph_awaits_async_factory(tmp_path: Path) -> None:
    (tmp_path / "graph.py").write_text(
        "async def make_graph():\n    return {'kind': 'graph'}\n",
        encoding="utf-8",
    )

    graph = run._load_graph("./graph.py:make_graph", tmp_path)

    assert graph == {"kind": "graph"}


def test_load_graph_preserves_callable_graph(tmp_path: Path) -> None:
    (tmp_path / "graph.py").write_text(
        "class Graph:\n"
        "    def __call__(self):\n"
        "        raise AssertionError('graph should not be invoked')\n"
        "\n"
        "    async def astream(self):\n"
        "        yield None\n"
        "\n"
        "graph = Graph()\n",
        encoding="utf-8",
    )

    graph = run._load_graph("./graph.py:graph", tmp_path)

    assert callable(graph.astream)


def test_main_requires_name_for_multiple_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, {"first": "a:graph", "second": "b:graph"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="2"):
        run.main([])

    assert "specify one of: first, second" in capsys.readouterr().err


def test_main_rejects_unknown_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, {"agent": "graph:agent"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="2"):
        run.main(["missing"])

    assert "unknown graph 'missing'; choose one of: agent" in capsys.readouterr().err