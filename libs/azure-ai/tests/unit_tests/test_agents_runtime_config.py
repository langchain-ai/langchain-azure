"""Unit tests for langchain_azure_ai.agents.runtime._config."""

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out azure.ai.agentserver.invocations and starlette so that importing
# the runtime module never fails due to missing optional dependencies.
# ---------------------------------------------------------------------------


def _make_agentserver_stub() -> None:
    for mod_name in ("azure.ai.agentserver", "azure.ai.agentserver.invocations"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    class _FakeHost:
        def __init__(self) -> None:
            self._handler: Any = None

        def invoke_handler(self, func: Any) -> Any:
            self._handler = func
            return func

        def run(self, **kwargs: Any) -> None:
            pass

    sys.modules[
        "azure.ai.agentserver.invocations"
    ].InvocationAgentServerHost = _FakeHost  # type: ignore[attr-defined]


def _make_starlette_stub() -> None:
    for mod_name in ("starlette", "starlette.requests", "starlette.responses"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    class _FakeRequest:
        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        async def json(self) -> dict[str, Any]:
            return self._body

    class _FakeJSONResponse:
        def __init__(self, content: Any) -> None:
            self.content = content

    sys.modules["starlette.requests"].Request = _FakeRequest  # type: ignore[attr-defined]
    sys.modules["starlette.responses"].Response = object  # type: ignore[attr-defined]
    sys.modules["starlette.responses"].JSONResponse = _FakeJSONResponse  # type: ignore[attr-defined]


_make_agentserver_stub()
_make_starlette_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_langgraph_json(tmp_path: Path, content: dict[str, Any]) -> Path:
    cfg = tmp_path / "langgraph.json"
    cfg.write_text(json.dumps(content), encoding="utf-8")
    return cfg


def _write_graph_module(tmp_path: Path) -> Path:
    """Write a minimal graph.py that exposes a ``graph`` variable."""
    graph_file = tmp_path / "graph.py"
    graph_file.write_text(
        "class _FakeGraph:\n"
        "    async def ainvoke(self, state, config=None):\n"
        "        return state\n"
        "graph = _FakeGraph()\n",
        encoding="utf-8",
    )
    return graph_file


# ---------------------------------------------------------------------------
# Tests: load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for _config.load_config."""

    def test_valid_config(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_config

        cfg_path = _write_langgraph_json(
            tmp_path, {"graphs": {"agent": "./graph.py:graph"}}
        )
        result = load_config(cfg_path)
        assert result["graphs"] == {"agent": "./graph.py:graph"}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_config

        with pytest.raises(FileNotFoundError, match="langgraph.json not found"):
            load_config(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_config

        bad = tmp_path / "langgraph.json"
        bad.write_text("not-json", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_config(bad)

    def test_missing_graphs_key_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_config

        cfg_path = _write_langgraph_json(tmp_path, {"env": ".env"})
        with pytest.raises(ValueError, match="non-empty 'graphs' mapping"):
            load_config(cfg_path)

    def test_empty_graphs_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_config

        cfg_path = _write_langgraph_json(tmp_path, {"graphs": {}})
        with pytest.raises(ValueError, match="non-empty 'graphs' mapping"):
            load_config(cfg_path)


# ---------------------------------------------------------------------------
# Tests: load_env
# ---------------------------------------------------------------------------


class TestLoadEnv:
    """Tests for _config.load_env."""

    def test_loads_env_file(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_env

        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_VAR=hello\n", encoding="utf-8")

        fake_load_dotenv = MagicMock()
        with patch(
            "langchain_azure_ai.agents.runtime._config.load_dotenv",
            fake_load_dotenv,
            create=True,
        ):
            # Patch the import inside load_env
            dotenv_mod = types.ModuleType("dotenv")
            dotenv_mod.load_dotenv = fake_load_dotenv  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {"dotenv": dotenv_mod}):
                load_env(env_file)

        fake_load_dotenv.assert_called_once_with(dotenv_path=env_file, override=False)

    def test_missing_env_file_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_env

        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = MagicMock()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"dotenv": dotenv_mod}):
            with pytest.raises(FileNotFoundError, match=".env file not found"):
                load_env(tmp_path / ".env")

    def test_missing_dotenv_package_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import load_env

        env_file = tmp_path / ".env"
        env_file.write_text("X=1\n", encoding="utf-8")

        # Setting sys.modules["dotenv"] = None marks the module as explicitly
        # unavailable, causing any `import dotenv` to raise ImportError.
        saved = sys.modules.get("dotenv", ...)
        sys.modules["dotenv"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="python-dotenv"):
                load_env(env_file)
        finally:
            if saved is ...:
                del sys.modules["dotenv"]
            else:
                sys.modules["dotenv"] = saved  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests: import_graph
# ---------------------------------------------------------------------------


class TestImportGraph:
    """Tests for _config.import_graph."""

    def test_file_based_reference(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import import_graph

        graph_file = _write_graph_module(tmp_path)
        graph = import_graph(f"./{graph_file.name}:graph", tmp_path)
        assert hasattr(graph, "ainvoke")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import import_graph

        with pytest.raises(ImportError, match="Graph file not found"):
            import_graph("./nonexistent.py:graph", tmp_path)

    def test_missing_attr_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import import_graph

        graph_file = _write_graph_module(tmp_path)
        with pytest.raises(AttributeError):
            import_graph(f"./{graph_file.name}:no_such_attr", tmp_path)

    def test_invalid_ref_format_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import import_graph

        with pytest.raises(ValueError, match="Invalid graph reference"):
            import_graph("./graph.py", tmp_path)


# ---------------------------------------------------------------------------
# Tests: resolve_graph
# ---------------------------------------------------------------------------


class TestResolveGraph:
    """Tests for _config.resolve_graph."""

    def test_single_graph_no_name_needed(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import resolve_graph

        graph_file = _write_graph_module(tmp_path)
        cfg = {"graphs": {"agent": f"./{graph_file.name}:graph"}}
        graph = resolve_graph(cfg, tmp_path, graph_name=None)
        assert hasattr(graph, "ainvoke")

    def test_named_graph(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import resolve_graph

        graph_file = _write_graph_module(tmp_path)
        cfg = {"graphs": {"agent": f"./{graph_file.name}:graph"}}
        graph = resolve_graph(cfg, tmp_path, graph_name="agent")
        assert hasattr(graph, "ainvoke")

    def test_multiple_graphs_require_name(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import resolve_graph

        graph_file = _write_graph_module(tmp_path)
        ref = f"./{graph_file.name}:graph"
        cfg = {"graphs": {"a": ref, "b": ref}}
        with pytest.raises(ValueError, match="multiple graphs"):
            resolve_graph(cfg, tmp_path, graph_name=None)

    def test_unknown_graph_name_raises(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime._config import resolve_graph

        cfg = {"graphs": {"agent": "./graph.py:graph"}}
        with pytest.raises(ValueError, match="not found in langgraph.json"):
            resolve_graph(cfg, tmp_path, graph_name="missing")


# ---------------------------------------------------------------------------
# Tests: AzureAIAgentServerRuntime.from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    """Integration tests for AzureAIAgentServerRuntime.from_config."""

    def test_from_config_single_graph(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph_file = _write_graph_module(tmp_path)
        _write_langgraph_json(
            tmp_path, {"graphs": {"agent": f"./{graph_file.name}:graph"}}
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runtime = AzureAIAgentServerRuntime.from_config(
                config_path=tmp_path / "langgraph.json"
            )

        assert runtime._host is not None

    def test_from_config_loads_env(self, tmp_path: Path) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph_file = _write_graph_module(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("RUNTIME_TEST_VAR=42\n", encoding="utf-8")
        _write_langgraph_json(
            tmp_path,
            {"graphs": {"agent": f"./{graph_file.name}:graph"}, "env": ".env"},
        )

        fake_load_dotenv = MagicMock()
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = fake_load_dotenv  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"dotenv": dotenv_mod}):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                AzureAIAgentServerRuntime.from_config(
                    config_path=tmp_path / "langgraph.json"
                )

        fake_load_dotenv.assert_called_once_with(dotenv_path=env_file, override=False)

    def test_from_config_missing_file_raises(self, tmp_path: Path) -> None:
        import warnings

        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FileNotFoundError, match="langgraph.json not found"):
                AzureAIAgentServerRuntime.from_config(
                    config_path=tmp_path / "langgraph.json"
                )
