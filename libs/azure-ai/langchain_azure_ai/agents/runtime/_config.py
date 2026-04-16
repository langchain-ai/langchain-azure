"""Helpers for bootstrapping AzureAIAgentServerRuntime from a langgraph.json config."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

_DOTENV_ERROR_MSG = (
    "python-dotenv is required to load .env files. Install it with: "
    "`pip install python-dotenv` or install the 'runtime' extra: "
    "`pip install langchain-azure-ai[runtime]`"
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse a ``langgraph.json`` configuration file.

    Args:
        config_path: Absolute path to the ``langgraph.json`` file.

    Returns:
        The parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is missing required fields.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"langgraph.json not found at '{config_path}'. "
            "Pass config_path= to specify its location."
        )
    try:
        data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse '{config_path}': {exc}") from exc

    if "graphs" not in data or not data["graphs"]:
        raise ValueError(f"'{config_path}' must contain a non-empty 'graphs' mapping.")
    return data


def load_env(env_path: Path) -> None:
    """Load environment variables from a ``.env`` file using python-dotenv.

    Args:
        env_path: Absolute path to the ``.env`` file.

    Raises:
        ImportError: If python-dotenv is not installed.
        FileNotFoundError: If the file does not exist.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(_DOTENV_ERROR_MSG) from exc

    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at '{env_path}'.")
    load_dotenv(dotenv_path=env_path, override=False)


def import_graph(graph_ref: str, base_dir: Path) -> Any:
    """Dynamically import a LangGraph graph from a ``<file>:<attr>`` reference.

    Supports both file-based references (``./src/graph.py:graph``) and
    module-path references (``my_package.graph:graph``).

    Args:
        graph_ref: A string in the form ``<path_or_module>:<attribute>``.
        base_dir: Directory used to resolve relative file paths.

    Returns:
        The graph object (typically a compiled ``StateGraph``).

    Raises:
        ValueError: If ``graph_ref`` is not in the expected format.
        ImportError: If the module or file cannot be loaded.
        AttributeError: If the attribute does not exist in the loaded module.
    """
    if ":" not in graph_ref:
        raise ValueError(
            f"Invalid graph reference '{graph_ref}'. "
            "Expected '<file_or_module>:<attribute>', e.g. './src/graph.py:graph'."
        )

    module_part, attr = graph_ref.rsplit(":", 1)

    # File-based reference (starts with ./ or ../ or ends with .py)
    looks_like_file = (
        module_part.startswith("./")
        or module_part.startswith("../")
        or module_part.endswith(".py")
    )

    if looks_like_file:
        file_path = (base_dir / module_part).resolve()
        if not file_path.exists():
            raise ImportError(
                f"Graph file not found: '{file_path}' (resolved from '{module_part}')."
            )
        # Ensure the file's parent directory is on sys.path so relative imports work
        parent = str(file_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create a module spec for '{file_path}'.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    else:
        # Module-path reference (e.g. my_package.submod:graph)
        module = importlib.import_module(module_part)

    return getattr(module, attr)


def resolve_graph(
    config: dict[str, Any],
    base_dir: Path,
    graph_name: Optional[str],
) -> Any:
    """Resolve the graph to use from a parsed langgraph.json config.

    Args:
        config: Parsed langgraph.json dict (must have a ``graphs`` key).
        base_dir: Directory used to resolve relative file paths.
        graph_name: Name of the graph entry to use.  If ``None`` and there
            is exactly one entry, that entry is used automatically.

    Returns:
        The imported graph object.

    Raises:
        ValueError: If ``graph_name`` is required but not provided, or is not
            found in the config.
    """
    graphs: dict[str, str] = config["graphs"]

    if graph_name is None:
        if len(graphs) == 1:
            graph_name = next(iter(graphs))
        else:
            raise ValueError(
                f"langgraph.json defines multiple graphs ({list(graphs)}). "
                "Pass graph_name= to select one."
            )

    if graph_name not in graphs:
        raise ValueError(
            f"Graph '{graph_name}' not found in langgraph.json. "
            f"Available graphs: {list(graphs)}."
        )

    return import_graph(graphs[graph_name], base_dir)
