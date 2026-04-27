# Copyright (c) Microsoft. All rights reserved.

"""Utilities for loading graph configurations from ``langgraph.json`` files."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_graph_from_langgraph_config(
    path: str | os.PathLike[str] = "langgraph.json",
    *,
    graph_name: str | None = None,
) -> Any:
    """Load a compiled LangGraph graph from a ``langgraph.json`` config file.

    Args:
        path: Path to the ``langgraph.json`` file. Defaults to
            ``"langgraph.json"`` in the current working directory.
        graph_name: Key name of the graph to load from the ``"graphs"``
            section. Required when more than one graph is defined; omit
            when the file contains exactly one graph.

    Returns:
        The compiled LangGraph graph object referenced by the config.

    Raises:
        FileNotFoundError: If the config file or the graph module file is
            not found.
        KeyError: If *graph_name* is not present in the ``"graphs"`` section.
        ValueError: If *graph_name* is omitted when multiple graphs are
            defined, or if a graph spec string has an invalid format.
        AttributeError: If the graph attribute does not exist in the loaded
            module.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"langgraph config file not found: {config_path}")

    with config_path.open() as f:
        config = json.load(f)

    graphs: dict[str, str] = config.get("graphs", {})
    if not graphs:
        raise ValueError(f"No graphs defined in {config_path}")

    if graph_name is None:
        if len(graphs) == 1:
            graph_name = next(iter(graphs))
        else:
            raise ValueError(
                f"Multiple graphs are defined in {config_path}. "
                f"Pass graph_name= with one of: {sorted(graphs)}"
            )
    elif graph_name not in graphs:
        raise KeyError(
            f"Graph {graph_name!r} not found in {config_path}. "
            f"Available graphs: {sorted(graphs)}"
        )

    spec = graphs[graph_name]
    return _load_graph_from_spec(spec, config_path.parent)


def _load_graph_from_spec(spec: str, config_dir: Path) -> Any:
    """Load a graph object from a ``'path/to/file.py:attribute'`` spec string.

    Args:
        spec: Graph location in ``'<filepath>:<attribute>'`` format, as used
            in ``langgraph.json``.
        config_dir: Directory containing the ``langgraph.json`` file; used
            to resolve relative file paths inside *spec*.

    Returns:
        The graph object referenced by *spec*.
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid graph spec {spec!r}. "
            "Expected format: 'path/to/file.py:attribute_name'."
        )

    file_path_str, attr_name = spec.rsplit(":", 1)
    file_path = (config_dir / file_path_str).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Graph module file not found: {file_path}")

    # Ensure the module's parent directory is on sys.path so that relative
    # imports inside the graph module resolve correctly.
    module_dir = str(file_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = f"_langchain_azure_ai_graph_{file_path.stem}"
    spec_obj = importlib.util.spec_from_file_location(module_name, file_path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot create module spec from {file_path}")

    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, attr_name):
        public_names = [n for n in dir(module) if not n.startswith("_")]
        raise AttributeError(
            f"Module {file_path} has no attribute {attr_name!r}. "
            f"Available names: {public_names}"
        )

    return getattr(module, attr_name)
