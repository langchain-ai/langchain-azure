# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run a graph from a LangGraph configuration file."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Sequence

from azure.ai.agentserver.responses import ResponsesServerOptions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Host a graph from a LangGraph configuration file with "
            "a Microsoft Foundry agent server."
        )
    )
    parser.add_argument(
        "agent",
        nargs="?",
        help="graph name; required when the configuration defines multiple graphs",
    )
    parser.add_argument(
        "--protocol",
        required=True,
        choices=("responses", "invocations"),
        help="agent server protocol to expose",
    )
    parser.add_argument(
        "--config",
        default="langgraph.json",
        help="path to the LangGraph configuration file (default: langgraph.json)",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "ResponsesServerOptions value; repeat for multiple options "
            "(for example, --option resilient_background=true)"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind")
    parser.add_argument(
        "--port", type=int, help="port to bind; defaults to PORT or 8088"
    )
    return parser


def _read_graphs(config_path: Path) -> dict[str, str]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"{config_path.name} was not found in {config_path.parent}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {config_path.name}: {exc}") from exc

    graphs = config.get("graphs") if isinstance(config, dict) else None
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError(f"{config_path.name} must define a non-empty 'graphs' object")

    targets: dict[str, str] = {}
    for name, graph_definition in graphs.items():
        if not isinstance(name, str):
            raise ValueError(f"{config_path.name} graph names must be strings")
        if isinstance(graph_definition, str):
            targets[name] = graph_definition
        elif isinstance(graph_definition, dict) and isinstance(
            graph_definition.get("path"), str
        ):
            targets[name] = graph_definition["path"]
        else:
            raise ValueError(
                f"{config_path.name} graph {name!r} must be a target string "
                "or an object with a string 'path'"
            )
    return targets


def _select_graph(graphs: dict[str, str], agent: str | None) -> tuple[str, str]:
    if agent is None:
        if len(graphs) == 1:
            return next(iter(graphs.items()))
        names = ", ".join(sorted(graphs))
        raise ValueError(f"multiple graphs are configured; specify one of: {names}")
    try:
        return agent, graphs[agent]
    except KeyError as exc:
        names = ", ".join(sorted(graphs))
        raise ValueError(f"unknown graph {agent!r}; choose one of: {names}") from exc


def _module_name_for_path(module_path: Path) -> tuple[str, Path]:
    parts = [module_path.stem]
    package_root = module_path.parent
    while (package_root / "__init__.py").is_file():
        parts.append(package_root.name)
        package_root = package_root.parent
    if len(parts) == 1:
        parts[0] = f"_langchain_azure_hosting_{module_path.stem}"
    return ".".join(reversed(parts)), package_root


def _load_file_module(module_path: Path) -> ModuleType:
    if not module_path.is_file():
        raise ValueError(f"graph module file does not exist: {module_path}")
    module_name, import_root = _module_name_for_path(module_path)
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load graph module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


async def _await_graph(value: Awaitable[Any]) -> Any:
    return await value


def _resolve_graph(value: Any, target: str) -> Any:
    if callable(getattr(value, "astream", None)):
        return value
    if callable(value):
        try:
            inspect.signature(value).bind()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"graph factory {target!r} must accept no arguments"
            ) from exc
        value = value()
    if inspect.isawaitable(value):
        value = asyncio.run(_await_graph(value))
    return value


def _load_graph(target: str, directory: Path) -> Any:
    try:
        module_target, symbol_target = target.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"invalid graph target {target!r}; expected 'module-or-file:symbol'"
        ) from exc
    if not module_target or not symbol_target:
        raise ValueError(
            f"invalid graph target {target!r}; expected 'module-or-file:symbol'"
        )

    if module_target.endswith(".py") or "/" in module_target or "\\" in module_target:
        module = _load_file_module((directory / module_target).resolve())
    else:
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        module = importlib.import_module(module_target)

    value: Any = module
    try:
        for part in symbol_target.split("."):
            value = getattr(value, part)
    except AttributeError as exc:
        raise ValueError(
            f"graph target {target!r} has no symbol {symbol_target!r}"
        ) from exc
    return _resolve_graph(value, target)


def _parse_option(raw_option: str) -> tuple[str, Any]:
    try:
        name, raw_value = raw_option.split("=", 1)
    except ValueError as exc:
        raise ValueError(f"invalid option {raw_option!r}; expected NAME=VALUE") from exc
    name = name.strip()
    raw_value = raw_value.strip()
    if not name or not raw_value:
        raise ValueError(f"invalid option {raw_option!r}; expected NAME=VALUE")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return name, value


def _server_options(raw_options: Sequence[str]) -> ResponsesServerOptions:
    kwargs = dict(_parse_option(raw_option) for raw_option in raw_options)
    try:
        return ResponsesServerOptions(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid server options: {exc}") from exc


def _run_server(
    graph: Any,
    *,
    protocol: str,
    options: ResponsesServerOptions | None,
    host: str,
    port: int | None,
) -> None:
    if protocol == "responses":
        from . import ResponsesHostServer

        ResponsesHostServer(graph, options=options).run(host=host, port=port)
        return
    if protocol == "invocations":
        from . import InvocationsHostServer

        InvocationsHostServer(graph).run(host=host, port=port)
        return
    raise ValueError(f"unsupported protocol: {protocol}")


def main(argv: Sequence[str] | None = None) -> None:
    """Load the selected graph and run it with the requested protocol."""
    parser = _parser()
    args = parser.parse_args(argv)
    directory = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = directory / config_path
    config_path = config_path.resolve()
    try:
        graphs = _read_graphs(config_path)
        _, target = _select_graph(graphs, args.agent)
        graph = _load_graph(target, config_path.parent)
        if args.protocol == "invocations" and args.option:
            raise ValueError("--option is only supported by the responses protocol")
        options = _server_options(args.option) if args.protocol == "responses" else None
        if args.port is not None:
            port = args.port
        else:
            port = int(os.environ.get("PORT", "8088"))
        _run_server(
            graph,
            protocol=args.protocol,
            options=options,
            host=args.host,
            port=port,
        )
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
