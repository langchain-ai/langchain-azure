# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Manual verification script for the Azure AI Foundry Toolbox MCP endpoint.

Reads ``FOUNDRY_PROJECT_ENDPOINT`` and ``TOOLBOX_NAME`` from the
``samples/hosting/langgraph-hosted-agents/.env`` file (or your shell),
opens a single MCP session against the toolbox, lists every tool the
toolbox exposes, and then calls one tool to prove the gateway is live
end-to-end.

This is a low-level diagnostic — it talks directly to the MCP gateway
via ``mcp.client.streamable_http`` (same transport used by
``langchain-mcp-adapters``) and bypasses the LangChain wrappers used by
the rest of the samples. Use it when ``responses/04_foundry_toolbox``
fails and you want to confirm whether the problem is in the toolbox
configuration or the agent wiring above it.

Prereqs:

* a working ``az login`` (or any other ``DefaultAzureCredential`` source),
* ``FOUNDRY_PROJECT_ENDPOINT`` and ``TOOLBOX_NAME`` set in the shell or
  in ``samples/hosting/langgraph-hosted-agents/.env``,
* ``pip install mcp azure-identity python-dotenv``.

Usage (from ``samples/hosting/langgraph-hosted-agents/``)::

    # List tools and call ``web_search`` (preferred) or the first tool
    # otherwise. When ``web_search`` is picked automatically a default
    # query is supplied so the call succeeds without arguments.
    python tests/verify_toolbox_mcp.py

    # Call a specific tool by name (no arguments by default).
    python tests/verify_toolbox_mcp.py --tool web_search

    # Call a specific tool with JSON arguments.
    python tests/verify_toolbox_mcp.py --tool code_interpreter \
        --args '{"code": "print(17 * 25)"}'

    # Just list tools — do not call anything.
    python tests/verify_toolbox_mcp.py --list-only

    # Override the API version (defaults to v1).
    python tests/verify_toolbox_mcp.py --api-version v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SAMPLES_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SAMPLES_DIR / ".env")

_AAD_SCOPE = "https://ai.azure.com/.default"
_FEATURES_HEADER = "Foundry-Features"
_DEFAULT_FEATURES = "Toolboxes=V1Preview"


def _build_toolbox_url(project_endpoint: str, toolbox_name: str, api_version: str) -> str:
    """Construct the toolbox MCP endpoint URL.

    Matches the format used by
    ``langchain_azure_ai.tools.AzureAIProjectToolbox``::

        {project_endpoint}/toolboxes/{toolbox_name}/mcp?api-version={api_version}
    """
    base = project_endpoint.rstrip("/")
    return f"{base}/toolboxes/{toolbox_name}/mcp?api-version={api_version}"


def _truncate(text: str, limit: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dump_tool_result(result: Any, *, limit: int = 1500) -> None:
    """Print a tool-call result, preferring its text content blocks."""
    text_blocks: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            text_blocks.append(text)
    if text_blocks:
        joined = "\n".join(text_blocks)
        snippet = joined if len(joined) <= limit else joined[:limit] + f"\n... [truncated, {len(joined)} chars total]"
        print("  text content:")
        for line in snippet.splitlines():
            print(f"    {line}")
        return

    try:
        payload = result.model_dump()
    except AttributeError:
        payload = {"repr": repr(result)}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if len(rendered) > limit:
        rendered = rendered[:limit] + f"\n... [truncated, {len(rendered)} chars total]"
    print("  raw result:")
    for line in rendered.splitlines():
        print(f"    {line}")


async def verify_toolbox(
    *,
    tool_name: Optional[str],
    arguments: dict[str, Any],
    list_only: bool,
    api_version: str,
) -> int:
    project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    toolbox_name = os.environ.get("TOOLBOX_NAME")
    if not project_endpoint:
        print("ERROR: FOUNDRY_PROJECT_ENDPOINT is not set.", file=sys.stderr)
        return 2
    if not toolbox_name:
        print("ERROR: TOOLBOX_NAME is not set.", file=sys.stderr)
        return 2

    url = _build_toolbox_url(project_endpoint, toolbox_name, api_version)
    token = DefaultAzureCredential().get_token(_AAD_SCOPE).token
    headers = {
        "Authorization": f"Bearer {token}",
        _FEATURES_HEADER: _DEFAULT_FEATURES,
    }

    print(f"Toolbox MCP URL : {url}")
    print(f"Toolbox name    : {toolbox_name}")
    print()

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = list(tools_result.tools)
            print(f"Tools found: {len(tools)}")
            for tool in tools:
                desc = _truncate(tool.description or "")
                print(f"  - {tool.name}: {desc}")

            if list_only:
                return 0
            if not tools:
                print("\nNo tools to call.")
                return 0

            target_name = tool_name or tools[0].name
            available = {t.name for t in tools}
            if target_name not in available:
                print(
                    f"\nERROR: tool '{target_name}' is not exposed by toolbox '{toolbox_name}'.",
                    file=sys.stderr,
                )
                return 3

            print()
            print(f"Calling tool: {target_name}")
            print(f"  arguments: {json.dumps(arguments, ensure_ascii=False)}")
            try:
                result = await session.call_tool(target_name, arguments=arguments)
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                print(f"  ERROR: tool call raised {exc.__class__.__name__}: {exc}", file=sys.stderr)
                return 4

            print(f"  isError: {getattr(result, 'isError', False)}")
            _dump_tool_result(result)
            return 0 if not getattr(result, "isError", False) else 5


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List tools from the Foundry toolbox configured in .env "
            "and call one tool to prove the MCP gateway works."
        ),
    )
    parser.add_argument(
        "--tool",
        dest="tool",
        default=None,
        help="Tool name to call. Defaults to the first tool returned by tools/list.",
    )
    parser.add_argument(
        "--args",
        dest="arguments",
        default="{}",
        help='JSON object passed as the tool arguments (default: "{}").',
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list tools; do not call anything.",
    )
    parser.add_argument(
        "--api-version",
        default="v1",
        help='Toolbox API version (default: "v1").',
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --args must be valid JSON ({exc}).", file=sys.stderr)
        return 2
    if not isinstance(arguments, dict):
        print("ERROR: --args must decode to a JSON object.", file=sys.stderr)
        return 2

    return asyncio.run(
        verify_toolbox(
            tool_name=args.tool,
            arguments=arguments,
            list_only=args.list_only,
            api_version=args.api_version,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
