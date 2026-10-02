#!/usr/bin/env python3
"""
update_docs.py — Azure integration documentation update agent.

Reads the azure-integration-docs skill instructions and drives an AI agent
(via GitHub Models API) to generate or update LangChain documentation pages
for the specified Azure integration library.

The agent is equipped with tools that mirror how the skill is used interactively:
it can explore and read files in both repos, then write updated .mdx files
directly into the langchain-ai/docs working tree.

Required environment variables:
  GITHUB_TOKEN         GitHub token with models:read permission
  LANGCHAIN_AZURE_ROOT Absolute path to the langchain-azure repo root
  DOCS_ROOT            Absolute path to the langchain-ai/docs repo root
  UPDATE_LIBRARY       Library path relative to repo root (e.g. libs/azure-ai)

Optional environment variables:
  UPDATE_PATH          Specific module sub-path (e.g. langchain_azure_ai/chat_models)
  UPDATE_CHANGELOG     Changelog / release-notes text to use as context
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# ── Constants ──────────────────────────────────────────────────────────────────

MODELS_API_URL = "https://models.inference.ai.azure.com/chat/completions"
MODEL = "gpt-4o"
MAX_TURNS = 30           # Upper bound on agent conversation turns
MAX_FILE_BYTES = 80_000  # Truncate source files larger than ~80 KB


# ── File helpers ───────────────────────────────────────────────────────────────

def read_text(path: Path) -> str:
    """Read a text file, truncating very large files to stay within context limits."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"[Error reading file: {exc}]"
    if len(raw) > MAX_FILE_BYTES:
        truncated = raw[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
        return truncated + f"\n\n[… file truncated at {MAX_FILE_BYTES} bytes …]"
    return raw.decode("utf-8", errors="replace")


def safe_resolve(root: Path, rel: str) -> Path | None:
    """Resolve *rel* against *root*, rejecting any path-traversal attempts."""
    try:
        target = (root / rel).resolve()
    except Exception:
        return None
    if not str(target).startswith(str(root.resolve())):
        return None
    return target


def list_dir(path: Path) -> str:
    """Return a human-readable directory listing."""
    if not path.exists():
        return f"[Directory not found: {path}]"
    items: list[str] = []
    for item in sorted(path.iterdir()):
        if item.name.startswith("."):
            continue
        prefix = "[DIR] " if item.is_dir() else "      "
        items.append(f"{prefix}{item.name}")
    return "\n".join(items) if items else "(empty)"


# ── Tool schema ────────────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_langchain_azure_dir",
            "description": (
                "List the files and sub-directories inside a directory of the "
                "langchain-azure repository. Use this to explore package structure "
                "before deciding which files to read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the langchain-azure repo root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_langchain_azure_file",
            "description": "Read a file from the langchain-azure repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the langchain-azure repo root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_docs_dir",
            "description": (
                "List the files and sub-directories inside a directory of the "
                "langchain-ai/docs repository. Use this to find existing integration "
                "pages and TEMPLATE.mdx files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the langchain-ai/docs repo root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_docs_file",
            "description": "Read a file from the langchain-ai/docs repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the langchain-ai/docs repo root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_docs_file",
            "description": (
                "Write or update a documentation file in the langchain-ai/docs repository. "
                "Call this once per file with the COMPLETE, final content. "
                "Always read the relevant TEMPLATE.mdx and any existing page BEFORE calling this. "
                "Use this for the main integration page AND for any index or provider pages "
                "that the skill instructs you to update."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path relative to the docs repo root, "
                            "e.g. 'src/oss/python/integrations/chat/azure_ai.mdx'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────

def execute_tool(
    call: dict,
    azure_root: Path,
    docs_root: Path,
    files_written: list[str],
) -> str:
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError as exc:
        return f"[Error: invalid JSON in tool arguments — {exc}]"

    rel: str = args.get("path", "")

    if name == "list_langchain_azure_dir":
        target = safe_resolve(azure_root, rel)
        if target is None:
            return "[Error: path traversal not allowed]"
        return list_dir(target)

    if name == "read_langchain_azure_file":
        target = safe_resolve(azure_root, rel)
        if target is None:
            return "[Error: path traversal not allowed]"
        if not target.is_file():
            return f"[File not found: {rel}]"
        return read_text(target)

    if name == "list_docs_dir":
        target = safe_resolve(docs_root, rel)
        if target is None:
            return "[Error: path traversal not allowed]"
        return list_dir(target)

    if name == "read_docs_file":
        target = safe_resolve(docs_root, rel)
        if target is None:
            return "[Error: path traversal not allowed]"
        if not target.is_file():
            return f"[File not found: {rel}]"
        return read_text(target)

    if name == "write_docs_file":
        content: str = args.get("content", "")
        target = safe_resolve(docs_root, rel)
        if target is None:
            return "[Error: path traversal not allowed]"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        files_written.append(rel)
        print(f"  ✓ Written: {rel}", flush=True)
        return f"File written successfully: {rel}"

    return f"[Unknown tool: {name}]"


# ── API helpers ────────────────────────────────────────────────────────────────

def call_models_api(token: str, messages: list[dict]) -> dict:
    """Call the GitHub Models chat-completions endpoint with retry on 429."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.1,
    }

    for attempt in range(4):
        resp = requests.post(MODELS_API_URL, headers=headers, json=payload, timeout=180)
        if resp.status_code == 429:
            wait = 2 ** attempt * 15  # 15s, 30s, 60s, 120s
            print(f"  [Rate limited — retrying in {wait}s]", flush=True)
            time.sleep(wait)
            continue
        if not resp.ok:
            print(f"API error {resp.status_code}: {resp.text}", file=sys.stderr)
            sys.exit(1)
        return resp.json()

    print("ERROR: GitHub Models API rate limit exceeded after retries.", file=sys.stderr)
    sys.exit(1)


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_system_prompt(skill_content: str) -> str:
    return f"""You are an automated documentation agent for LangChain Azure integrations.

You have tools to read files from both the langchain-azure (source code) and langchain-ai/docs
(documentation) repositories, and to write documentation files back to the docs repo.

Follow the skill instructions below exactly. They govern what to read, how to structure pages,
and which discovery surfaces to update:

{skill_content}

Recommended turn-by-turn workflow:
1. Call list_docs_dir("src/oss/python/integrations") to see the available building-block folders.
2. Identify the correct building block from the library path, then read its TEMPLATE.mdx.
3. List and read relevant source files from langchain-azure: pyproject.toml, README.md, and
   the key Python modules (especially __init__.py and the main implementation files).
4. Read any existing Azure integration pages in the target docs folder.
5. Call write_docs_file for the main integration page, then for any index.mdx and
   providers/microsoft.mdx pages that need updating.
6. Finish with a brief summary of what was written and what was not checked.

Use tools to gather all required context before writing. Never invent class names, env vars,
or install commands — verify each one against the source code you have read.
"""


def build_user_message(library: str, specific_path: str, changelog: str) -> str:
    parts = [f"Update the LangChain documentation for the Azure integration library: **`{library}`**"]
    if specific_path:
        parts.append(f"\nFocus on the specific module path: `{specific_path}`")
    if changelog:
        parts.append(
            f"\nThe following changelog entry provides context for what changed in this release:\n"
            f"```\n{changelog}\n```"
        )
    parts.append(
        "\nFollow the skill workflow: read the relevant template first, "
        "then read the source code, then write the documentation."
    )
    return "\n".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Read inputs from environment (safe for multi-line values like changelog)
    library = os.environ.get("UPDATE_LIBRARY", "").strip()
    specific_path = os.environ.get("UPDATE_PATH", "").strip()
    changelog = os.environ.get("UPDATE_CHANGELOG", "").strip()

    if not library:
        print("ERROR: UPDATE_LIBRARY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    azure_root = Path(os.environ.get("LANGCHAIN_AZURE_ROOT", ".")).resolve()
    docs_root_str = os.environ.get("DOCS_ROOT", "")
    if not docs_root_str:
        print("ERROR: DOCS_ROOT environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    docs_root = Path(docs_root_str).resolve()

    if not docs_root.is_dir():
        print(f"ERROR: DOCS_ROOT is not a valid directory: {docs_root}", file=sys.stderr)
        sys.exit(1)

    # Load the project-level skill instructions
    skill_path = azure_root / ".github/skills/azure-integration-docs/SKILL.md"
    if not skill_path.is_file():
        print(f"ERROR: Skill file not found: {skill_path}", file=sys.stderr)
        sys.exit(1)
    skill_content = skill_path.read_text(encoding="utf-8")

    print(f"Library  : {library}")
    if specific_path:
        print(f"Path     : {specific_path}")
    if changelog:
        print(f"Changelog: {changelog[:80]}{'…' if len(changelog) > 80 else ''}")
    print(f"Model    : {MODEL}")
    print()

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(skill_content)},
        {"role": "user", "content": build_user_message(library, specific_path, changelog)},
    ]

    files_written: list[str] = []

    for turn in range(MAX_TURNS):
        print(f"[Turn {turn + 1}/{MAX_TURNS}]", flush=True)
        data = call_models_api(token, messages)
        choice = data["choices"][0]
        msg = choice["message"]
        messages.append(msg)

        finish_reason = choice["finish_reason"]

        if finish_reason == "tool_calls":
            tool_calls = msg.get("tool_calls", [])
            tool_results: list[dict] = []

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                preview = raw_args[:100].replace("\n", " ")
                print(f"  → {fn_name}({preview}{'…' if len(raw_args) > 100 else ''})", flush=True)

                result = execute_tool(tc, azure_root, docs_root, files_written)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            messages.extend(tool_results)

        elif finish_reason == "stop":
            content = msg.get("content", "")
            if content:
                print(f"\nAgent summary:\n{content}", flush=True)
            break

        else:
            print(f"Unexpected finish_reason: {finish_reason}", flush=True)
            break

    # Report outcome
    print(f"\n{'─' * 60}")
    if files_written:
        unique_files = sorted(set(files_written))
        print(f"✅ Documentation update complete — {len(unique_files)} file(s) written:")
        for f in unique_files:
            print(f"   • {f}")
    else:
        print("⚠️  No documentation files were written.", file=sys.stderr)
        print(
            "   The agent may have determined no changes were needed, "
            "or an error occurred during the run.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
