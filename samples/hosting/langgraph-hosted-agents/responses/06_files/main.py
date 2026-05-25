"""Sample 06 - Responses API + filesystem tools.

Demonstrates a LangGraph agent that **reads files at runtime** through
local Python tools. The agent has two tools:

- ``list_files(subpath="")`` - list the contents of a directory below
  the configured data root.
- ``read_text_file(file_path)`` - read a UTF-8 text file from the data
  root.

Today the langchain Responses host only forwards text content blocks
(see the hosting converter at
``libs/azure-ai/langchain_azure_ai/agents/hosting/_converters/_request.py``).
Until the hosting layer learns to passthrough Responses ``input_file``
items, files reach the agent via the filesystem inside the container
rather than as request attachments.

Required environment variables (set in ``.env`` or your shell):

    FOUNDRY_PROJECT_ENDPOINT        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    DATA_DIR                        optional, defaults to ./data alongside main.py
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python main.py

Then in another terminal::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"List the files available to you, then summarize notes.txt.","model":"gpt-4o"}'
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

load_dotenv()

_HERE = Path(__file__).resolve().parent
_MAX_READ_BYTES = 64 * 1024


def _data_root() -> Path:
    """Resolve the data directory shared with the agent.

    Resolved once per call to ``read_text_file`` / ``list_files`` so the
    sample can be repointed at a different directory via ``DATA_DIR``
    without restarting the host.
    """
    return Path(os.environ.get("DATA_DIR", _HERE / "data")).resolve()


def _resolve_inside_root(relative_path: str) -> Path:
    """Resolve ``relative_path`` under the data root, rejecting escapes."""
    root = _data_root()
    candidate = (root / relative_path).resolve() if relative_path else root
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Path {relative_path!r} escapes the data root {root}."
        ) from exc
    return candidate


@tool
def list_files(
    subpath: Annotated[str, "Subdirectory under the data root. Use '' for the root."] = "",
) -> str:
    """List files and directories under the data root."""
    target = _resolve_inside_root(subpath)
    if not target.exists():
        return f"Directory {subpath or '.'!r} does not exist."
    if not target.is_dir():
        return f"{subpath!r} is not a directory."
    entries = []
    for entry in sorted(target.iterdir()):
        kind = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else None
        rel = entry.relative_to(_data_root()).as_posix()
        entries.append(f"- {rel} ({kind}{', ' + str(size) + ' bytes' if size is not None else ''})")
    return "\n".join(entries) if entries else "(empty)"


@tool
def read_text_file(
    file_path: Annotated[str, "Path of the file to read, relative to the data root."],
) -> str:
    """Read a UTF-8 text file from the data root (capped at 64 KiB)."""
    target = _resolve_inside_root(file_path)
    if not target.exists() or not target.is_file():
        return f"File {file_path!r} not found."
    data = target.read_bytes()[:_MAX_READ_BYTES]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    project = AIProjectClient(endpoint=project_endpoint, credential=credential)
    openai_client = project.get_openai_client()
    token_provider = get_bearer_token_provider(credential, _AZURE_AI_SCOPE)

    return ChatOpenAI(
        model=deployment,
        base_url=str(openai_client.base_url),
        api_key=token_provider,
    )


def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    # Touch the data root once at startup so we fail loudly if it is missing.
    root = _data_root()
    root.mkdir(parents=True, exist_ok=True)

    graph = create_agent(
        _build_chat_model(),
        tools=[list_files, read_text_file],
    )
    port = int(os.environ.get("PORT", "8088"))
    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
