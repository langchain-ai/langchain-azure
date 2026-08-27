# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "deepagents>=0.7.1,<0.8",
#     "fastapi>=0.141,<1",
#     "langchain-azure-ai",
#     "langchain-azure-storage[deepagents]",
#     "langchain[anthropic,openai]",
#     "uvicorn[standard]>=0.44,<1",
# ]
#
# [tool.uv.sources]
# langchain-azure-storage = { path = "../../../libs/azure-storage", editable = true }
# ///
"""Process a mortgage packet with Deep Agents and Azure Blob Storage."""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

# Allows import of parent's _shared.py module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import build_blob_backend, build_model
from bootstrap import (
    MortgageSettings,
    bootstrap_mortgage_demo,
)
from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain_azure_storage.deepagents import AzureBlobBackend
from langchain_core.language_models import BaseChatModel

FINAL_DECISION = "04-underwriting-decision.md"
EXPECTED_OUTPUTS = (
    "01-packet-index.json",
    "02-classification.json",
    "03-extracted-facts.json",
    FINAL_DECISION,
)
DEFAULT_REQUEST = (
    "Process the mortgage packet in /source/ and write all required stage artifacts "
    "to /output/."
)

MortgageStreamObserver = Callable[[tuple[str, ...], str, Any], Awaitable[None]]


@dataclass(frozen=True)
class MortgageArtifact:
    """A verified artifact persisted by the mortgage agent."""

    name: str
    virtual_path: str
    blob_name: str
    run_id: str
    content: str


@dataclass(frozen=True)
class MortgageProcessingResult:
    """Verified output from one mortgage packet processing run."""

    run_id: str
    response: str
    artifacts: tuple[MortgageArtifact, ...]
    decision: MortgageArtifact


def create_mortgage_agent(
    model: str | BaseChatModel, backend: CompositeBackend
) -> Any:
    """Create one agent with four declarative mortgage specialists."""
    return create_deep_agent(
        model=model,
        backend=backend,
        subagents=build_mortgage_subagents(),
        system_prompt=(
            "Process the mortgage packet using the specialist team and the conventions "
            "loaded from AGENTS.md."
        ),
        memory=["/guidance/AGENTS.md"],
        permissions=[
            FilesystemPermission(
                operations=["write"],
                paths=["/source/**", "/guidance/**"],
                mode="deny",
            )
        ],
    )


def build_mortgage_model(settings: MortgageSettings) -> str | BaseChatModel:
    """Build the model once for reuse across mortgage processing runs."""
    return build_model(
        model_name=settings.require_model_name(),
        model_options={
            "streaming": True,
            "max_retries": 2,
            "timeout": settings.timeout_seconds,
        },
    )


def build_mortgage_subagents() -> list[SubAgent]:
    """Build the specialist subagents used by the mortgage agent."""
    skills = ["/guidance/skills/"]
    return [
        SubAgent(
            name="intake-split-agent",
            description="Checks packet completeness and creates the packet index.",
            system_prompt=(
                "Use the packet-intake skill and write only its required artifact."
            ),
            skills=skills,
        ),
        SubAgent(
            name="classification-agent",
            description="Classifies every document in the mortgage packet.",
            system_prompt=(
                "Use the document-classification skill and write only its required "
                "artifact."
            ),
            skills=skills,
        ),
        SubAgent(
            name="extraction-agent",
            description="Extracts supported financial and property facts.",
            system_prompt=(
                "Use the mortgage-fact-extraction skill and write only its required "
                "artifact."
            ),
            skills=skills,
        ),
        SubAgent(
            name="underwriting-agent",
            description="Applies packet policy and produces the final decision.",
            system_prompt=(
                "Use the mortgage-underwriting skill and write only its required "
                "artifact."
            ),
            skills=skills,
        ),
    ]


def create_blob_input_backends(
    settings: MortgageSettings,
) -> tuple[AzureBlobBackend, AzureBlobBackend]:
    """Create the shared source and guidance Blob backends."""
    source_backend = build_blob_backend(
        prefix=settings.source_prefix,
        container_name=settings.source_container,
        account_url=settings.account_url,
        connection_string=settings.connection_string,
    )
    guidance_backend = build_blob_backend(
        container_name=settings.guidance_container,
        account_url=settings.account_url,
        connection_string=settings.connection_string,
    )
    return source_backend, guidance_backend


def create_blob_output_backend(
    settings: MortgageSettings,
    *,
    run_id: str | None = None,
) -> AzureBlobBackend:
    """Create an output backend at the base prefix or scoped to one run."""
    prefix = settings.output_prefix
    if run_id:
        prefix += f"{run_id}/"
    return build_blob_backend(
        prefix=prefix,
        container_name=settings.output_container,
        account_url=settings.account_url,
        connection_string=settings.connection_string,
    )


def create_mortgage_backend(
    source_backend: AzureBlobBackend,
    guidance_backend: AzureBlobBackend,
    output_backend: AzureBlobBackend,
) -> CompositeBackend:
    """Route mortgage files across the three Blob backends."""
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/source/": source_backend,
            "/guidance/": guidance_backend,
            "/output/": output_backend,
        },
    )


async def process_mortgage_packet(
    prompt: str,
    *,
    settings: MortgageSettings,
    model: str | BaseChatModel,
    source_backend: AzureBlobBackend,
    guidance_backend: AzureBlobBackend,
    observe_stream: MortgageStreamObserver | None = None,
) -> MortgageProcessingResult:
    """Process one packet and verify every expected artifact in Blob Storage."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Mortgage processing request cannot be empty")

    run_id = _create_mortgage_run_id()
    output_backend = create_blob_output_backend(
        settings,
        run_id=run_id,
    )
    backend = create_mortgage_backend(
        source_backend,
        guidance_backend,
        output_backend,
    )

    async with output_backend:
        agent = create_mortgage_agent(model, backend)
        try:
            async with asyncio.timeout(settings.timeout_seconds):
                graph_input = {"messages": [{"role": "user", "content": prompt}]}
                if observe_stream is None:
                    state = await agent.ainvoke(graph_input)
                else:
                    state = None
                    async for namespace, stream_mode, data in agent.astream(
                        graph_input,
                        stream_mode=["messages", "updates", "values"],
                        subgraphs=True,
                    ):
                        await observe_stream(namespace, stream_mode, data)
                        if stream_mode == "values" and not namespace:
                            state = data
                    if state is None:
                        raise RuntimeError("Mortgage agent returned no final state")
        except TimeoutError as exc:
            raise RuntimeError(
                f"Mortgage processing exceeded {settings.timeout_seconds} seconds"
            ) from exc

        messages = state.get("messages", [])
        response = _message_text(messages[-1].content) if messages else ""
        artifacts = await _verify_mortgage_artifacts(
            settings=settings,
            output_backend=output_backend,
            run_id=run_id,
        )

    decision = next(
        artifact for artifact in artifacts if artifact.name == FINAL_DECISION
    )
    return MortgageProcessingResult(
        run_id=run_id,
        response=response or "Mortgage packet processing completed.",
        artifacts=artifacts,
        decision=decision,
    )


def _create_mortgage_run_id() -> str:
    """Create the timestamped identifier used to isolate one run's outputs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{secrets.token_hex(2)}"


async def _verify_mortgage_artifacts(
    *,
    settings: MortgageSettings,
    output_backend: AzureBlobBackend,
    run_id: str,
) -> tuple[MortgageArtifact, ...]:
    """Read back every required output and return verified artifacts."""
    downloads = await output_backend.adownload_files(
        [f"/{name}" for name in EXPECTED_OUTPUTS]
    )
    output_prefix = f"{settings.output_prefix}{run_id}/"
    artifacts: list[MortgageArtifact] = []
    for name, download in zip(EXPECTED_OUTPUTS, downloads, strict=True):
        if download.error is not None or download.content is None:
            raise RuntimeError(f"Expected output {name} was not written")
        blob_name = f"{output_prefix}{name}"
        artifacts.append(
            MortgageArtifact(
                name=name,
                virtual_path=f"/output/{name}",
                blob_name=blob_name,
                run_id=run_id,
                content=download.content.decode("utf-8"),
            )
        )
    return tuple(artifacts)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_message_text(item) for item in content)))
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        nested = content.get("content")
        if isinstance(nested, (str, list, dict)):
            return _message_text(nested)
    return ""


async def main(*, serve: bool = False) -> None:
    """Bootstrap shared resources, then run headless or serve the browser UI."""
    settings = MortgageSettings.from_env()
    source_backend, guidance_backend = create_blob_input_backends(settings)
    async with source_backend, guidance_backend:
        await bootstrap_mortgage_demo(
            settings, source_backend, guidance_backend
        )
        print("[startup] Building chat model...", flush=True)
        model = build_mortgage_model(settings)
        if serve:
            await _serve(
                settings=settings,
                model=model,
                source_backend=source_backend,
                guidance_backend=guidance_backend,
            )
        else:
            result = await process_mortgage_packet(
                DEFAULT_REQUEST,
                settings=settings,
                model=model,
                source_backend=source_backend,
                guidance_backend=guidance_backend,
            )
            print(result.decision.content)


async def _serve(
    *,
    settings: MortgageSettings,
    model: str | BaseChatModel,
    source_backend: AzureBlobBackend,
    guidance_backend: AzureBlobBackend,
) -> None:
    process_prompt = partial(
        process_mortgage_packet,
        settings=settings,
        model=model,
        source_backend=source_backend,
        guidance_backend=guidance_backend,
    )

    from server.main import create_app, run_server

    async with create_blob_output_backend(settings) as output_backend:
        app = create_app(
            packet_id=settings.packet_id,
            model_name=settings.require_model_name(),
            account_url=settings.account_url,
            source_container=settings.source_container,
            source_prefix=settings.source_prefix,
            guidance_container=settings.guidance_container,
            output_container=settings.output_container,
            output_prefix=settings.output_prefix,
            source_backend=source_backend,
            output_backend=output_backend,
            process_prompt=process_prompt,
        )
        await run_server(app)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a mortgage packet with Deep Agents and Azure Blob Storage."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start the browser application instead of processing one packet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(serve=args.serve))