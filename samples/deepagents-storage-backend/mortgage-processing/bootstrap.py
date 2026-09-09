"""Configure and prepare Azure Blob Storage for the mortgage sample."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from _shared import ensure_container
from deepagents.backends.protocol import FILE_NOT_FOUND
from langchain_azure_storage.deepagents import AzureBlobBackend

SAMPLE_ROOT = Path(__file__).parent
DEFAULT_TIMEOUT_SECONDS = 180


def _normalize_blob_prefix(value: str) -> str:
    parts = [part for part in value.strip("/").split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise RuntimeError("Blob prefixes cannot contain '.' or '..' segments")
    return "/".join(parts) + ("/" if parts else "")


@dataclass(frozen=True)
class MortgageSettings:
    """Environment-backed settings for the mortgage processing sample."""

    account_url: str | None
    connection_string: str | None
    model_name: str | None
    packet_id: str
    source_container: str
    source_prefix: str
    guidance_container: str
    output_container: str
    output_prefix: str
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> MortgageSettings:
        """Load settings while preserving the sample's environment overrides."""
        timeout_seconds = int(
            os.environ.get("MORTGAGE_DEMO_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        if not 20 <= timeout_seconds <= 300:
            raise RuntimeError(
                "MORTGAGE_DEMO_TIMEOUT_SECONDS must be between 20 and 300"
            )
        packet_id = os.environ.get("MORTGAGE_PACKET_ID", "MORT-2026-0042")
        if not packet_id or Path(packet_id).name != packet_id or packet_id in {".", ".."}:
            raise RuntimeError(
                "MORTGAGE_PACKET_ID must be one non-empty directory name"
            )
        return cls(
            account_url=os.environ.get("AZURE_STORAGE_ACCOUNT_URL"),
            connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
            model_name=os.environ.get("MODEL_NAME"),
            packet_id=packet_id,
            source_container=os.environ.get(
                "AZURE_STORAGE_MORTGAGE_SOURCE_CONTAINER", "mortgage-packets"
            ),
            source_prefix=_normalize_blob_prefix(
                os.environ.get("AZURE_STORAGE_MORTGAGE_SOURCE_PREFIX") or packet_id
            ),
            guidance_container=os.environ.get(
                "AZURE_STORAGE_MORTGAGE_GUIDANCE_CONTAINER",
                "mortgage-agent-context",
            ),
            output_container=os.environ.get(
                "AZURE_STORAGE_MORTGAGE_OUTPUT_CONTAINER", "mortgage-decisions"
            ),
            output_prefix=_normalize_blob_prefix(
                os.environ.get("AZURE_STORAGE_MORTGAGE_OUTPUT_PREFIX") or packet_id
            ),
            timeout_seconds=timeout_seconds,
        )

    def require_model_name(self) -> str:
        """Return the configured model name or raise an actionable error."""
        if not self.model_name:
            raise RuntimeError("Set MODEL_NAME in .env before processing a packet")
        return self.model_name


def _packaged_files(root: Path, remote_root: str = "/") -> dict[str, bytes]:
    return {
        f"{remote_root.rstrip('/')}/{path.relative_to(root).as_posix()}": path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


async def _upload_missing_demo_files(
    backend: AzureBlobBackend,
    files: dict[str, bytes],
    *,
    container_name: str,
    label: str,
) -> None:
    print(f"[bootstrap] Checking {label} in {container_name}...", flush=True)
    responses = await backend.adownload_files(list(files))
    missing_paths: list[str] = []
    for response in responses:
        if response.error == FILE_NOT_FOUND:
            missing_paths.append(response.path)
        elif response.error is not None:
            raise RuntimeError(response.error)

    uploads = [(path, files[path]) for path in missing_paths]
    if not uploads:
        print(f"[bootstrap] {label.capitalize()} already present.", flush=True)
        return
    for response in await backend.aupload_files(uploads):
        if response.error is not None:
            raise RuntimeError(response.error)
    print(f"[bootstrap] Uploaded {len(uploads)} missing {label}.", flush=True)


async def bootstrap_mortgage_demo(
    settings: MortgageSettings,
    source_backend: AzureBlobBackend,
    guidance_backend: AzureBlobBackend,
) -> None:
    """Create missing containers and seed missing packet and guidance files."""
    print("[bootstrap] Checking Blob containers...", flush=True)
    for container_name in (
        settings.source_container,
        settings.guidance_container,
        settings.output_container,
    ):
        await asyncio.to_thread(
            ensure_container,
            container_name,
            account_url=settings.account_url,
            connection_string=settings.connection_string,
        )
    print("[bootstrap] Blob containers are ready.", flush=True)

    packet_directory = SAMPLE_ROOT / "data" / settings.packet_id
    if not packet_directory.is_dir():
        raise RuntimeError(
            f"No packaged mortgage data found for {settings.packet_id} at "
            f"{packet_directory}"
        )
    packet_files = _packaged_files(packet_directory)
    guidance_files = {
        "/AGENTS.md": (SAMPLE_ROOT / "AGENTS.md").read_bytes(),
        **_packaged_files(SAMPLE_ROOT / "skills", "/skills"),
    }
    await asyncio.gather(
        _upload_missing_demo_files(
            source_backend,
            packet_files,
            container_name=settings.source_container,
            label="packet files",
        ),
        _upload_missing_demo_files(
            guidance_backend,
            guidance_files,
            container_name=settings.guidance_container,
            label="guidance files",
        ),
    )
    print("[bootstrap] Mortgage sample storage is ready.", flush=True)