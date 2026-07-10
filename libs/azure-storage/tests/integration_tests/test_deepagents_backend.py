"""Blob-specific integration tests for AzureBlobBackend.

These cover behavior that is specific to the Azure Blob implementation
(concurrency guarantees, synthesized directories, the raw-content middleware
contract, native sync methods). The shared ``BackendProtocol`` surface is
covered by ``test_deepagents_backend_contract.py``.

They run against a live storage account or an emulator such as Azurite; see
``conftest.py`` for the environment variables that select the target.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

# Every test constructs a backend via the conftest fixture, so silence the
# beta warning at the module level.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

if TYPE_CHECKING:
    from langchain_azure_storage.deepagents import AzureBlobBackend


class TestConcurrency:
    async def test_concurrent_write_allows_only_one_success(
        self, backend: AzureBlobBackend
    ) -> None:
        # write() uses a conditional (If-None-Match: *) upload, so exactly one
        # of two racing writers can create the blob.
        results = await asyncio.gather(
            backend.awrite("/race.txt", "first"),
            backend.awrite("/race.txt", "second"),
        )

        succeeded = [result for result in results if result.error is None]
        failed = [result for result in results if result.error is not None]

        assert len(succeeded) == 1
        assert len(failed) == 1
        failure = failed[0].error
        assert failure is not None
        assert "already exists" in failure

    async def test_concurrent_edits_never_lose_an_update(
        self, backend: AzureBlobBackend
    ) -> None:
        # Edits are guarded by the blob's ETag: a concurrent writer either
        # serializes cleanly or the loser gets a retryable error, never a
        # silent lost update.
        await backend.awrite("/race-edit.txt", "alpha beta")
        results = await asyncio.gather(
            backend.aedit("/race-edit.txt", "alpha", "one"),
            backend.aedit("/race-edit.txt", "beta", "two"),
        )

        succeeded = [result for result in results if result.error is None]
        failed = [result for result in results if result.error is not None]
        assert len(succeeded) >= 1
        for failure in failed:
            assert failure.error is not None
            assert "modified concurrently" in failure.error

        final = await backend.aread("/race-edit.txt")
        assert final.file_data is not None
        if len(succeeded) == 2:
            assert final.file_data["content"] == "one two"


class TestMiddlewareContract:
    async def test_read_returns_raw_unformatted_content(
        self, backend: AzureBlobBackend
    ) -> None:
        # The backend returns raw content; line numbering, the empty-file
        # reminder, and base64 handling are applied by the Deep Agents
        # middleware at the tool boundary.
        await backend.awrite("/test.txt", "line one\nline two\nline three")
        result = await backend.aread("/test.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "line one\nline two\nline three"
        assert result.file_data["encoding"] == "utf-8"
        assert "1\t" not in result.file_data["content"]

    async def test_read_empty_file_returns_empty_content(
        self, backend: AzureBlobBackend
    ) -> None:
        await backend.awrite("/empty.txt", "")
        result = await backend.aread("/empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""


class TestSynthesizedDirectories:
    async def test_ls_synthesizes_directories(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/project/src/main.py", "code")
        await backend.awrite("/project/README.md", "readme")
        result = await backend.als("/project")
        assert result.entries is not None
        assert result.error is None
        paths = [i["path"] for i in result.entries]
        assert "/project/README.md" in paths
        dir_paths = [i["path"] for i in result.entries if i.get("is_dir")]
        assert "/project/src/" in dir_paths

    async def test_ls_nonexistent_returns_empty(
        self, backend: AzureBlobBackend
    ) -> None:
        result = await backend.als("/nonexistent")
        assert result.entries is not None
        assert result.error is None
        assert result.entries == []


class TestGrepPathGlob:
    async def test_grep_with_recursive_path_glob_filter(
        self, backend: AzureBlobBackend
    ) -> None:
        # A glob containing a slash is matched against the path relative to
        # the search root, not the file name.
        await backend.awrite("/src/top.py", "import os")
        await backend.awrite("/src/nested/deep.py", "import sys")
        result = await backend.agrep("import", "/", glob="src/*/*.py")
        assert result.matches is not None
        assert result.error is None
        paths = [m["path"] for m in result.matches]
        assert "/src/nested/deep.py" in paths
        assert "/src/top.py" not in paths


class TestSyncMethods:
    """End-to-end coverage for the native synchronous methods."""

    def test_sync_methods_round_trip(self, backend: AzureBlobBackend) -> None:
        write_result = backend.write("/sync/hello.txt", "hello world TODO")
        assert write_result.error is None

        read_content = backend.read("/sync/hello.txt")
        assert read_content.error is None
        assert read_content.file_data is not None
        assert "hello world TODO" in read_content.file_data["content"]

        backend.write("/sync/two.txt", "data")
        edit_result = backend.edit("/sync/hello.txt", "TODO", "DONE")
        assert edit_result.error is None

        ls_result = backend.ls("/sync")
        assert ls_result.entries is not None
        assert any(fi.get("path") == "/sync/hello.txt" for fi in ls_result.entries)

        glob_result = backend.glob("**/*.txt", "/sync")
        assert glob_result.matches is not None
        assert {fi.get("path") for fi in glob_result.matches} >= {
            "/sync/hello.txt",
            "/sync/two.txt",
        }

        grep_result = backend.grep("DONE", "/sync")
        assert grep_result.matches is not None
        assert any(m.get("path") == "/sync/hello.txt" for m in grep_result.matches)

        upload = backend.upload_files([("/sync/three.bin", b"payload")])
        assert upload[0].error is None

        download = backend.download_files(["/sync/three.bin"])
        assert download[0].error is None
        assert download[0].content == b"payload"

    def test_concurrent_sync_reads(self, backend: AzureBlobBackend) -> None:
        # Concurrent sync calls from multiple worker threads must not interfere.
        backend.write("/concurrent/a.txt", "alpha")
        backend.write("/concurrent/b.txt", "bravo")
        backend.write("/concurrent/c.txt", "charlie")

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(
                executor.map(
                    backend.read,
                    ["/concurrent/a.txt", "/concurrent/b.txt", "/concurrent/c.txt"],
                )
            )
        for content, expected in zip(
            results, ("alpha", "bravo", "charlie"), strict=True
        ):
            assert content.error is None
            assert content.file_data is not None
            assert expected in content.file_data["content"]
