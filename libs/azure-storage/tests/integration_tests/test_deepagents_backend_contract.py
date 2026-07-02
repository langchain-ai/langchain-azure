"""BackendProtocol-style contract tests for AzureBlobBackend.

These tests mirror the shape of Deep Agents' backend tests without depending
on langchain-tests. They focus on the shared file backend contract and leave
Azure-specific behavior to the unit and integration suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_azure_storage.deepagents import AzureBlobBackend


class TestBackendProtocolContract:
    async def test_write_read_edit_lifecycle(self, backend: AzureBlobBackend) -> None:
        write = await backend.awrite("/contract/lifecycle.txt", "alpha\nbeta\ngamma")
        assert write.error is None
        assert write.path == "/contract/lifecycle.txt"

        read = await backend.aread("/contract/lifecycle.txt")
        assert read.error is None
        assert read.file_data is not None
        assert "alpha" in read.file_data["content"]
        assert "beta" in read.file_data["content"]

        edit = await backend.aedit("/contract/lifecycle.txt", "beta", "delta")
        assert edit.error is None
        assert edit.path == "/contract/lifecycle.txt"
        assert edit.occurrences == 1

        reread = await backend.aread("/contract/lifecycle.txt")
        assert reread.error is None
        assert reread.file_data is not None
        assert "delta" in reread.file_data["content"]
        assert "beta" not in reread.file_data["content"]

    async def test_write_existing_file_returns_error_and_preserves_content(
        self, backend: AzureBlobBackend
    ) -> None:
        first = await backend.awrite("/contract/existing.txt", "first")
        second = await backend.awrite("/contract/existing.txt", "second")

        assert first.error is None
        assert second.error is not None
        assert "already exists" in second.error.lower()

        read = await backend.aread("/contract/existing.txt")
        assert read.error is None
        assert read.file_data is not None
        assert "first" in read.file_data["content"]
        assert "second" not in read.file_data["content"]

    async def test_ls_returns_direct_children(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/contract/list/a.txt", "a")
        await backend.awrite("/contract/list/b.txt", "b")
        await backend.awrite("/contract/list/nested/c.txt", "c")

        result = await backend.als("/contract/list")
        assert result.error is None
        assert result.entries is not None
        paths = {info["path"] for info in result.entries}
        assert paths >= {
            "/contract/list/a.txt",
            "/contract/list/b.txt",
            "/contract/list/nested/",
        }

    async def test_glob_matches_files(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/contract/glob/root.py", "root")
        await backend.awrite("/contract/glob/root.txt", "root")
        await backend.awrite("/contract/glob/pkg/nested.py", "nested")

        result = await backend.aglob("**/*.py", "/contract/glob")
        assert result.error is None
        assert result.matches is not None
        assert {info["path"] for info in result.matches} == {
            "/contract/glob/root.py",
            "/contract/glob/pkg/nested.py",
        }

    async def test_grep_matches_literal_text(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/contract/grep/a.py", "needle\nother")
        await backend.awrite("/contract/grep/b.txt", "needle")
        await backend.awrite("/contract/grep/c.py", "haystack")

        result = await backend.agrep("needle", "/contract/grep", glob="*.py")
        assert result.error is None
        assert result.matches is not None
        assert [match["path"] for match in result.matches] == ["/contract/grep/a.py"]

    async def test_upload_download_batch_order_and_partial_errors(
        self, backend: AzureBlobBackend
    ) -> None:
        files = [
            ("/contract/batch/one.bin", b"one"),
            ("/contract/batch/two.bin", b"two"),
            ("/contract/batch/three.bin", bytes(range(16))),
        ]

        upload = await backend.aupload_files(files)
        assert [response.path for response in upload] == [path for path, _ in files]
        assert [response.error for response in upload] == [None, None, None]

        download = await backend.adownload_files(
            [
                "/contract/batch/one.bin",
                "/contract/batch/missing.bin",
                "/contract/batch/three.bin",
            ]
        )
        assert download[0].content == b"one"
        assert download[0].error is None
        assert download[1].content is None
        assert download[1].error == "file_not_found"
        assert download[2].content == bytes(range(16))
        assert download[2].error is None
