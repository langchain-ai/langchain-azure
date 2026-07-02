"""Integration tests for AzureBlobBackend (requires Azurite)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_azure_storage.deepagents import AzureBlobBackend


class TestWrite:
    async def test_write_new_file(self, backend: AzureBlobBackend) -> None:
        result = await backend.awrite("/hello.txt", "Hello, World!")
        assert result.error is None
        assert result.path == "/hello.txt"
        assert result.files_update is None

    async def test_write_existing_file_errors(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/exists.txt", "content")
        result = await backend.awrite("/exists.txt", "new content")
        assert result.error is not None
        assert "already exists" in result.error

    async def test_concurrent_write_allows_only_one_success(
        self, backend: AzureBlobBackend
    ) -> None:
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


class TestRead:
    async def test_read_returns_numbered_lines(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/test.txt", "line one\nline two\nline three")
        result = await backend.aread("/test.txt")
        # Backend returns raw content; the middleware adds line numbers.
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert content == "line one\nline two\nline three"
        assert result.file_data["encoding"] == "utf-8"
        assert "1\t" not in content

    async def test_read_with_offset_and_limit(self, backend: AzureBlobBackend) -> None:
        lines = "\n".join(f"line {i}" for i in range(1, 11))
        await backend.awrite("/lines.txt", lines)
        result = await backend.aread("/lines.txt", offset=2, limit=3)
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert content == "line 3\nline 4\nline 5\n"
        assert "line 1" not in content
        assert "3\t" not in content  # no line-number formatting

    async def test_read_nonexistent_file(self, backend: AzureBlobBackend) -> None:
        result = await backend.aread("/nope.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_read_empty_file(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/empty.txt", "")
        result = await backend.aread("/empty.txt")
        # Raw empty content; the empty-file reminder is added by the middleware.
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

    async def test_read_binary_returns_base64(self, backend: AzureBlobBackend) -> None:
        import base64

        data = b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x01\x02"  # not valid UTF-8
        upload = await backend.aupload_files([("/img.png", data)])
        assert upload[0].error is None

        result = await backend.aread("/img.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == data


class TestEdit:
    async def test_edit_replaces_string(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/edit.txt", "Hello World")
        result = await backend.aedit("/edit.txt", "World", "Universe")
        assert result.error is None
        assert result.path == "/edit.txt"
        assert result.occurrences == 1
        assert result.files_update is None

        content = await backend.aread("/edit.txt")
        assert content.file_data is not None
        assert "Universe" in content.file_data["content"]

    async def test_edit_nonexistent_file(self, backend: AzureBlobBackend) -> None:
        result = await backend.aedit("/nope.txt", "old", "new")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_string_not_found(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/edit2.txt", "Hello World")
        result = await backend.aedit("/edit2.txt", "Nonexistent", "Replacement")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_multiple_occurrences_without_replace_all(
        self, backend: AzureBlobBackend
    ) -> None:
        await backend.awrite("/multi.txt", "aaa bbb aaa")
        result = await backend.aedit("/multi.txt", "aaa", "ccc")
        assert result.error is not None
        assert "2 times" in result.error

    async def test_edit_replace_all(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/multi2.txt", "aaa bbb aaa")
        result = await backend.aedit("/multi2.txt", "aaa", "ccc", replace_all=True)
        assert result.error is None
        assert result.occurrences == 2

        content = await backend.aread("/multi2.txt")
        assert content.file_data is not None
        assert "ccc" in content.file_data["content"]
        assert "aaa" not in content.file_data["content"]


class TestLs:
    async def test_ls_files_in_directory(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/src/main.py", "print('hello')")
        await backend.awrite("/src/utils.py", "# utils")
        result = await backend.als("/src")
        assert result.entries is not None
        assert result.error is None
        paths = [i["path"] for i in result.entries]
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths

    async def test_ls_synthesizes_directories(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/project/src/main.py", "code")
        await backend.awrite("/project/README.md", "readme")
        result = await backend.als("/project")
        assert result.entries is not None
        assert result.error is None
        paths = [i["path"] for i in result.entries]
        # Should have README.md as file and src/ as directory.
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


class TestGlob:
    async def test_glob_star_pattern(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/src/main.py", "code")
        await backend.awrite("/src/utils.py", "utils")
        await backend.awrite("/src/readme.md", "docs")
        result = await backend.aglob("*.py", "/src")
        assert result.matches is not None
        assert result.error is None
        paths = [i["path"] for i in result.matches]
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths
        assert "/src/readme.md" not in paths

    async def test_glob_recursive(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/project/src/main.py", "code")
        await backend.awrite("/project/src/lib/helpers.py", "helpers")
        await backend.awrite("/project/docs/guide.md", "guide")
        result = await backend.aglob("**/*.py", "/project")
        assert result.matches is not None
        assert result.error is None
        paths = [i["path"] for i in result.matches]
        assert "/project/src/main.py" in paths
        assert "/project/src/lib/helpers.py" in paths
        assert "/project/docs/guide.md" not in paths


class TestGrep:
    async def test_grep_finds_pattern(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/search/file1.py", "import os\nimport sys")
        await backend.awrite("/search/file2.py", "print('hello')")
        result = await backend.agrep("import", "/search")
        assert result.matches is not None
        assert result.error is None
        assert len(result.matches) == 2
        paths = [m["path"] for m in result.matches]
        assert "/search/file1.py" in paths

    async def test_grep_with_glob_filter(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/mixed/code.py", "import os")
        await backend.awrite("/mixed/notes.md", "import notes")
        result = await backend.agrep("import", "/mixed", glob="*.py")
        assert result.matches is not None
        assert result.error is None
        paths = [m["path"] for m in result.matches]
        assert "/mixed/code.py" in paths
        assert "/mixed/notes.md" not in paths

    async def test_grep_with_recursive_path_glob_filter(
        self, backend: AzureBlobBackend
    ) -> None:
        await backend.awrite("/src/top.py", "import os")
        await backend.awrite("/src/nested/deep.py", "import sys")
        result = await backend.agrep("import", "/", glob="src/*/*.py")
        assert result.matches is not None
        assert result.error is None
        paths = [m["path"] for m in result.matches]
        assert "/src/nested/deep.py" in paths
        assert "/src/top.py" not in paths

    async def test_grep_no_matches(self, backend: AzureBlobBackend) -> None:
        await backend.awrite("/grep_empty/file.txt", "nothing here")
        result = await backend.agrep("ZZZZZ", "/grep_empty")
        assert result.matches is not None
        assert result.error is None
        assert len(result.matches) == 0


class TestUploadDownload:
    async def test_roundtrip(self, backend: AzureBlobBackend) -> None:
        data = b"binary content \x00\x01\x02"
        upload_responses = await backend.aupload_files([("/bin/data.bin", data)])
        assert upload_responses[0].error is None

        download_responses = await backend.adownload_files(["/bin/data.bin"])
        assert download_responses[0].error is None
        assert download_responses[0].content == data

    async def test_download_nonexistent(self, backend: AzureBlobBackend) -> None:
        responses = await backend.adownload_files(["/nope/file.bin"])
        assert responses[0].error == "file_not_found"
        assert responses[0].content is None


class TestSyncMethods:
    """The synchronous methods run end-to-end against Azurite.

    They are invoked via ``asyncio.to_thread`` so the blocking sync SDK calls
    don't stall the test's event loop; the async fixtures still set up the
    shared container.
    """

    async def test_sync_methods_round_trip(self, backend: AzureBlobBackend) -> None:
        write_result = await asyncio.to_thread(
            backend.write, "/sync/hello.txt", "hello world TODO"
        )
        assert write_result.error is None

        read_content = await asyncio.to_thread(backend.read, "/sync/hello.txt")
        assert read_content.error is None
        assert read_content.file_data is not None
        assert "hello world TODO" in read_content.file_data["content"]

        await asyncio.to_thread(backend.write, "/sync/two.txt", "data")
        edit_result = await asyncio.to_thread(
            backend.edit, "/sync/hello.txt", "TODO", "DONE"
        )
        assert edit_result.error is None

        ls_result = await asyncio.to_thread(backend.ls, "/sync")
        assert ls_result.entries is not None
        assert any(fi.get("path") == "/sync/hello.txt" for fi in ls_result.entries)

        glob_result = await asyncio.to_thread(backend.glob, "**/*.txt", "/sync")
        assert glob_result.matches is not None
        assert {fi.get("path") for fi in glob_result.matches} >= {
            "/sync/hello.txt",
            "/sync/two.txt",
        }

        grep_result = await asyncio.to_thread(backend.grep, "DONE", "/sync")
        assert grep_result.matches is not None
        assert any(m.get("path") == "/sync/hello.txt" for m in grep_result.matches)

        upload = await asyncio.to_thread(
            backend.upload_files, [("/sync/three.bin", b"payload")]
        )
        assert upload[0].error is None

        download = await asyncio.to_thread(backend.download_files, ["/sync/three.bin"])
        assert download[0].error is None
        assert download[0].content == b"payload"

    async def test_concurrent_sync_reads(self, backend: AzureBlobBackend) -> None:
        # Concurrent sync calls from multiple worker threads must not interfere.
        await backend.awrite("/concurrent/a.txt", "alpha")
        await backend.awrite("/concurrent/b.txt", "bravo")
        await backend.awrite("/concurrent/c.txt", "charlie")

        results = await asyncio.gather(
            asyncio.to_thread(backend.read, "/concurrent/a.txt"),
            asyncio.to_thread(backend.read, "/concurrent/b.txt"),
            asyncio.to_thread(backend.read, "/concurrent/c.txt"),
        )
        for content, expected in zip(
            results, ("alpha", "bravo", "charlie"), strict=True
        ):
            assert content.error is None
            assert content.file_data is not None
            assert expected in content.file_data["content"]
