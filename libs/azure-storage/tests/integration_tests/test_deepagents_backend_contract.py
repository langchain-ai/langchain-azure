"""BackendProtocol contract tests for AzureBlobBackend.

This module is a fork of ``langchain_tests.integration_tests.sandboxes.
SandboxIntegrationTests`` trimmed to the ``BackendProtocol`` surface (no
``execute``): shell-based verification is replaced with ``read``/
``download_files``, sandbox-only cases (command execution, binary preview size
limits, chmod-based permission errors, relative-path rejection) are dropped,
and path/directory assertions are adapted to blob semantics (absolute virtual
paths, synthesized directories with trailing slashes).

TODO: langchain-ai/langchain#37905 tracks a shared ``BackendProtocol`` test
suite in ``langchain-tests``; once that lands, delete this fork and subclass
the shared suite instead.

Blob-specific behavior (concurrency, middleware contract, sync methods) is
covered in ``test_deepagents_backend.py``.
"""

from __future__ import annotations

import base64
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


class TestWriteContract:
    def test_write_new_file(self, backend: AzureBlobBackend) -> None:
        content = "Hello, backend!\nLine 2\nLine 3"
        result = backend.write("/new_file.txt", content)
        assert result.error is None
        assert result.path == "/new_file.txt"
        read_back = backend.read("/new_file.txt")
        assert read_back.error is None
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == content

    def test_write_creates_parent_dirs(self, backend: AzureBlobBackend) -> None:
        content = "Nested file content"
        result = backend.write("/deep/nested/dir/file.txt", content)
        assert result.error is None
        assert result.path == "/deep/nested/dir/file.txt"
        read_back = backend.read("/deep/nested/dir/file.txt")
        assert read_back.error is None
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == content

    def test_write_existing_file_fails(self, backend: AzureBlobBackend) -> None:
        backend.write("/existing.txt", "First content")
        result = backend.write("/existing.txt", "Second content")
        assert result.error is not None
        assert "already exists" in result.error.lower()
        read_back = backend.read("/existing.txt")
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == "First content"

    def test_write_special_characters(self, backend: AzureBlobBackend) -> None:
        content = (
            "Special chars: $VAR, `command`, $(subshell), 'quotes', \"quotes\"\n"
            "Tab\there\n"
            "Backslash: \\\\"
        )
        result = backend.write("/special.txt", content)
        assert result.error is None
        read_back = backend.read("/special.txt")
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == content

    def test_write_empty_file(self, backend: AzureBlobBackend) -> None:
        result = backend.write("/empty.txt", "")
        assert result.error is None
        read_back = backend.read("/empty.txt")
        assert read_back.error is None
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == ""

    def test_write_path_with_spaces(self, backend: AzureBlobBackend) -> None:
        content = "Content in file with spaces"
        result = backend.write("/dir with spaces/file name.txt", content)
        assert result.error is None
        read_back = backend.read("/dir with spaces/file name.txt")
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == content

    def test_write_unicode_content(self, backend: AzureBlobBackend) -> None:
        content = "Hello 👋 世界 مرحبا Привет 🌍\nLine with émojis 🎉"
        result = backend.write("/unicode.txt", content)
        assert result.error is None
        read_back = backend.read("/unicode.txt")
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == content

    def test_write_consecutive_slashes_in_path(self, backend: AzureBlobBackend) -> None:
        result = backend.write("/dir//file.txt", "Content")
        assert result.error is None
        assert result.path == "/dir/file.txt"
        read_back = backend.read("/dir/file.txt")
        assert read_back.file_data is not None
        assert read_back.file_data["content"] == "Content"

    def test_write_very_long_content(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Line {i} with some content here" for i in range(1000)])
        result = backend.write("/very_long.txt", content)
        assert result.error is None
        read_back = backend.read("/very_long.txt")
        assert read_back.error is None
        assert read_back.file_data is not None
        assert "Line 0 with some content here" in read_back.file_data["content"]

    def test_write_content_with_only_newlines(self, backend: AzureBlobBackend) -> None:
        result = backend.write("/only_newlines.txt", "\n\n\n\n\n")
        assert result.error is None
        download = backend.download_files(["/only_newlines.txt"])
        assert download[0].error is None
        assert download[0].content == b"\n\n\n\n\n"


class TestReadContract:
    def test_read_basic_file(self, backend: AzureBlobBackend) -> None:
        content = "Line 1\nLine 2\nLine 3"
        backend.write("/read_test.txt", content)
        result = backend.read("/read_test.txt")
        assert result.error is None
        assert result.file_data is not None
        assert all(
            line in result.file_data["content"]
            for line in ("Line 1", "Line 2", "Line 3")
        )

    def test_read_binary_file(self, backend: AzureBlobBackend) -> None:
        raw_bytes = bytes(range(256))
        backend.upload_files([("/binary.png", raw_bytes)])
        result = backend.read("/binary.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    def test_read_binary_file_100_kib(self, backend: AzureBlobBackend) -> None:
        raw_bytes = bytes(range(256)) * 400
        backend.upload_files([("/binary_100kib.png", raw_bytes)])
        result = backend.read("/binary_100kib.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    def test_read_nonexistent_file(self, backend: AzureBlobBackend) -> None:
        result = backend.read("/nonexistent.txt")
        assert result.error is not None
        assert (
            "not_found" in result.error.lower() or "not found" in result.error.lower()
        )

    def test_read_empty_file(self, backend: AzureBlobBackend) -> None:
        backend.write("/empty_read.txt", "")
        result = backend.read("/empty_read.txt")
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "empty" in content.lower() or content.strip() == ""

    def test_read_with_offset(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Row_{i}_content" for i in range(1, 11)])
        backend.write("/offset_test.txt", content)
        result = backend.read("/offset_test.txt", offset=5)
        assert result.error is None
        assert result.file_data is not None
        assert "Row_6_content" in result.file_data["content"]
        assert "Row_1_content" not in result.file_data["content"]

    def test_read_with_limit(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Row_{i}_content" for i in range(1, 101)])
        backend.write("/limit_test.txt", content)
        result = backend.read("/limit_test.txt", offset=0, limit=5)
        assert result.error is None
        assert result.file_data is not None
        assert "Row_1_content" in result.file_data["content"]
        assert "Row_5_content" in result.file_data["content"]
        assert "Row_6_content" not in result.file_data["content"]

    def test_read_with_offset_and_limit(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Row_{i}_content" for i in range(1, 21)])
        backend.write("/offset_limit_test.txt", content)
        result = backend.read("/offset_limit_test.txt", offset=10, limit=5)
        assert result.error is None
        assert result.file_data is not None
        assert "Row_11_content" in result.file_data["content"]
        assert "Row_15_content" in result.file_data["content"]
        assert "Row_10_content" not in result.file_data["content"]
        assert "Row_16_content" not in result.file_data["content"]

    def test_read_unicode_content(self, backend: AzureBlobBackend) -> None:
        content = "Hello 👋 世界\nПривет мир\nمرحبا العالم"  # noqa: RUF001
        backend.write("/unicode_read.txt", content)
        result = backend.read("/unicode_read.txt")
        assert result.error is None
        assert result.file_data is not None
        assert "👋" in result.file_data["content"]
        assert "世界" in result.file_data["content"]
        assert "Привет" in result.file_data["content"]

    def test_read_file_with_very_long_lines(self, backend: AzureBlobBackend) -> None:
        long_line = "x" * 3000
        backend.write("/long_lines.txt", f"Short line\n{long_line}\nAnother short line")
        result = backend.read("/long_lines.txt")
        assert result.error is None
        assert result.file_data is not None
        assert "Short line" in result.file_data["content"]

    def test_read_with_zero_limit(self, backend: AzureBlobBackend) -> None:
        backend.write("/zero_limit.txt", "Line 1\nLine 2\nLine 3")
        result = backend.read("/zero_limit.txt", offset=0, limit=0)
        content = result.file_data["content"] if result.file_data else ""
        assert "Line 1" not in content or content.strip() == ""

    def test_read_offset_beyond_file_length(self, backend: AzureBlobBackend) -> None:
        backend.write("/offset_beyond.txt", "Line 1\nLine 2\nLine 3")
        result = backend.read("/offset_beyond.txt", offset=100, limit=10)
        content = result.file_data["content"] if result.file_data else ""
        error = result.error or ""
        for line in ("Line 1", "Line 2", "Line 3"):
            assert line not in content
            assert line not in error

    def test_read_offset_at_exact_file_length(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Line {i}" for i in range(1, 6)])
        backend.write("/offset_exact.txt", content)
        result = backend.read("/offset_exact.txt", offset=5, limit=10)
        text = result.file_data["content"] if result.file_data else ""
        error = result.error or ""
        assert "Line 1" not in text
        assert "Line 1" not in error
        assert "Line 5" not in text
        assert "Line 5" not in error

    def test_read_very_large_file_in_chunks(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Line_{i:04d}_content" for i in range(1000)])
        backend.write("/large_chunked.txt", content)

        first = backend.read("/large_chunked.txt", offset=0, limit=100)
        middle = backend.read("/large_chunked.txt", offset=500, limit=100)
        last = backend.read("/large_chunked.txt", offset=900, limit=100)

        assert first.error is None
        assert first.file_data is not None
        assert "Line_0000_content" in first.file_data["content"]
        assert "Line_0099_content" in first.file_data["content"]
        assert "Line_0100_content" not in first.file_data["content"]

        assert middle.error is None
        assert middle.file_data is not None
        assert "Line_0500_content" in middle.file_data["content"]
        assert "Line_0599_content" in middle.file_data["content"]
        assert "Line_0499_content" not in middle.file_data["content"]

        assert last.error is None
        assert last.file_data is not None
        assert "Line_0900_content" in last.file_data["content"]
        assert "Line_0999_content" in last.file_data["content"]

    def test_read_path_is_sanitized(self, backend: AzureBlobBackend) -> None:
        malicious_path = "'; import os; os.system('echo INJECTED'); #"
        result = backend.read(malicious_path)
        assert result.error is not None
        assert result.file_data is None


class TestEditContract:
    def test_edit_single_occurrence(self, backend: AzureBlobBackend) -> None:
        backend.write("/edit_single.txt", "Hello world\nGoodbye world\nHello again")
        result = backend.edit("/edit_single.txt", "Goodbye", "Farewell")
        assert result.error is None
        assert result.occurrences == 1
        read_back = backend.read("/edit_single.txt")
        assert read_back.error is None
        assert read_back.file_data is not None
        assert "Farewell world" in read_back.file_data["content"]
        assert "Goodbye" not in read_back.file_data["content"]

    def test_edit_multiple_occurrences_without_replace_all(
        self, backend: AzureBlobBackend
    ) -> None:
        backend.write("/edit_multi.txt", "apple\nbanana\napple\norange\napple")
        result = backend.edit("/edit_multi.txt", "apple", "pear", replace_all=False)
        assert result.error is not None
        read_back = backend.read("/edit_multi.txt")
        assert read_back.file_data is not None
        assert "apple" in read_back.file_data["content"]
        assert "pear" not in read_back.file_data["content"]

    def test_edit_multiple_occurrences_with_replace_all(
        self, backend: AzureBlobBackend
    ) -> None:
        backend.write("/edit_replace_all.txt", "apple\nbanana\napple\norange\napple")
        result = backend.edit(
            "/edit_replace_all.txt", "apple", "pear", replace_all=True
        )
        assert result.error is None
        assert result.occurrences == 3
        read_back = backend.read("/edit_replace_all.txt")
        assert read_back.file_data is not None
        assert "apple" not in read_back.file_data["content"]
        assert read_back.file_data["content"].count("pear") == 3

    def test_edit_string_not_found(self, backend: AzureBlobBackend) -> None:
        backend.write("/edit_not_found.txt", "Hello world")
        result = backend.edit("/edit_not_found.txt", "nonexistent", "replacement")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_nonexistent_file(self, backend: AzureBlobBackend) -> None:
        result = backend.edit("/nonexistent_edit.txt", "old", "new")
        assert result.error is not None
        assert (
            "not_found" in result.error.lower() or "not found" in result.error.lower()
        )

    def test_edit_special_characters(self, backend: AzureBlobBackend) -> None:
        backend.write(
            "/edit_special.txt", "Price: $100.00\nPattern: [a-z]*\nPath: /usr/bin"
        )
        first = backend.edit("/edit_special.txt", "$100.00", "$200.00")
        second = backend.edit("/edit_special.txt", "[a-z]*", "[0-9]+")
        assert first.error is None
        assert second.error is None
        read_back = backend.read("/edit_special.txt")
        assert read_back.file_data is not None
        assert "$200.00" in read_back.file_data["content"]
        assert "[0-9]+" in read_back.file_data["content"]

    def test_edit_multiline_support(self, backend: AzureBlobBackend) -> None:
        backend.write("/edit_multiline.txt", "Line 1\nLine 2\nLine 3")
        result = backend.edit("/edit_multiline.txt", "Line 1\nLine 2", "Combined")
        assert result.error is None
        assert result.occurrences == 1
        read_back = backend.read("/edit_multiline.txt")
        assert read_back.file_data is not None
        assert "Combined" in read_back.file_data["content"]


class TestLsContract:
    def test_ls_lists_files(self, backend: AzureBlobBackend) -> None:
        backend.write("/ls/a.txt", "a")
        backend.write("/ls/b.txt", "b")
        result = backend.ls("/ls")
        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert "/ls/a.txt" in paths
        assert "/ls/b.txt" in paths

    def test_ls_lists_nested_directories(self, backend: AzureBlobBackend) -> None:
        backend.write("/ls_nested/root.txt", "content")
        backend.write("/ls_nested/subdir/child.txt", "content")
        result = backend.ls("/ls_nested")
        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        # Synthesized directories carry a trailing slash.
        assert "/ls_nested/subdir/" in paths
        assert "/ls_nested/root.txt" in paths

    def test_ls_unicode_filenames(self, backend: AzureBlobBackend) -> None:
        backend.write("/ls_unicode/测试文件.txt", "content")
        backend.write("/ls_unicode/файл.txt", "content")
        result = backend.ls("/ls_unicode")
        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert "/ls_unicode/测试文件.txt" in paths
        assert "/ls_unicode/файл.txt" in paths

    def test_ls_large_directory(self, backend: AzureBlobBackend) -> None:
        files = [(f"/ls_large/file_{i:03d}.txt", b"content") for i in range(50)]
        upload = backend.upload_files(files)
        assert all(response.error is None for response in upload)
        result = backend.ls("/ls_large")
        assert result.error is None
        assert result.entries is not None
        assert len(result.entries) == 50
        paths = [entry["path"] for entry in result.entries]
        assert "/ls_large/file_000.txt" in paths
        assert "/ls_large/file_049.txt" in paths

    def test_ls_path_with_trailing_slash(self, backend: AzureBlobBackend) -> None:
        backend.write("/ls_trailing/file.txt", "content")
        result = backend.ls("/ls_trailing/")
        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert "/ls_trailing/file.txt" in paths

    def test_ls_special_characters_in_filenames(
        self, backend: AzureBlobBackend
    ) -> None:
        backend.write("/ls_special/file(1).txt", "content")
        backend.write("/ls_special/file[2].txt", "content")
        backend.write("/ls_special/file-3.txt", "content")
        result = backend.ls("/ls_special")
        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert "/ls_special/file(1).txt" in paths
        assert "/ls_special/file[2].txt" in paths
        assert "/ls_special/file-3.txt" in paths

    def test_ls_path_is_sanitized(self, backend: AzureBlobBackend) -> None:
        malicious_path = "'; import os; os.system('echo INJECTED'); #"
        result = backend.ls(malicious_path)
        assert result.error is not None or result.entries == []


class TestGlobContract:
    def test_glob_basic_pattern(self, backend: AzureBlobBackend) -> None:
        backend.write("/glob_test/file1.txt", "content")
        backend.write("/glob_test/file2.txt", "content")
        backend.write("/glob_test/file3.py", "content")
        result = backend.glob("*.txt", path="/glob_test")
        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert len(paths) == 2
        assert "/glob_test/file1.txt" in paths
        assert "/glob_test/file2.txt" in paths
        assert not any(path.endswith(".py") for path in paths)

    def test_glob_recursive_pattern(self, backend: AzureBlobBackend) -> None:
        backend.write("/glob_recursive/root.txt", "content")
        backend.write("/glob_recursive/subdir1/nested1.txt", "content")
        backend.write("/glob_recursive/subdir2/nested2.txt", "content")
        result = backend.glob("**/*.txt", path="/glob_recursive")
        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert any(path.endswith("nested1.txt") for path in paths)
        assert any(path.endswith("nested2.txt") for path in paths)

    def test_glob_no_matches(self, backend: AzureBlobBackend) -> None:
        backend.write("/glob_empty/file.txt", "content")
        result = backend.glob("*.py", path="/glob_empty")
        assert result.error is None
        assert result.matches == []

    def test_glob_hidden_files_explicitly(self, backend: AzureBlobBackend) -> None:
        backend.write("/glob_hidden/.hidden1", "content")
        backend.write("/glob_hidden/.hidden2", "content")
        backend.write("/glob_hidden/visible.txt", "content")
        result = backend.glob(".*", path="/glob_hidden")
        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert "/glob_hidden/.hidden1" in paths or "/glob_hidden/.hidden2" in paths
        assert "/glob_hidden/visible.txt" not in paths

    def test_glob_with_character_class(self, backend: AzureBlobBackend) -> None:
        backend.write("/glob_charclass/file1.txt", "content")
        backend.write("/glob_charclass/file2.txt", "content")
        backend.write("/glob_charclass/file3.txt", "content")
        backend.write("/glob_charclass/fileA.txt", "content")
        result = backend.glob("file[1-2].txt", path="/glob_charclass")
        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert len(paths) == 2
        assert "/glob_charclass/file1.txt" in paths
        assert "/glob_charclass/file2.txt" in paths

    def test_glob_with_question_mark(self, backend: AzureBlobBackend) -> None:
        backend.write("/glob_question/file1.txt", "content")
        backend.write("/glob_question/file2.txt", "content")
        backend.write("/glob_question/file10.txt", "content")
        result = backend.glob("file?.txt", path="/glob_question")
        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert len(paths) == 2
        assert "/glob_question/file1.txt" in paths
        assert "/glob_question/file2.txt" in paths
        assert "/glob_question/file10.txt" not in paths


class TestGrepContract:
    def test_grep_basic_search(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_test/file1.txt", "Hello world\nGoodbye world")
        backend.write("/grep_test/file2.txt", "Hello there\nGoodbye friend")
        result = backend.grep("Hello", path="/grep_test")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 2
        paths = [match["path"] for match in result.matches]
        assert any(path.endswith("file1.txt") for path in paths)
        assert any(path.endswith("file2.txt") for path in paths)
        assert all(match["line"] == 1 for match in result.matches)

    def test_grep_literal(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_lit/grep.txt", "a (b)\nstr | int\n")
        result = backend.grep("str | int", path="/grep_lit")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) > 0
        assert result.matches[0]["path"].endswith("/grep.txt")
        assert result.matches[0]["text"].strip() == "str | int"

    def test_grep_with_glob_pattern(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_glob/test.txt", "pattern")
        backend.write("/grep_glob/test.py", "pattern")
        backend.write("/grep_glob/test.md", "pattern")
        result = backend.grep("pattern", path="/grep_glob", glob="*.py")
        assert result.error is None
        assert result.matches == [
            {"path": "/grep_glob/test.py", "line": 1, "text": "pattern"}
        ]

    def test_grep_no_matches(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_none/file.txt", "Hello world")
        result = backend.grep("nonexistent", path="/grep_none")
        assert result.error is None
        assert result.matches == []

    def test_grep_multiple_matches_per_file(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_multi/fruits.txt", "apple\nbanana\napple\norange\napple")
        result = backend.grep("apple", path="/grep_multi")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 3
        assert [match["line"] for match in result.matches] == [1, 3, 5]

    def test_grep_literal_string_matching(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_literal/numbers.txt", "test123\ntest456\nabcdef")
        result = backend.grep("test123", path="/grep_literal")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "test123" in result.matches[0]["text"]

    def test_grep_unicode_pattern(self, backend: AzureBlobBackend) -> None:
        backend.write(
            "/grep_unicode/unicode.txt",
            "Hello 世界\nПривет мир\n测试 pattern",  # noqa: RUF001
        )
        result = backend.grep("世界", path="/grep_unicode")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "世界" in result.matches[0]["text"]

    def test_grep_case_sensitivity(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_case/case.txt", "Hello\nhello\nHELLO")
        result = backend.grep("Hello", path="/grep_case")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["text"] == "Hello"

    def test_grep_with_special_characters(self, backend: AzureBlobBackend) -> None:
        backend.write(
            "/grep_special/special.txt",
            "Price: $100\nPath: /usr/bin\nPattern: [a-z]*",
        )
        dollar = backend.grep("$100", path="/grep_special")
        brackets = backend.grep("[a-z]*", path="/grep_special")

        assert dollar.error is None
        assert dollar.matches is not None
        assert len(dollar.matches) == 1
        assert "$100" in dollar.matches[0]["text"]

        assert brackets.error is None
        assert brackets.matches is not None
        assert len(brackets.matches) == 1
        assert "[a-z]*" in brackets.matches[0]["text"]

    def test_grep_empty_directory(self, backend: AzureBlobBackend) -> None:
        result = backend.grep("anything", path="/grep_empty_dir")
        assert result.error is None
        assert result.matches == []

    def test_grep_across_nested_directories(self, backend: AzureBlobBackend) -> None:
        backend.write("/grep_nested/root.txt", "target here")
        backend.write("/grep_nested/sub1/level1.txt", "target here")
        backend.write("/grep_nested/sub1/sub2/level2.txt", "target here")
        result = backend.grep("target", path="/grep_nested")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 3

    def test_grep_with_globstar_include_pattern(
        self, backend: AzureBlobBackend
    ) -> None:
        backend.write("/grep_globstar/a/b/target.py", "needle")
        backend.write("/grep_globstar/a/ignore.txt", "needle")
        result = backend.grep("needle", path="/grep_globstar", glob="*.py")
        assert result.error is None
        assert result.matches == [
            {"path": "/grep_globstar/a/b/target.py", "line": 1, "text": "needle"}
        ]

    def test_grep_reports_correct_line_numbers(self, backend: AzureBlobBackend) -> None:
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        backend.write("/grep_lines/long.txt", content)
        result = backend.grep("Line 50", path="/grep_lines")
        assert result.error is None
        assert result.matches == [
            {"path": "/grep_lines/long.txt", "line": 50, "text": "Line 50"}
        ]


class TestUploadDownloadContract:
    def test_upload_single_file(self, backend: AzureBlobBackend) -> None:
        content = b"Hello, backend!"
        upload = backend.upload_files([("/upload_single.txt", content)])
        assert len(upload) == 1
        assert upload[0].path == "/upload_single.txt"
        assert upload[0].error is None
        download = backend.download_files(["/upload_single.txt"])
        assert download[0].content == content

    def test_download_single_file(self, backend: AzureBlobBackend) -> None:
        content = b"Download test content"
        backend.upload_files([("/download_single.txt", content)])
        download = backend.download_files(["/download_single.txt"])
        assert len(download) == 1
        assert download[0].path == "/download_single.txt"
        assert download[0].content == content
        assert download[0].error is None

    def test_upload_download_roundtrip(self, backend: AzureBlobBackend) -> None:
        content = b"Roundtrip test: special chars \n\t\r\x00"
        upload = backend.upload_files([("/roundtrip.txt", content)])
        assert upload[0].path == "/roundtrip.txt"
        assert upload[0].error is None
        download = backend.download_files(["/roundtrip.txt"])
        assert download[0].path == "/roundtrip.txt"
        assert download[0].content == content
        assert download[0].error is None

    def test_upload_multiple_files_order_preserved(
        self, backend: AzureBlobBackend
    ) -> None:
        files = [
            ("/multi_1.txt", b"Content 1"),
            ("/multi_2.txt", b"Content 2"),
            ("/multi_3.txt", b"Content 3"),
        ]
        upload = backend.upload_files(files)
        assert [response.path for response in upload] == [path for path, _ in files]
        assert [response.error for response in upload] == [None, None, None]

    def test_download_multiple_files_order_preserved(
        self, backend: AzureBlobBackend
    ) -> None:
        files = [
            ("/batch_1.txt", b"Batch 1"),
            ("/batch_2.txt", b"Batch 2"),
            ("/batch_3.txt", b"Batch 3"),
        ]
        backend.upload_files(files)
        download = backend.download_files([path for path, _ in files])
        assert [response.path for response in download] == [path for path, _ in files]
        assert [response.content for response in download] == [
            content for _, content in files
        ]
        assert [response.error for response in download] == [None, None, None]

    def test_upload_binary_content_roundtrip(self, backend: AzureBlobBackend) -> None:
        content = bytes(range(256))
        upload = backend.upload_files([("/binary_file.bin", content)])
        assert upload[0].error is None
        download = backend.download_files(["/binary_file.bin"])
        assert download[0].content == content

    def test_upload_large_file_roundtrip(self, backend: AzureBlobBackend) -> None:
        content = b"0123456789abcdef" * 1024 * 640
        assert len(content) == 10 * 1024 * 1024
        upload = backend.upload_files([("/large_upload.bin", content)])
        assert upload[0].error is None
        download = backend.download_files(["/large_upload.bin"])
        assert download[0].error is None
        assert download[0].content == content

    def test_upload_missing_parent_dir_roundtrips(
        self, backend: AzureBlobBackend
    ) -> None:
        # Blob storage has no real directories, so uploading into a
        # nonexistent "directory" must succeed and roundtrip.
        content = b"nested upload"
        upload = backend.upload_files([("/missing_parent/upload.txt", content)])
        assert upload[0].error is None
        download = backend.download_files(["/missing_parent/upload.txt"])
        assert download[0].content == content

    def test_download_error_file_not_found(self, backend: AzureBlobBackend) -> None:
        download = backend.download_files(["/nonexistent_download.txt"])
        assert len(download) == 1
        assert download[0].path == "/nonexistent_download.txt"
        assert download[0].content is None
        assert download[0].error == "file_not_found"

    def test_download_error_directory_path(self, backend: AzureBlobBackend) -> None:
        backend.write("/some_directory/file.txt", "content")
        download = backend.download_files(["/some_directory"])
        assert len(download) == 1
        assert download[0].content is None
        assert download[0].error in {"is_directory", "file_not_found", "invalid_path"}

    def test_download_traversal_path_returns_invalid_path(
        self, backend: AzureBlobBackend
    ) -> None:
        download = backend.download_files(["/a/../b.txt"])
        assert len(download) == 1
        assert download[0].path == "/a/../b.txt"
        assert download[0].content is None
        assert download[0].error == "invalid_path"

    def test_upload_traversal_path_returns_invalid_path(
        self, backend: AzureBlobBackend
    ) -> None:
        upload = backend.upload_files([("/a/../b.txt", b"nope")])
        assert len(upload) == 1
        assert upload[0].path == "/a/../b.txt"
        assert upload[0].error == "invalid_path"


class TestAsyncLargePayloadContract:
    async def test_awrite_aread_large_text_payload(
        self, backend: AzureBlobBackend
    ) -> None:
        line = "0123456789abcdef" * 256
        lines = [line for _ in range(2560)]
        content = "\n".join(lines)

        write_result = await backend.awrite("/large_async_text.txt", content)
        assert write_result.error is None
        assert write_result.path == "/large_async_text.txt"

        read_result = await backend.aread("/large_async_text.txt")
        assert read_result.error is None
        assert read_result.file_data is not None
        assert read_result.file_data["encoding"] == "utf-8"
        assert read_result.file_data["content"].startswith(lines[0])

    async def test_aread_large_text_payload_paginated_roundtrip(
        self, backend: AzureBlobBackend
    ) -> None:
        lines = [f"Line_{i:04d}_content" for i in range(2500)]
        content = "\n".join(lines)

        write_result = await backend.awrite("/large_async_chunked.txt", content)
        assert write_result.error is None

        # slice_read_response keeps each line's terminator, so a mid-file page
        # ends with "\n" and the pages concatenate back to the exact content.
        parts: list[str] = []
        for offset in range(0, len(lines), 100):
            page = await backend.aread(
                "/large_async_chunked.txt", offset=offset, limit=100
            )
            assert page.error is None
            assert page.file_data is not None
            assert page.file_data["content"].startswith(lines[offset])
            parts.append(page.file_data["content"])

        assert "".join(parts) == content

    async def test_adownload_large_text_payload_roundtrip(
        self, backend: AzureBlobBackend
    ) -> None:
        line = "0123456789abcdef" * 256
        content = "\n".join([line for _ in range(2560)])

        write_result = await backend.awrite("/large_async_download.txt", content)
        assert write_result.error is None

        download = await backend.adownload_files(["/large_async_download.txt"])
        assert download[0].error is None
        assert download[0].content == content.encode("utf-8")

    def test_write_read_download_large_text_with_escaped_content(
        self, backend: AzureBlobBackend
    ) -> None:
        line = (
            "prefix\t☃世界π≈3.14159"
            " | spaces   preserved"
            " | quotes ' \""
            " | brackets [] {{}}"
            " | shell $VAR `cmd` $(subshell)"
            " | slash /tmp/path and backslash \\\\"
            " | control-ish \\r \\n"
            " | suffix"
        )
        lines = [f"{i:04d}:{line}" for i in range(2500)]
        content = "\n".join(lines)

        write_result = backend.write("/large_sync_escaped.txt", content)
        assert write_result.error is None

        # Pages keep their line terminators; see the paginated test above.
        pages: list[str] = []
        for offset in range(0, len(lines), 100):
            page = backend.read("/large_sync_escaped.txt", offset=offset, limit=100)
            assert page.error is None
            assert page.file_data is not None
            assert page.file_data["content"].startswith(lines[offset])
            pages.append(page.file_data["content"])

        assert "".join(pages) == content

        download = backend.download_files(["/large_sync_escaped.txt"])
        assert download[0].error is None
        assert download[0].content == content.encode("utf-8")

    async def test_awrite_aread_adownload_large_text_with_escaped_content(
        self, backend: AzureBlobBackend
    ) -> None:
        line = (
            "prefix\t☃世界π≈3.14159"
            " | spaces   preserved"
            " | quotes ' \""
            " | brackets [] {{}}"
            " | shell $VAR `cmd` $(subshell)"
            " | slash /tmp/path and backslash \\\\"
            " | control-ish \\r \\n"
            " | suffix"
        )
        lines = [f"{i:04d}:{line}" for i in range(2500)]
        content = "\n".join(lines)

        write_result = await backend.awrite("/large_async_escaped.txt", content)
        assert write_result.error is None

        # Pages keep their line terminators; see the paginated test above.
        pages: list[str] = []
        for offset in range(0, len(lines), 100):
            page = await backend.aread(
                "/large_async_escaped.txt", offset=offset, limit=100
            )
            assert page.error is None
            assert page.file_data is not None
            assert page.file_data["content"].startswith(lines[offset])
            pages.append(page.file_data["content"])

        assert "".join(pages) == content

        download = await backend.adownload_files(["/large_async_escaped.txt"])
        assert download[0].error is None
        assert download[0].content == content.encode("utf-8")

    async def test_aread_binary_image_file(self, backend: AzureBlobBackend) -> None:
        raw_bytes = bytes(range(256))
        upload = await backend.aupload_files([("/async_binary.png", raw_bytes)])
        assert upload[0].error is None

        result = await backend.aread("/async_binary.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    async def test_aread_binary_file_100_kib(self, backend: AzureBlobBackend) -> None:
        raw_bytes = bytes(range(256)) * 400
        upload = await backend.aupload_files([("/async_binary_100kib.png", raw_bytes)])
        assert upload[0].error is None

        result = await backend.aread("/async_binary_100kib.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    async def test_aupload_adownload_large_file_roundtrip(
        self, backend: AzureBlobBackend
    ) -> None:
        content = b"0123456789abcdef" * 1024 * 640
        assert len(content) == 10 * 1024 * 1024

        upload = await backend.aupload_files([("/large_async_upload.bin", content)])
        assert upload[0].error is None

        download = await backend.adownload_files(["/large_async_upload.bin"])
        assert download[0].error is None
        assert download[0].content == content
