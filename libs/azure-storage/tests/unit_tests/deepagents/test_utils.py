"""Unit tests for the pure path utilities in ``deepagents._utils``."""

from __future__ import annotations

import pytest

# The [deepagents] extra (which the utils module lives under) is Python >= 3.11
# only, so skip the whole module when it is unavailable.
pytest.importorskip("deepagents")

from langchain_azure_storage.deepagents._utils import (  # noqa: E402
    build_file_info,
    from_blob_key,
    get_prefix_for_path,
    normalize_path,
    to_blob_key,
)


class TestNormalizePath:
    def test_strips_leading_slash(self) -> None:
        assert normalize_path("/src/main.py") == "src/main.py"

    def test_root(self) -> None:
        assert normalize_path("/") == ""

    def test_double_slashes(self) -> None:
        assert normalize_path("//src//main.py") == "src/main.py"

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="Path traversal"):
            normalize_path("/src/../secrets.txt")

    def test_rejects_windows_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="Windows absolute paths"):
            normalize_path("C:/temp/file.txt")


class TestToBlobKey:
    def test_with_prefix(self) -> None:
        assert to_blob_key("workspace/", "/src/main.py") == "workspace/src/main.py"

    def test_without_prefix(self) -> None:
        assert to_blob_key("", "/src/main.py") == "src/main.py"

    def test_prefix_no_trailing_slash(self) -> None:
        assert to_blob_key("workspace", "/src/main.py") == "workspace/src/main.py"


class TestFromBlobKey:
    def test_with_prefix(self) -> None:
        assert from_blob_key("workspace/", "workspace/src/main.py") == "/src/main.py"

    def test_without_prefix(self) -> None:
        assert from_blob_key("", "src/main.py") == "/src/main.py"


class TestGetPrefixForPath:
    def test_root_with_prefix(self) -> None:
        assert get_prefix_for_path("workspace/", "/") == "workspace/"

    def test_subdir_with_prefix(self) -> None:
        assert get_prefix_for_path("workspace/", "/src") == "workspace/src/"

    def test_subdir_no_prefix(self) -> None:
        assert get_prefix_for_path("", "/src") == "src/"


class TestBuildFileInfo:
    def test_defaults(self) -> None:
        assert build_file_info("/src/main.py") == {
            "path": "/src/main.py",
            "is_dir": False,
            "size": 0,
            "modified_at": "",
        }

    def test_directory(self) -> None:
        assert build_file_info("/src/", is_dir=True)["is_dir"] is True
