"""Unit tests for the pure path utilities in ``deepagents._utils``."""

from __future__ import annotations

import pytest

# The [deepagents] extra (which the utils module lives under) is Python >= 3.11
# only, so skip the whole module when it is unavailable.
pytest.importorskip("deepagents")

from langchain_azure_storage.deepagents._utils import (  # noqa: E402
    _NON_TEXT_EXTENSIONS,
    build_file_info,
    from_blob_key,
    get_prefix_for_path,
    is_text_file,
    normalize_path,
    to_blob_key,
)


class TestNormalizePath:
    def test_strips_leading_slash(self) -> None:
        assert normalize_path("/src/main.py") == "src/main.py"

    def test_root(self) -> None:
        assert normalize_path("/") == ""

    def test_empty_string(self) -> None:
        assert normalize_path("") == ""

    def test_trailing_slash(self) -> None:
        assert normalize_path("/src/") == "src"

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

    def test_root(self) -> None:
        assert to_blob_key("workspace/", "/") == "workspace/"

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

    def test_root_no_prefix(self) -> None:
        assert get_prefix_for_path("", "/") == ""


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


class TestIsTextFile:
    def test_text_extension(self) -> None:
        assert is_text_file("/src/main.py") is True

    def test_non_text_extension(self) -> None:
        assert is_text_file("/img/logo.png") is False

    def test_extension_is_case_insensitive(self) -> None:
        assert is_text_file("/img/LOGO.PNG") is False

    def test_no_extension_defaults_to_text(self) -> None:
        assert is_text_file("/notes") is True

    def test_unknown_extension_defaults_to_text(self) -> None:
        assert is_text_file("/data.custom") is True

    def test_vendored_set_matches_installed_deepagents(self) -> None:
        # Drift canary for the vendored `_NON_TEXT_EXTENSIONS` set: it must
        # classify exactly like the installed deepagents' private
        # `_get_file_type` helper. If a deepagents release adds, removes, or
        # relocates extensions, this fails in CI instead of the backend
        # silently classifying reads differently from the reference backends.
        from deepagents.backends.utils import _EXTENSION_TO_FILE_TYPE, _get_file_type

        upstream_non_text = {
            ext
            for ext in _EXTENSION_TO_FILE_TYPE
            if _get_file_type(f"f{ext}") != "text"
        }
        assert upstream_non_text == _NON_TEXT_EXTENSIONS
