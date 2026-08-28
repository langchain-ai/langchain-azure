"""Unit tests for the vendored text/binary extension classifier."""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

from langchain_azure_compute.sandboxes._results import (  # noqa: E402
    _NON_TEXT_EXTENSIONS,
    is_text_file,
)


class TestClassification:
    @pytest.mark.parametrize(
        "path",
        ["/a/notes.txt", "/a/script.py", "/a/README", "/a/data.json", "/a/.gitignore"],
    )
    def test_text_extensions(self, path: str) -> None:
        assert is_text_file(path) is True

    @pytest.mark.parametrize(
        "path", ["/a/logo.png", "/a/clip.mkv", "/a/song.flac", "/a/deck.pptx"]
    )
    def test_non_text_extensions(self, path: str) -> None:
        assert is_text_file(path) is False

    def test_extension_match_is_case_insensitive(self) -> None:
        assert is_text_file("/a/LOGO.PNG") is False


class TestParityWithDeepagents:
    """The vendored table must not drift from the installed deepagents.

    `_get_backend_read_file_type` is private, which is why it is vendored
    rather than imported -- but a silent drift would change which files
    `read()` returns as base64, so it is asserted against here instead.
    """

    def test_matches_installed_classification(self) -> None:
        from deepagents.backends.utils import (  # noqa: PLC2701
            _EXTENSION_TO_FILE_TYPE,
            _get_backend_read_file_type,
        )

        for ext in set(_EXTENSION_TO_FILE_TYPE) | _NON_TEXT_EXTENSIONS:
            path = f"/tmp/sample{ext}"
            expected = _get_backend_read_file_type(path) == "text"
            assert is_text_file(path) is expected, ext

    def test_covers_the_extra_video_containers(self) -> None:
        from deepagents.backends.utils import (  # noqa: PLC2701
            _VIDEO_EXTRA_EXTENSIONS,
        )

        assert set(_VIDEO_EXTRA_EXTENSIONS) <= _NON_TEXT_EXTENSIONS
