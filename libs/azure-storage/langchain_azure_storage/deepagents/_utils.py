"""Internal helpers for building FileInfo and content formatting."""

from __future__ import annotations

from deepagents.backends.protocol import FileInfo


def build_file_info(
    path: str,
    *,
    is_dir: bool = False,
    size: int = 0,
    modified_at: str = "",
) -> FileInfo:
    """Build a FileInfo TypedDict.

    Args:
        path: Virtual filesystem path.
        is_dir: Whether this entry is a directory.
        size: File size in bytes.
        modified_at: ISO 8601 modification timestamp.

    Returns:
        FileInfo dict.
    """
    return {
        "path": path,
        "is_dir": is_dir,
        "size": size,
        "modified_at": modified_at,
    }
