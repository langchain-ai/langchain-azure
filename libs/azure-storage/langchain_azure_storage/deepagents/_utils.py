"""Internal helpers: virtual-path/blob-key conversion and FileInfo building."""

from __future__ import annotations

from deepagents.backends.protocol import FileInfo
from deepagents.backends.utils import validate_path


def normalize_path(path: str) -> str:
    """Normalize a virtual filesystem path.

    Validates the path against Deep Agents' virtual filesystem rules, then
    returns it without a leading slash for blob key construction.

    Args:
        path: Virtual filesystem path (e.g., "/src/main.py").

    Returns:
        Normalized path without leading slash (e.g., "src/main.py").
    """
    if path == "":
        return ""

    normalized = validate_path(path)
    if normalized == "/":
        return ""
    return "/".join(part for part in normalized.split("/") if part)


def to_blob_key(prefix: str, path: str) -> str:
    """Convert a virtual filesystem path to a blob key.

    Args:
        prefix: Container prefix (e.g., "agent-workspace/").
        path: Virtual filesystem path (e.g., "/src/main.py").

    Returns:
        Full blob key (e.g., "agent-workspace/src/main.py").
    """
    normalized = normalize_path(path)
    if not prefix:
        return normalized
    # Ensure prefix ends with /
    p = prefix if prefix.endswith("/") else prefix + "/"
    return p + normalized


def from_blob_key(prefix: str, blob_name: str) -> str:
    """Convert a blob key back to a virtual filesystem path.

    Args:
        prefix: Container prefix (e.g., "agent-workspace/").
        blob_name: Full blob key (e.g., "agent-workspace/src/main.py").

    Returns:
        Virtual filesystem path with leading slash (e.g., "/src/main.py").
    """
    if prefix:
        p = prefix if prefix.endswith("/") else prefix + "/"
        if blob_name.startswith(p):
            blob_name = blob_name[len(p) :]
    return "/" + blob_name if blob_name else "/"


def get_prefix_for_path(prefix: str, path: str) -> str:
    """Get the blob prefix for listing a directory path.

    Args:
        prefix: Container prefix.
        path: Virtual filesystem directory path.

    Returns:
        Blob prefix string for listing (with trailing slash).
    """
    normalized = normalize_path(path)
    if not prefix and not normalized:
        return ""
    if not prefix:
        return normalized + "/" if normalized else ""
    p = prefix if prefix.endswith("/") else prefix + "/"
    if not normalized:
        return p
    return p + normalized + "/"


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
