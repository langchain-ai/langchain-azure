"""Constants for the Azure Content Understanding document loader."""

from __future__ import annotations

from typing import Dict

# Mapping from filetype / mimetypes variant MIMEs to CU's canonical values.
# filetype uses some x-prefixed variants that differ from CU's supported set.
MIME_ALIASES: Dict[str, str] = {
    "audio/x-wav": "audio/wav",
    "audio/x-flac": "audio/flac",
    "video/x-m4v": "video/mp4",
}

# Mapping from media type prefix to the appropriate prebuilt CU analyzer.
# Used when analyzer_id is None (auto-detect mode).
MEDIA_TYPE_ANALYZER_MAP: Dict[str, str] = {
    "audio/": "prebuilt-audioSearch",
    "video/": "prebuilt-videoSearch",
}

DEFAULT_ANALYZER: str = "prebuilt-documentSearch"
