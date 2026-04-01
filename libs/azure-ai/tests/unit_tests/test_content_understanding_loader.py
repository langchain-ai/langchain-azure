"""Unit tests for AzureAIContentUnderstandingLoader."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from langchain_azure_ai.document_loaders.content_understanding import (
    AzureAIContentUnderstandingLoader,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_span(offset: int, length: int) -> Mock:
    span = Mock()
    span.offset = offset
    span.length = length
    return span


_VALUE_ATTR_FOR_TYPE: Dict[str, str] = {
    "string": "value_string",
    "number": "value_number",
    "integer": "value_integer",
    "date": "value_date",
    "time": "value_time",
    "boolean": "value_boolean",
    "array": "value_array",
    "object": "value_object",
    "json": "value_json",
}


def _make_field(
    field_type: str,
    confidence: Optional[float] = 0.95,
    **kwargs: Any,
) -> Mock:
    field = Mock()
    field.type = field_type
    field.confidence = confidence
    # Set all possible value attributes to None first
    for attr in _VALUE_ATTR_FOR_TYPE.values():
        setattr(field, attr, None)
    # Override with provided kwargs
    for k, v in kwargs.items():
        setattr(field, k, v)
    # Set .value to mirror the SDK's convenience property
    value_attr = _VALUE_ATTR_FOR_TYPE.get(field_type)
    field.value = getattr(field, value_attr) if value_attr else None
    return field


def _make_page(page_number: int, offset: int, length: int) -> Mock:
    page = Mock()
    page.page_number = page_number
    page.spans = [_make_span(offset, length)]
    return page


def _make_document_content(
    markdown: str = "# Test\n\nContent",
    mime_type: str = "application/pdf",
    start_page: int = 1,
    end_page: int = 1,
    pages: Optional[List[Mock]] = None,
    fields: Optional[Dict[str, Mock]] = None,
    segments: Optional[List[Mock]] = None,
    category: Optional[str] = None,
) -> Mock:
    content = Mock(
        spec=[
            "kind",
            "mime_type",
            "markdown",
            "start_page_number",
            "end_page_number",
            "pages",
            "fields",
            "segments",
            "category",
        ]
    )
    content.kind = "document"
    content.mime_type = mime_type
    content.markdown = markdown
    content.start_page_number = start_page
    content.end_page_number = end_page
    content.pages = pages
    content.fields = fields
    content.segments = segments
    content.category = category
    # Make isinstance checks work
    content.__class__ = type("DocumentContent", (), {})
    return content


def _make_audio_visual_content(
    markdown: str = "Hello world transcript",
    mime_type: str = "audio/mpeg",
    start_time_ms: int = 0,
    end_time_ms: int = 60000,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fields: Optional[Dict[str, Mock]] = None,
    segments: Optional[List[Mock]] = None,
    category: Optional[str] = None,
) -> Mock:
    content = Mock(
        spec=[
            "kind",
            "mime_type",
            "markdown",
            "start_time_ms",
            "end_time_ms",
            "width",
            "height",
            "fields",
            "segments",
            "category",
        ]
    )
    content.kind = "audioVisual"
    content.mime_type = mime_type
    content.markdown = markdown
    content.start_time_ms = start_time_ms
    content.end_time_ms = end_time_ms
    content.width = width
    content.height = height
    content.fields = fields
    content.segments = segments
    content.category = category
    content.__class__ = type("AudioVisualContent", (), {})
    return content


def _make_result(
    contents: List[Mock],
    analyzer_id: str = "prebuilt-documentSearch",
) -> Mock:
    result = Mock()
    result.analyzer_id = analyzer_id
    result.contents = contents
    return result


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Tests for __init__ parameter validation."""

    def test_no_input_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Exactly one of"):
            AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                credential="key",
            )

    def test_multiple_input_sources_raises(self) -> None:
        with pytest.raises(ValueError, match="Exactly one of"):
            AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                credential="key",
                file_path="test.pdf",
                url="https://example.com/test.pdf",
            )

    def test_invalid_output_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="output_mode must be one of"):
            AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                credential="key",
                url="https://example.com/test.pdf",
                output_mode="invalid",  # type: ignore[arg-type]
            )

    def test_valid_file_path_input(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            file_path="report.pdf",
        )
        assert loader._source == "report.pdf"

    def test_valid_url_input(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
        )
        assert loader._source == "https://example.com/report.pdf"

    def test_valid_bytes_input(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"fake pdf bytes",
        )
        assert loader._source == "bytes_input"

    def test_custom_source_label(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"data",
            source="my-scan.pdf",
        )
        assert loader._source == "my-scan.pdf"

    def test_string_credential_converted(self) -> None:
        from azure.core.credentials import AzureKeyCredential

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="my-api-key",
            url="https://example.com/test.pdf",
        )
        assert isinstance(loader._credential, AzureKeyCredential)


class TestProjectEndpoint:
    """Tests for project_endpoint support."""

    def test_project_endpoint_extracts_base_url(self) -> None:
        from azure.core.credentials import TokenCredential

        cred = MagicMock(spec=TokenCredential)
        loader = AzureAIContentUnderstandingLoader(
            project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
            credential=cred,
            url="https://example.com/test.pdf",
        )
        assert loader._endpoint == "https://my-resource.services.ai.azure.com"

    def test_project_endpoint_and_endpoint_raises(self) -> None:
        from azure.core.credentials import TokenCredential

        cred = MagicMock(spec=TokenCredential)
        with pytest.raises(ValueError, match="mutually exclusive"):
            AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
                credential=cred,
                url="https://example.com/test.pdf",
            )

    def test_project_endpoint_rejects_api_key(self) -> None:
        with pytest.raises(ValueError, match="TokenCredential"):
            AzureAIContentUnderstandingLoader(
                project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
                credential="my-api-key",
                url="https://example.com/test.pdf",
            )

    @patch.dict(
        "os.environ",
        {
            "AZURE_AI_PROJECT_ENDPOINT": "https://env-resource.services.ai.azure.com/api/projects/env-project"
        },
    )
    def test_project_endpoint_from_env_var(self) -> None:
        from azure.core.credentials import TokenCredential

        cred = MagicMock(spec=TokenCredential)
        loader = AzureAIContentUnderstandingLoader(
            credential=cred,
            url="https://example.com/test.pdf",
        )
        assert loader._endpoint == "https://env-resource.services.ai.azure.com"

    def test_project_endpoint_defaults_credential(self) -> None:
        with patch("azure.identity.DefaultAzureCredential") as mock_dac:
            from azure.core.credentials import TokenCredential

            mock_cred = MagicMock(spec=TokenCredential)
            mock_dac.return_value = mock_cred
            loader = AzureAIContentUnderstandingLoader(
                project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
                url="https://example.com/test.pdf",
            )
            assert loader._endpoint == "https://my-resource.services.ai.azure.com"
            mock_dac.assert_called_once()

    def test_no_endpoint_no_env_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="An endpoint is required"):
                AzureAIContentUnderstandingLoader(
                    credential="key",
                    url="https://example.com/test.pdf",
                )


# ---------------------------------------------------------------------------
# MIME type detection and default analyzer
# ---------------------------------------------------------------------------


class TestMimeTypeAndAnalyzer:
    """Tests for MIME type detection and default analyzer selection."""

    @pytest.mark.parametrize(
        "path,expected_analyzer",
        [
            ("report.pdf", "prebuilt-documentSearch"),
            ("photo.png", "prebuilt-documentSearch"),
            ("scan.jpg", "prebuilt-documentSearch"),
            ("recording.mp3", "prebuilt-audioSearch"),
            ("recording.wav", "prebuilt-audioSearch"),
            ("video.mp4", "prebuilt-videoSearch"),
            ("video.mov", "prebuilt-videoSearch"),
        ],
    )
    def test_default_analyzer_by_extension(
        self, path: str, expected_analyzer: str
    ) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            file_path=path,
        )
        assert loader._analyzer_id == expected_analyzer

    def test_explicit_analyzer_overrides_default(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            analyzer_id="my-custom-analyzer",
            file_path="report.pdf",
        )
        assert loader._analyzer_id == "my-custom-analyzer"

    def test_bytes_source_defaults_to_document_search(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"data",
        )
        assert loader._analyzer_id == "prebuilt-documentSearch"

    def test_mime_alias_normalized(self) -> None:
        """Extension that maps to a variant MIME gets normalized via _MIME_ALIASES."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            file_path="recording.wav",
        )
        # mimetypes may return "audio/x-wav" or "audio/wav" depending on OS;
        # _MIME_ALIASES normalizes x-wav → wav, so the result is always wav.
        assert loader._mime_type == "audio/wav"
        assert loader._analyzer_id == "prebuilt-audioSearch"

    @patch("langchain_azure_ai.document_loaders.content_understanding._filetype")
    def test_bytes_source_sniffed_as_mp4(self, mock_ft: MagicMock) -> None:
        """bytes_source with video magic bytes → filetype sniffs video/mp4."""
        mock_kind = MagicMock()
        mock_kind.mime = "video/mp4"
        mock_ft.guess.return_value = mock_kind

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"\x00\x00\x00\x1cftypisom",
        )
        assert loader._mime_type == "video/mp4"
        assert loader._analyzer_id == "prebuilt-videoSearch"

    @patch("langchain_azure_ai.document_loaders.content_understanding._filetype")
    def test_bytes_source_sniffed_as_audio(self, mock_ft: MagicMock) -> None:
        """bytes_source with audio magic bytes → filetype sniffs audio/mpeg."""
        mock_kind = MagicMock()
        mock_kind.mime = "audio/mpeg"
        mock_ft.guess.return_value = mock_kind

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"\xff\xfb\x90\x00" + b"\x00" * 200,
        )
        assert loader._mime_type == "audio/mpeg"
        assert loader._analyzer_id == "prebuilt-audioSearch"

    @patch("langchain_azure_ai.document_loaders.content_understanding._filetype")
    def test_bytes_source_sniff_returns_alias(self, mock_ft: MagicMock) -> None:
        """filetype returning a variant MIME gets normalized via _MIME_ALIASES."""
        mock_kind = MagicMock()
        mock_kind.mime = "audio/x-wav"
        mock_ft.guess.return_value = mock_kind

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"RIFF" + b"\x00" * 200,
        )
        assert loader._mime_type == "audio/wav"
        assert loader._analyzer_id == "prebuilt-audioSearch"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding._filetype",
        None,
    )
    def test_bytes_source_no_filetype_library(self) -> None:
        """Without filetype lib installed, bytes_source returns None MIME."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"some bytes",
        )
        assert loader._mime_type is None
        assert loader._analyzer_id == "prebuilt-documentSearch"

    @patch("langchain_azure_ai.document_loaders.content_understanding._filetype")
    def test_bytes_source_filetype_returns_none(self, mock_ft: MagicMock) -> None:
        """filetype can't identify the bytes → falls back to None."""
        mock_ft.guess.return_value = None

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"unknown format",
        )
        assert loader._mime_type is None
        assert loader._analyzer_id == "prebuilt-documentSearch"

    def test_extension_based_takes_priority_over_sniffing(self) -> None:
        """When file_path has a clear extension, binary sniffing is skipped."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            file_path="recording.mp3",
        )
        assert loader._mime_type == "audio/mpeg"
        assert loader._analyzer_id == "prebuilt-audioSearch"


# ---------------------------------------------------------------------------
# Field flattening
# ---------------------------------------------------------------------------


class TestFieldFlattening:
    """Tests for _flatten_fields and related helpers."""

    @pytest.fixture()
    def loader(self) -> AzureAIContentUnderstandingLoader:
        return AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )

    def test_string_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        field = _make_field("string", confidence=0.98, value_string="Contoso")
        result = loader._flatten_single_field(field)
        assert result == {"type": "string", "value": "Contoso", "confidence": 0.98}

    def test_number_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        field = _make_field("number", confidence=0.95, value_number=1250.00)
        result = loader._flatten_single_field(field)
        assert result == {"type": "number", "value": 1250.00, "confidence": 0.95}

    def test_integer_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        field = _make_field("integer", confidence=0.90, value_integer=42)
        result = loader._flatten_single_field(field)
        assert result == {"type": "integer", "value": 42, "confidence": 0.90}

    def test_boolean_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        field = _make_field("boolean", confidence=0.99, value_boolean=True)
        result = loader._flatten_single_field(field)
        assert result == {"type": "boolean", "value": True, "confidence": 0.99}

    def test_date_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        import datetime

        field = _make_field(
            "date", confidence=0.92, value_date=datetime.date(2025, 3, 15)
        )
        result = loader._flatten_single_field(field)
        assert result == {
            "type": "date",
            "value": datetime.date(2025, 3, 15),
            "confidence": 0.92,
        }

    def test_object_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        street = _make_field("string", value_string="123 Main St")
        city = _make_field("string", value_string="Seattle")
        field = _make_field(
            "object",
            confidence=0.89,
            value_object={"street": street, "city": city},
        )
        result = loader._flatten_single_field(field)
        assert result["type"] == "object"
        assert result["value"] == {"street": "123 Main St", "city": "Seattle"}
        assert result["confidence"] == 0.89

    def test_array_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        item1 = _make_field("string", confidence=0.90, value_string="Widget A")
        item2 = _make_field("string", confidence=0.88, value_string="Widget B")
        field = _make_field("array", confidence=0.85, value_array=[item1, item2])
        result = loader._flatten_single_field(field)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"value": "Widget A", "confidence": 0.90}
        assert result[1] == {"value": "Widget B", "confidence": 0.88}

    def test_flatten_fields_dict(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        fields = {
            "name": _make_field("string", confidence=0.98, value_string="Contoso"),
            "total": _make_field("number", confidence=0.95, value_number=1250.0),
        }
        result = loader._flatten_fields(fields)
        assert "name" in result
        assert result["name"]["value"] == "Contoso"
        assert "total" in result
        assert result["total"]["value"] == 1250.0

    def test_field_with_none_confidence(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        field = _make_field("string", confidence=None, value_string="NoConf")
        result = loader._flatten_single_field(field)
        assert result == {"type": "string", "value": "NoConf", "confidence": None}

    def test_array_field_with_none_confidence(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        item = _make_field("string", confidence=None, value_string="Item")
        field = _make_field("array", confidence=None, value_array=[item])
        result = loader._flatten_single_field(field)
        assert isinstance(result, list)
        assert result[0] == {"value": "Item", "confidence": None}


# ---------------------------------------------------------------------------
# Document mapping — markdown mode
# ---------------------------------------------------------------------------


class TestMarkdownMode:
    """Tests for output_mode='markdown'."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_single_document(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(
            markdown="# Invoice\n\nVendor: Contoso",
            fields={
                "vendor": _make_field(
                    "string", confidence=0.98, value_string="Contoso"
                ),
            },
        )
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            analyzer_id="prebuilt-documentSearch",
            url="https://example.com/invoice.pdf",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "# Invoice\n\nVendor: Contoso"
        assert docs[0].metadata["source"] == "https://example.com/invoice.pdf"
        assert docs[0].metadata["kind"] == "document"
        assert docs[0].metadata["output_mode"] == "markdown"
        assert docs[0].metadata["analyzer_id"] == "prebuilt-documentSearch"
        assert "fields" in docs[0].metadata
        assert docs[0].metadata["fields"]["vendor"]["value"] == "Contoso"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_audio_content(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content(
            markdown="Hello, this is a test recording.",
            mime_type="audio/mpeg",
        )
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="prebuilt-audioSearch"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            analyzer_id="prebuilt-audioSearch",
            url="https://example.com/call.mp3",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Hello, this is a test recording."
        assert docs[0].metadata["kind"] == "audioVisual"
        assert docs[0].metadata["start_time_ms"] == 0
        assert docs[0].metadata["end_time_ms"] == 60000

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_none_markdown_yields_empty_string(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown=None)  # type: ignore[arg-type]
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/empty.pdf",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == ""


# ---------------------------------------------------------------------------
# Document mapping — page mode
# ---------------------------------------------------------------------------


class TestPageMode:
    """Tests for output_mode='page'."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_two_pages(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        full_md = "Page one text<!-- PageBreak -->Page two text"
        pages = [
            _make_page(1, 0, 14),  # "Page one text\n"
            _make_page(2, 14, 30),  # "<!-- PageBreak -->Page two text"
        ]
        content = _make_document_content(
            markdown=full_md,
            start_page=1,
            end_page=2,
            pages=pages,
        )
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="page",
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2
        # Page break markers should be stripped
        assert "<!-- PageBreak -->" not in docs[0].page_content
        assert "<!-- PageBreak -->" not in docs[1].page_content

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_page_mode_on_audio_falls_back(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content()
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="prebuilt-audioSearch"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/call.mp3",
            output_mode="page",
        )
        docs = loader.load()

        # Falls back to markdown mode — single document
        assert len(docs) == 1
        assert docs[0].metadata["output_mode"] == "page"


# ---------------------------------------------------------------------------
# Document mapping — segment mode
# ---------------------------------------------------------------------------


class TestSegmentMode:
    """Tests for output_mode='segment'."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segments(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        full_md = "Segment one content. Segment two content."
        seg1 = Mock()
        seg1.span = _make_span(0, 20)
        seg1.category = "intro"
        seg1.fields = None
        seg1.start_time_ms = None
        seg1.end_time_ms = None

        seg2 = Mock()
        seg2.span = _make_span(21, 21)
        seg2.category = "body"
        seg2.fields = None
        seg2.start_time_ms = None
        seg2.end_time_ms = None

        content = _make_document_content(
            markdown=full_md,
            segments=[seg1, seg2],
        )
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["segment_id"] == 0
        assert docs[0].metadata["category"] == "intro"
        assert docs[1].metadata["segment_id"] == 1
        assert docs[1].metadata["category"] == "body"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segment_mode_standalone_content_returned(
        self, mock_cls: MagicMock
    ) -> None:
        """Content without segments array is treated as a standalone segment."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(
            markdown="Standalone page content", segments=None
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Standalone page content"
        assert docs[0].metadata["segment_id"] == 0


# ---------------------------------------------------------------------------
# metadata_selection filtering
# ---------------------------------------------------------------------------


class TestOutputSelection:
    """Tests for metadata_selection field filtering."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_fields_excluded_when_not_selected(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(
            fields={
                "name": _make_field("string", value_string="Test"),
            },
        )
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            metadata_selection=["tables"],  # fields not included
        )
        docs = loader.load()

        assert "fields" not in docs[0].metadata

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_fields_included_by_default(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(
            fields={
                "name": _make_field("string", value_string="Test"),
            },
        )
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        docs = loader.load()

        assert "fields" in docs[0].metadata


# ---------------------------------------------------------------------------
# Async loading
# ---------------------------------------------------------------------------


class TestAsyncLoad:
    """Tests for alazy_load()."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_aload(self, mock_sync_cls: MagicMock) -> None:
        content = _make_document_content()
        mock_result = _make_result([content])

        mock_async_client = MagicMock()
        mock_async_poller = AsyncMock()
        mock_async_poller.result.return_value = mock_result
        mock_async_client.begin_analyze = AsyncMock(return_value=mock_async_poller)
        mock_async_client.close = AsyncMock()

        with patch(
            "langchain_azure_ai.document_loaders.content_understanding"
            ".ContentUnderstandingClient"
        ):
            loader = AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                credential="key",
                url="https://example.com/test.pdf",
            )

        with patch(
            "azure.ai.contentunderstanding.aio.ContentUnderstandingClient",
            return_value=mock_async_client,
        ):
            docs = asyncio.get_event_loop().run_until_complete(loader.aload())

        assert len(docs) == 1
        assert docs[0].page_content == "# Test\n\nContent"
        mock_async_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Span extraction helper
# ---------------------------------------------------------------------------


class TestSpanExtraction:
    """Tests for _extract_text_from_spans."""

    def test_single_span(self) -> None:
        text = "Hello, world!"
        spans = [_make_span(0, 5)]
        result = AzureAIContentUnderstandingLoader._extract_text_from_spans(text, spans)
        assert result == "Hello"

    def test_multiple_spans(self) -> None:
        text = "Hello, world!"
        spans = [_make_span(0, 5), _make_span(7, 5)]
        result = AzureAIContentUnderstandingLoader._extract_text_from_spans(text, spans)
        assert result == "Helloworld"


# ---------------------------------------------------------------------------
# Operation ID and Document ID
# ---------------------------------------------------------------------------


class TestOperationIdAndDocumentId:
    """Tests for operation_id metadata and Document.id."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_operation_id_in_metadata_markdown(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content()
        mock_poller = MagicMock()
        mock_poller.operation_id = "op-abc-123"
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        docs = loader.load()

        assert docs[0].metadata["operation_id"] == "op-abc-123"
        assert docs[0].id == "op-abc-123_0"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_document_id_page_mode(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        full_md = "Page one text<!-- PageBreak -->Page two text"
        pages = [
            _make_page(1, 0, 14),
            _make_page(2, 14, 30),
        ]
        content = _make_document_content(
            markdown=full_md, start_page=1, end_page=2, pages=pages
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = "op-xyz-789"
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="page",
        )
        docs = loader.load()

        assert docs[0].id == "op-xyz-789_0_page_1"
        assert docs[1].id == "op-xyz-789_0_page_2"
        assert docs[0].metadata["operation_id"] == "op-xyz-789"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_document_id_segment_mode(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        full_md = "Segment one content. Segment two content."
        seg1 = Mock()
        seg1.span = _make_span(0, 20)
        seg1.category = "intro"
        seg1.fields = None
        seg1.start_time_ms = None
        seg1.end_time_ms = None

        seg2 = Mock()
        seg2.span = _make_span(21, 21)
        seg2.category = "body"
        seg2.fields = None
        seg2.start_time_ms = None
        seg2.end_time_ms = None

        content = _make_document_content(markdown=full_md, segments=[seg1, seg2])
        mock_poller = MagicMock()
        mock_poller.operation_id = "op-seg-456"
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert docs[0].id == "op-seg-456_0_segment_0"
        assert docs[1].id == "op-seg-456_0_segment_1"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_no_operation_id_sets_no_id(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content()
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        docs = loader.load()

        assert docs[0].id is None
        assert "operation_id" not in docs[0].metadata


# ---------------------------------------------------------------------------
# Content-level category
# ---------------------------------------------------------------------------


class TestContentLevelCategory:
    """Tests for content-level classification result in metadata."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_category_in_metadata(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content()
        content.category = "invoice"
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        docs = loader.load()

        assert docs[0].metadata["category"] == "invoice"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_no_category_when_absent(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content()
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        docs = loader.load()

        assert "category" not in docs[0].metadata


# ---------------------------------------------------------------------------
# Additional field type coverage
# ---------------------------------------------------------------------------


class TestFieldEdgeCases:
    """Tests for field value extraction edge cases."""

    @pytest.fixture()
    def loader(self) -> AzureAIContentUnderstandingLoader:
        return AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )

    def test_time_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        import datetime

        field = _make_field(
            "time", confidence=0.88, value_time=datetime.time(14, 30, 0)
        )
        result = loader._flatten_single_field(field)
        assert result == {
            "type": "time",
            "value": datetime.time(14, 30, 0),
            "confidence": 0.88,
        }

    def test_time_field_none_value(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        field = _make_field("time", confidence=0.50, value_time=None)
        result = loader._flatten_single_field(field)
        assert result == {"type": "time", "value": None, "confidence": 0.50}

    def test_date_field_none_value(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        field = _make_field("date", confidence=0.50, value_date=None)
        result = loader._flatten_single_field(field)
        assert result == {"type": "date", "value": None, "confidence": 0.50}

    def test_json_field(self, loader: AzureAIContentUnderstandingLoader) -> None:
        field = _make_field(
            "json",
            confidence=0.93,
            value_json={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        result = loader._flatten_single_field(field)
        assert result["type"] == "json"
        assert result["value"] == {"nested": {"key": "value"}, "list": [1, 2, 3]}
        assert result["confidence"] == 0.93

    def test_unknown_field_type_returns_none_value(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        field = _make_field("unknown_type", confidence=0.50)
        result = loader._flatten_single_field(field)
        assert result == {"type": "unknown_type", "value": None, "confidence": 0.50}

    def test_nested_object_with_numbers(
        self, loader: AzureAIContentUnderstandingLoader
    ) -> None:
        lat = _make_field("number", value_number=47.6062)
        lon = _make_field("number", value_number=-122.3321)
        field = _make_field(
            "object",
            confidence=0.91,
            value_object={"lat": lat, "lon": lon},
        )
        result = loader._flatten_single_field(field)
        assert result["value"]["lat"] == 47.6062
        assert result["value"]["lon"] == -122.3321

    def test_empty_fields_dict(self, loader: AzureAIContentUnderstandingLoader) -> None:
        result = loader._flatten_fields({})
        assert result == {}


# ---------------------------------------------------------------------------
# Video content metadata (width / height)
# ---------------------------------------------------------------------------


class TestVideoContentMetadata:
    """Tests for audioVisual content with width/height (video)."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_video_metadata_includes_dimensions(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content(
            markdown="Flight simulator clip.",
            mime_type="video/mp4",
            width=1920,
            height=1080,
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="prebuilt-videoSearch"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/video.mp4",
        )
        docs = loader.load()

        assert docs[0].metadata["width"] == 1920
        assert docs[0].metadata["height"] == 1080
        assert docs[0].metadata["kind"] == "audioVisual"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_audio_metadata_omits_dimensions(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content(
            markdown="Audio only.",
            mime_type="audio/mpeg",
            width=None,
            height=None,
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="prebuilt-audioSearch"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/call.mp3",
        )
        docs = loader.load()

        assert "width" not in docs[0].metadata
        assert "height" not in docs[0].metadata


# ---------------------------------------------------------------------------
# Segment mode — audioVisual segments with time ranges
# ---------------------------------------------------------------------------


class TestSegmentModeAudioVisual:
    """Tests for segment mode on audioVisual content (time-based segments)."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_audio_segments_with_time_ranges(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        full_md = "Hello and welcome. Goodbye."
        seg1 = Mock()
        seg1.span = _make_span(0, 18)
        seg1.category = "greeting"
        seg1.fields = None
        seg1.start_time_ms = 0
        seg1.end_time_ms = 5000

        seg2 = Mock()
        seg2.span = _make_span(19, 8)
        seg2.category = "farewell"
        seg2.fields = None
        seg2.start_time_ms = 5000
        seg2.end_time_ms = 10000

        content = _make_audio_visual_content(
            markdown=full_md,
            segments=[seg1, seg2],
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = "op-audio-seg"
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="prebuilt-audioSearch"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/call.mp3",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["start_time_ms"] == 0
        assert docs[0].metadata["end_time_ms"] == 5000
        assert docs[0].metadata["category"] == "greeting"
        assert docs[1].metadata["start_time_ms"] == 5000
        assert docs[1].metadata["end_time_ms"] == 10000
        assert docs[1].metadata["category"] == "farewell"
        assert docs[0].id == "op-audio-seg_0_segment_0"


# ---------------------------------------------------------------------------
# Segment mode — segment-level fields
# ---------------------------------------------------------------------------


class TestSegmentModeFields:
    """Tests for segment-level field extraction."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segment_fields_included(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        seg = Mock()
        seg.span = _make_span(0, 10)
        seg.category = "invoice"
        seg.fields = {
            "amount": _make_field("number", confidence=0.95, value_number=1500.0),
        }
        seg.start_time_ms = None
        seg.end_time_ms = None

        content = _make_document_content(
            markdown="Invoice 1500.00",
            segments=[seg],
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert "fields" in docs[0].metadata
        assert docs[0].metadata["fields"]["amount"]["value"] == 1500.0

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segment_fields_excluded_by_metadata_selection(
        self, mock_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        seg = Mock()
        seg.span = _make_span(0, 10)
        seg.category = "invoice"
        seg.fields = {
            "amount": _make_field("number", confidence=0.95, value_number=1500.0),
        }
        seg.start_time_ms = None
        seg.end_time_ms = None

        content = _make_document_content(
            markdown="Invoice 1500.00",
            segments=[seg],
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="segment",
            metadata_selection=["tables"],
        )
        docs = loader.load()

        assert "fields" not in docs[0].metadata


# ---------------------------------------------------------------------------
# Page mode — no pages fallback
# ---------------------------------------------------------------------------


class TestPageModeEdgeCases:
    """Tests for page mode edge cases."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_page_mode_empty_pages_list_falls_back(self, mock_cls: MagicMock) -> None:
        """Empty pages list should fall back to markdown mode."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(
            markdown="Full document content here.",
            pages=[],  # empty list
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            output_mode="page",
        )
        docs = loader.load()

        # Falls back to markdown — single document
        assert len(docs) == 1
        assert docs[0].page_content == "Full document content here."

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_page_mode_page_with_no_spans_falls_back(self, mock_cls: MagicMock) -> None:
        """A page with no spans should trigger fallback to markdown."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        page = Mock()
        page.page_number = 1
        page.spans = []

        content = _make_document_content(
            markdown="Page content without spans",
            pages=[page],
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            output_mode="page",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Page content without spans"


# ---------------------------------------------------------------------------
# Content range parameter
# ---------------------------------------------------------------------------


class TestContentRange:
    """Tests for content_range parameter passthrough."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_content_range_passed_to_analysis_input(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content()
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            content_range="1-3",
        )
        docs = loader.load()

        # Verify begin_analyze was called with the right input
        call_args = mock_client.begin_analyze.call_args
        inputs = call_args.kwargs.get("inputs") or call_args[1].get("inputs")
        assert inputs[0].content_range == "1-3"
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# Build analysis input
# ---------------------------------------------------------------------------


class TestBuildAnalysisInput:
    """Tests for _build_analysis_input with different input types."""

    def test_url_input_sets_url(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        inp = loader._build_analysis_input()
        assert inp.url == "https://example.com/test.pdf"
        assert inp.data is None

    def test_bytes_input_sets_data(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"fake pdf data",
        )
        inp = loader._build_analysis_input()
        assert inp.data == b"fake pdf data"
        assert inp.url is None

    def test_file_path_reads_file(self, tmp_path: Any) -> None:
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"file content bytes")

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            file_path=str(test_file),
        )
        inp = loader._build_analysis_input()
        assert inp.data == b"file content bytes"
        assert inp.url is None

    def test_content_range_included(self) -> None:
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            content_range="1-5",
        )
        inp = loader._build_analysis_input()
        assert inp.content_range == "1-5"


# ---------------------------------------------------------------------------
# Multiple contents in result
# ---------------------------------------------------------------------------


class TestMultipleContents:
    """Tests for results with more than one content item."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_multiple_content_items(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content1 = _make_document_content(
            markdown="Doc 1 content",
            start_page=1,
            end_page=2,
        )
        content2 = _make_document_content(
            markdown="Doc 2 content",
            start_page=3,
            end_page=4,
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = "op-multi"
        mock_poller.result.return_value = _make_result([content1, content2])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].page_content == "Doc 1 content"
        assert docs[0].id == "op-multi_0"
        assert docs[1].page_content == "Doc 2 content"
        assert docs[1].id == "op-multi_1"


# ---------------------------------------------------------------------------
# Segment mode — markdown fallback when spans absent
# ---------------------------------------------------------------------------


class TestSegmentMarkdownFallback:
    """Tests for segment text extraction fallback to segment.markdown."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segment_uses_markdown_when_no_spans(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        seg = Mock()
        seg.spans = None
        seg.span = None
        seg.markdown = "Segment markdown content"
        seg.category = "summary"
        seg.fields = None
        seg.start_time_ms = None
        seg.end_time_ms = None

        content = _make_document_content(
            markdown="Full document text",
            segments=[seg],
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Segment markdown content"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segment_empty_when_no_spans_or_markdown(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        seg = Mock()
        seg.spans = None
        seg.span = None
        seg.markdown = None
        seg.category = "empty"
        seg.fields = None
        seg.start_time_ms = None
        seg.end_time_ms = None

        content = _make_document_content(
            markdown="Full document text",
            segments=[seg],
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/report.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == ""


# ---------------------------------------------------------------------------
# Async load with operation ID
# ---------------------------------------------------------------------------


class TestAsyncOperationId:
    """Tests for async loading with operation_id."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_aload_captures_operation_id(self, mock_sync_cls: MagicMock) -> None:
        content = _make_document_content()
        mock_result = _make_result([content])

        mock_async_client = MagicMock()
        mock_async_poller = AsyncMock()
        mock_async_poller.operation_id = "async-op-111"
        mock_async_poller.result.return_value = mock_result
        mock_async_client.begin_analyze = AsyncMock(return_value=mock_async_poller)
        mock_async_client.close = AsyncMock()

        with patch(
            "langchain_azure_ai.document_loaders.content_understanding"
            ".ContentUnderstandingClient"
        ):
            loader = AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                credential="key",
                url="https://example.com/test.pdf",
            )

        with patch(
            "azure.ai.contentunderstanding.aio.ContentUnderstandingClient",
            return_value=mock_async_client,
        ):
            docs = asyncio.get_event_loop().run_until_complete(loader.aload())

        assert docs[0].metadata["operation_id"] == "async-op-111"
        assert docs[0].id == "async-op-111_0"


# ---------------------------------------------------------------------------
# AudioVisual content with fields
# ---------------------------------------------------------------------------


class TestAudioVisualFields:
    """Tests for audioVisual content with fields extraction."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_audio_content_with_fields(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content(
            markdown="Transcript here",
            fields={
                "speaker": _make_field("string", confidence=0.90, value_string="John"),
            },
        )
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="prebuilt-audioSearch"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/call.mp3",
        )
        docs = loader.load()

        assert "fields" in docs[0].metadata
        assert docs[0].metadata["fields"]["speaker"]["value"] == "John"


# ---------------------------------------------------------------------------
# Endpoint validation
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    """Tests for endpoint validation in constructor."""

    def test_empty_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="An endpoint is required"):
            AzureAIContentUnderstandingLoader(
                endpoint="",
                credential="key",
                url="https://example.com/test.pdf",
            )

    def test_none_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="An endpoint is required"):
            AzureAIContentUnderstandingLoader(
                endpoint=None,
                credential="key",
                url="https://example.com/test.pdf",
            )

    def test_whitespace_only_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="An endpoint is required"):
            AzureAIContentUnderstandingLoader(
                endpoint="   ",
                credential="key",
                url="https://example.com/test.pdf",
            )


# ---------------------------------------------------------------------------
# model_deployments parameter
# ---------------------------------------------------------------------------


class TestModelDeployments:
    """Tests that model_deployments is forwarded to begin_analyze."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_model_deployments_forwarded(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        deployments = {"gpt-4o": "my-deployment"}
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            model_deployments=deployments,
        )
        loader.load()

        call_kwargs = mock_client.begin_analyze.call_args
        assert call_kwargs.kwargs.get("model_deployments") == deployments

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_model_deployments_default_none(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        loader.load()

        call_kwargs = mock_client.begin_analyze.call_args
        assert call_kwargs.kwargs.get("model_deployments") is None


# ---------------------------------------------------------------------------
# Binary upload path (begin_analyze_binary)
# ---------------------------------------------------------------------------


class TestBinaryUploadPath:
    """Tests that binary inputs use begin_analyze_binary when possible."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_bytes_source_uses_binary_path(self, mock_cls: MagicMock) -> None:
        """bytes_source without model_deployments → begin_analyze_binary."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze_binary.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"fake pdf data",
        )
        loader.load()

        mock_client.begin_analyze_binary.assert_called_once()
        mock_client.begin_analyze.assert_not_called()
        call_kwargs = mock_client.begin_analyze_binary.call_args
        assert call_kwargs.kwargs["binary_input"] == b"fake pdf data"
        assert call_kwargs.kwargs["analyzer_id"] == "prebuilt-documentSearch"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_file_path_uses_binary_path(self, mock_cls: MagicMock) -> None:
        """file_path without model_deployments → begin_analyze_binary."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze_binary.return_value = mock_poller

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test pdf bytes")
            test_file = f.name

        try:
            loader = AzureAIContentUnderstandingLoader(
                endpoint="https://test.ai.azure.com",
                credential="key",
                file_path=test_file,
            )
            loader.load()

            mock_client.begin_analyze_binary.assert_called_once()
            mock_client.begin_analyze.assert_not_called()
            call_kwargs = mock_client.begin_analyze_binary.call_args
            assert call_kwargs.kwargs["binary_input"] == b"test pdf bytes"
        finally:
            import os

            os.unlink(test_file)

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_bytes_with_model_deployments_falls_back(self, mock_cls: MagicMock) -> None:
        """bytes_source WITH model_deployments → begin_analyze (JSON path)."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"fake pdf data",
            model_deployments={"gpt-4o": "my-deployment"},
        )
        loader.load()

        mock_client.begin_analyze.assert_called_once()
        mock_client.begin_analyze_binary.assert_not_called()

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_url_always_uses_json_path(self, mock_cls: MagicMock) -> None:
        """URL input → always uses begin_analyze (JSON path)."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        loader.load()

        mock_client.begin_analyze.assert_called_once()
        mock_client.begin_analyze_binary.assert_not_called()

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_binary_path_forwards_content_range(self, mock_cls: MagicMock) -> None:
        """content_range is forwarded to begin_analyze_binary."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze_binary.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            bytes_source=b"fake pdf data",
            content_range="1-3",
        )
        loader.load()

        call_kwargs = mock_client.begin_analyze_binary.call_args
        assert call_kwargs.kwargs["content_range"] == "1-3"


# ---------------------------------------------------------------------------
# analyze_kwargs pass-through
# ---------------------------------------------------------------------------


class TestAnalyzeKwargs:
    """Tests that analyze_kwargs are forwarded to begin_analyze."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_analyze_kwargs_forwarded(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            analyze_kwargs={"processing_location": "eu"},
        )
        loader.load()

        call_kwargs = mock_client.begin_analyze.call_args
        assert call_kwargs.kwargs.get("processing_location") == "eu"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_analyze_kwargs_default_empty(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        loader.load()

        # No extra kwargs beyond the expected ones
        call_kwargs = mock_client.begin_analyze.call_args
        assert "processing_location" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# content.analyzer_id priority
# ---------------------------------------------------------------------------


class TestContentAnalyzerId:
    """Tests that content-level analyzer_id takes priority."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_content_analyzer_id_takes_priority(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        # Add analyzer_id to the content mock
        content.analyzer_id = "content-level-analyzer"

        result = _make_result([content], analyzer_id="result-level-analyzer")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = result
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            analyzer_id="constructor-analyzer",
        )
        docs = loader.load()

        assert docs[0].metadata["analyzer_id"] == "content-level-analyzer"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_result_analyzer_id_fallback(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        # No analyzer_id on content; fallback to result-level

        result = _make_result([content], analyzer_id="result-level-analyzer")
        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = result
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            analyzer_id="constructor-analyzer",
        )
        docs = loader.load()

        assert docs[0].metadata["analyzer_id"] == "result-level-analyzer"


# ---------------------------------------------------------------------------
# Warnings logging
# ---------------------------------------------------------------------------


class TestWarningsLogging:
    """Tests that result.warnings are logged."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_warnings_are_logged(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        result = _make_result([content])

        warning1 = Mock()
        warning1.message = "Low confidence on page 2"
        warning2 = Mock()
        warning2.message = "Partial extraction"
        result.warnings = [warning1, warning2]

        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = result
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )

        with patch(
            "langchain_azure_ai.document_loaders.content_understanding.logger"
        ) as mock_logger:
            loader.load()
            warning_calls = [
                c
                for c in mock_logger.warning.call_args_list
                if "CU analysis warning" in str(c)
            ]
            assert len(warning_calls) == 2

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_no_warnings_no_log(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_document_content(markdown="Hello")
        result = _make_result([content])
        result.warnings = None

        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = result
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )

        with patch(
            "langchain_azure_ai.document_loaders.content_understanding.logger"
        ) as mock_logger:
            loader.load()
            warning_calls = [
                c
                for c in mock_logger.warning.call_args_list
                if "CU analysis warning" in str(c)
            ]
            assert len(warning_calls) == 0


# ---------------------------------------------------------------------------
# Segment mode — document classification (parent + sub-contents)
# ---------------------------------------------------------------------------


class TestSegmentModeDocClassification:
    """Tests for the real doc classification pattern.

    The service returns a parent content with a ``segments`` array AND
    separate sub-content items with ``category`` set.  The loader should
    use the sub-contents (authoritative) and skip the parent's span-based
    segments to avoid duplicates.

    Based on real output: ``test_doc_classify_output.json``.
    """

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_classification_uses_sub_contents_not_parent_spans(
        self, mock_cls: MagicMock
    ) -> None:
        """Parent with segments + 3 sub-contents → 3 docs from sub-contents."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Parent content (page 1-4) with segments array
        full_md = "Invoice content. Bank statement content. Loan application."
        seg1 = Mock()
        seg1.span = _make_span(0, 16)
        seg1.category = "Invoice"
        seg1.fields = None
        seg1.start_time_ms = None
        seg1.end_time_ms = None

        seg2 = Mock()
        seg2.span = _make_span(17, 23)
        seg2.category = "BankStatement"
        seg2.fields = None
        seg2.start_time_ms = None
        seg2.end_time_ms = None

        seg3 = Mock()
        seg3.span = _make_span(41, 17)
        seg3.category = "LoanApplication"
        seg3.fields = None
        seg3.start_time_ms = None
        seg3.end_time_ms = None

        parent = _make_document_content(
            markdown=full_md,
            start_page=1,
            end_page=4,
            segments=[seg1, seg2, seg3],
            category=None,
        )

        # Sub-contents with category (mirrors real service response)
        sub1 = _make_document_content(
            markdown="Invoice content.",
            start_page=1,
            end_page=1,
            category="Invoice",
        )
        sub2 = _make_document_content(
            markdown="Bank statement content.",
            start_page=2,
            end_page=3,
            category="BankStatement",
        )
        sub3 = _make_document_content(
            markdown="Loan application.",
            start_page=4,
            end_page=4,
            category="LoanApplication",
        )

        mock_poller = MagicMock()
        mock_poller.operation_id = "op-classify-1"
        mock_poller.result.return_value = _make_result(
            [parent, sub1, sub2, sub3],
            analyzer_id="my_doc_classifier",
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/mixed_docs.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        # Should get 3 docs from sub-contents, parent skipped
        assert len(docs) == 3
        assert docs[0].page_content == "Invoice content."
        assert docs[0].metadata["category"] == "Invoice"
        assert docs[0].metadata["start_page_number"] == 1
        assert docs[1].page_content == "Bank statement content."
        assert docs[1].metadata["category"] == "BankStatement"
        assert docs[2].page_content == "Loan application."
        assert docs[2].metadata["category"] == "LoanApplication"
        assert docs[2].metadata["start_page_number"] == 4

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_single_category_sub_content(self, mock_cls: MagicMock) -> None:
        """Single-page doc classified as one category → 1 doc."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        parent = _make_document_content(
            markdown="Invoice for services rendered.",
            segments=[
                Mock(
                    span=_make_span(0, 29),
                    category="Invoice",
                    fields=None,
                    start_time_ms=None,
                    end_time_ms=None,
                )
            ],
            category=None,
        )
        sub = _make_document_content(
            markdown="Invoice for services rendered.",
            category="Invoice",
        )

        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([parent, sub])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/invoice.pdf",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["category"] == "Invoice"


# ---------------------------------------------------------------------------
# Segment mode — standalone content items (video segmenter sub-analyzer)
# ---------------------------------------------------------------------------


class TestSegmentModeStandalone:
    """Tests for segment mode on standalone content items.

    Video segmenters with custom sub-analyzers return each segment as an
    independent content item with markdown, fields, and time ranges — but
    without a ``category`` or ``segments`` array.
    """

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_standalone_av_content_as_segment(self, mock_cls: MagicMock) -> None:
        """Single standalone audioVisual content is returned as a segment doc."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content(
            markdown="# Video: 00:00 => 00:43\nTranscript...",
            mime_type="video/mp4",
            start_time_ms=0,
            end_time_ms=43000,
            width=1080,
            height=608,
            fields={
                "Title": _make_field("string", value_string="Introduction"),
                "Summary": _make_field("string", value_string="Overview of the topic"),
            },
            segments=None,
        )

        mock_poller = MagicMock()
        mock_poller.operation_id = "op-standalone-1"
        mock_poller.result.return_value = _make_result(
            [content], analyzer_id="my_video_segmenter"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/video.mp4",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert "Transcript" in docs[0].page_content
        assert docs[0].metadata["kind"] == "audioVisual"
        assert docs[0].metadata["segment_id"] == 0
        assert docs[0].metadata["start_time_ms"] == 0
        assert docs[0].metadata["end_time_ms"] == 43000
        assert docs[0].metadata["width"] == 1080
        assert docs[0].metadata["fields"]["Title"]["value"] == "Introduction"
        assert docs[0].id == "op-standalone-1_0_segment_0"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_multiple_standalone_av_contents(self, mock_cls: MagicMock) -> None:
        """Multiple standalone content items are each treated as a segment."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        seg1 = _make_audio_visual_content(
            markdown="Segment 1: intro",
            start_time_ms=0,
            end_time_ms=15000,
            segments=None,
        )
        seg2 = _make_audio_visual_content(
            markdown="Segment 2: main topic",
            start_time_ms=15000,
            end_time_ms=40000,
            segments=None,
        )
        seg3 = _make_audio_visual_content(
            markdown="Segment 3: conclusion",
            start_time_ms=40000,
            end_time_ms=60000,
            segments=None,
        )

        mock_poller = MagicMock()
        mock_poller.operation_id = "op-multi-seg"
        mock_poller.result.return_value = _make_result(
            [seg1, seg2, seg3], analyzer_id="my_video_segmenter"
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/video.mp4",
            output_mode="segment",
        )
        docs = loader.load()

        assert len(docs) == 3
        assert docs[0].page_content == "Segment 1: intro"
        assert docs[0].metadata["start_time_ms"] == 0
        assert docs[1].page_content == "Segment 2: main topic"
        assert docs[1].metadata["start_time_ms"] == 15000
        assert docs[2].page_content == "Segment 3: conclusion"
        assert docs[2].metadata["start_time_ms"] == 40000
        # Each standalone item gets segment_id matching its content index
        assert docs[0].metadata["segment_id"] == 0
        assert docs[1].metadata["segment_id"] == 1
        assert docs[2].metadata["segment_id"] == 2

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_standalone_av_fields_with_metadata_selection(
        self, mock_cls: MagicMock
    ) -> None:
        """metadata_selection controls whether fields are included."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        content = _make_audio_visual_content(
            markdown="Video segment",
            fields={
                "Topics": _make_field("string", value_string="AI, ML"),
            },
            segments=None,
        )

        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([content])
        mock_client.begin_analyze.return_value = mock_poller

        # Fields excluded
        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/video.mp4",
            output_mode="segment",
            metadata_selection=["markdown"],
        )
        docs = loader.load()
        assert "fields" not in docs[0].metadata

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_segment_mode_empty_contents_raises(self, mock_cls: MagicMock) -> None:
        """Segment mode still raises ValueError when result has no contents at all."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = _make_result([])
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/video.mp4",
            output_mode="segment",
        )

        with pytest.raises(ValueError, match="no segments were found"):
            loader.load()


# ---------------------------------------------------------------------------
# Empty contents warning
# ---------------------------------------------------------------------------


class TestEmptyContentsWarning:
    """Tests that empty result.contents triggers a warning."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_empty_contents_warns(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        result = _make_result([])

        mock_poller = MagicMock()
        mock_poller.operation_id = None
        mock_poller.result.return_value = result
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )

        with patch(
            "langchain_azure_ai.document_loaders.content_understanding.logger"
        ) as mock_logger:
            docs = loader.load()
            assert docs == []
            warning_calls = [
                c
                for c in mock_logger.warning.call_args_list
                if "no content items" in str(c)
            ]
            assert len(warning_calls) == 1


class TestApiVersion:
    """Tests for the api_version parameter."""

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_api_version_forwarded(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result(
            [_make_document_content(markdown="hello")]
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
            api_version="2024-12-01-preview",
        )
        loader.load()

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_version"] == "2024-12-01-preview"

    @patch(
        "langchain_azure_ai.document_loaders.content_understanding"
        ".ContentUnderstandingClient"
    )
    def test_api_version_omitted_by_default(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_poller = MagicMock()
        mock_poller.result.return_value = _make_result(
            [_make_document_content(markdown="hello")]
        )
        mock_client.begin_analyze.return_value = mock_poller

        loader = AzureAIContentUnderstandingLoader(
            endpoint="https://test.ai.azure.com",
            credential="key",
            url="https://example.com/test.pdf",
        )
        loader.load()

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert "api_version" not in call_kwargs
