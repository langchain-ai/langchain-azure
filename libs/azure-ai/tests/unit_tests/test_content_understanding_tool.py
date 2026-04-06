"""Unit tests for AzureAIContentUnderstandingTool."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_field(
    field_type: str,
    value: Any = None,
    confidence: Optional[float] = 0.95,
) -> Mock:
    """Create a mock ContentField with .value property."""
    field = Mock()
    field.type = field_type
    field.confidence = confidence
    field.value = value
    return field


def _make_content(
    markdown: Optional[str] = "# Invoice\n\nTotal: $100",
    fields: Optional[Dict[str, Mock]] = None,
) -> Mock:
    content = Mock()
    content.markdown = markdown
    content.fields = fields
    return content


def _make_result(contents: list[Mock], warnings: Any = None) -> Mock:
    result = Mock()
    result.contents = contents
    result.warnings = warnings
    return result


def _make_tool(**extra: Any) -> Any:
    """Create an AzureAIContentUnderstandingTool with a mocked client."""
    from langchain_azure_ai.tools.services.content_understanding import (
        AzureAIContentUnderstandingTool,
    )

    with patch(
        "langchain_azure_ai.tools.services.content_understanding.ContentUnderstandingClient"
    ) as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        tool = AzureAIContentUnderstandingTool(
            endpoint="https://test.cognitiveservices.azure.com",
            credential="test-key",
            **extra,
        )
        tool._client = client
        return tool, client


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        tool, _ = _make_tool()
        assert tool.name == "azure_ai_content_understanding"
        assert tool.analyzer_id == "prebuilt-documentSearch"
        assert tool.model_deployments is None

    def test_custom_analyzer(self) -> None:
        tool, _ = _make_tool(analyzer_id="prebuilt-audioSearch")
        assert tool.analyzer_id == "prebuilt-audioSearch"

    def test_custom_model_deployments(self) -> None:
        deployments = {"gpt-4.1": "myGpt41"}
        tool, _ = _make_tool(model_deployments=deployments)
        assert tool.model_deployments == deployments


# ---------------------------------------------------------------------------
# _get_binary_data tests
# ---------------------------------------------------------------------------


class TestGetBinaryData:
    def test_url_returns_none(self) -> None:
        tool, _ = _make_tool()
        assert tool._get_binary_data("https://example.com/doc.pdf", "url") is None

    def test_base64_source(self) -> None:
        tool, _ = _make_tool()
        raw = base64.b64encode(b"fake-pdf-bytes").decode()
        result = tool._get_binary_data(raw, "base64")
        assert result == b"fake-pdf-bytes"

    def test_data_uri_treated_as_base64(self) -> None:
        tool, _ = _make_tool()
        raw = base64.b64encode(b"image-bytes").decode()
        data_uri = f"data:image/png;base64,{raw}"
        result = tool._get_binary_data(data_uri, "url")
        assert result == b"image-bytes"

    def test_path_source(self) -> None:
        tool, _ = _make_tool()
        fake_bytes = b"file-content"
        with patch("builtins.open", mock_open(read_data=fake_bytes)):
            result = tool._get_binary_data("/tmp/test.pdf", "path")
        assert result == fake_bytes


# ---------------------------------------------------------------------------
# _resolve_field_value tests
# ---------------------------------------------------------------------------


class TestResolveFieldValue:
    def test_string_field(self) -> None:
        from langchain_azure_ai.tools.services.content_understanding import (
            AzureAIContentUnderstandingTool,
        )

        field = _make_field("string", value="hello")
        assert AzureAIContentUnderstandingTool._resolve_field_value(field) == "hello"

    def test_number_field(self) -> None:
        from langchain_azure_ai.tools.services.content_understanding import (
            AzureAIContentUnderstandingTool,
        )

        field = _make_field("number", value=42.5)
        assert AzureAIContentUnderstandingTool._resolve_field_value(field) == 42.5

    def test_none_value(self) -> None:
        from langchain_azure_ai.tools.services.content_understanding import (
            AzureAIContentUnderstandingTool,
        )

        field = _make_field("string", value=None)
        assert AzureAIContentUnderstandingTool._resolve_field_value(field) is None

    def test_object_field_recursive(self) -> None:
        from langchain_azure_ai.tools.services.content_understanding import (
            AzureAIContentUnderstandingTool,
        )

        inner = _make_field("number", value=610)
        currency = _make_field("string", value="USD")
        field = _make_field("object", value={"Amount": inner, "CurrencyCode": currency})
        result = AzureAIContentUnderstandingTool._resolve_field_value(field)
        assert result == {"Amount": 610, "CurrencyCode": "USD"}

    def test_array_field_recursive(self) -> None:
        from langchain_azure_ai.tools.services.content_understanding import (
            AzureAIContentUnderstandingTool,
        )

        item1 = _make_field("string", value="item1", confidence=0.9)
        item2 = _make_field("string", value="item2", confidence=0.8)
        field = _make_field("array", value=[item1, item2])
        result = AzureAIContentUnderstandingTool._resolve_field_value(field)
        assert result == [
            {"value": "item1", "confidence": 0.9},
            {"value": "item2", "confidence": 0.8},
        ]


# ---------------------------------------------------------------------------
# _analyze tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_url_uses_begin_analyze(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result([_make_content(markdown="# Hello")])
        client.begin_analyze.return_value = poller

        result = tool._analyze("https://example.com/doc.pdf", "url")

        client.begin_analyze.assert_called_once()
        client.begin_analyze_binary.assert_not_called()
        assert result["contents"][0]["markdown"] == "# Hello"

    def test_path_uses_begin_analyze_binary(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Binary result")]
        )
        client.begin_analyze_binary.return_value = poller

        with patch("builtins.open", mock_open(read_data=b"file-bytes")):
            tool._analyze("/tmp/test.pdf", "path")

        client.begin_analyze_binary.assert_called_once()
        client.begin_analyze.assert_not_called()
        _, kwargs = client.begin_analyze_binary.call_args
        assert kwargs["binary_input"] == b"file-bytes"

    def test_base64_uses_begin_analyze_binary(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Base64 result")]
        )
        client.begin_analyze_binary.return_value = poller

        raw = base64.b64encode(b"pdf-bytes").decode()
        tool._analyze(raw, "base64")

        client.begin_analyze_binary.assert_called_once()
        client.begin_analyze.assert_not_called()

    def test_binary_with_model_deployments_uses_begin_analyze(self) -> None:
        """When model_deployments is set, fall back to begin_analyze even for binary."""
        deployments = {"gpt-4.1": "myDeploy"}
        tool, client = _make_tool(model_deployments=deployments)
        poller = MagicMock()
        poller.result.return_value = _make_result([_make_content()])
        client.begin_analyze.return_value = poller

        with patch("builtins.open", mock_open(read_data=b"file-bytes")):
            tool._analyze("/tmp/test.pdf", "path")

        client.begin_analyze.assert_called_once()
        client.begin_analyze_binary.assert_not_called()
        _, kwargs = client.begin_analyze.call_args
        assert kwargs["model_deployments"] == deployments

    def test_with_fields(self) -> None:
        tool, client = _make_tool()
        fields = {
            "total": _make_field("string", value="$100", confidence=0.98),
            "date": _make_field("string", value="2026-01-01", confidence=0.95),
        }
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Invoice", fields=fields)]
        )
        client.begin_analyze.return_value = poller

        result = tool._analyze("https://example.com/inv.pdf", "url")

        content = result["contents"][0]
        assert content["fields"]["total"]["value"] == "$100"
        assert content["fields"]["total"]["confidence"] == 0.98
        assert content["fields"]["date"]["value"] == "2026-01-01"

    def test_empty_contents(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result([])
        client.begin_analyze.return_value = poller

        result = tool._analyze("https://example.com/empty.pdf", "url")
        assert result["contents"] == []

    def test_no_markdown_no_fields(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown=None, fields=None)]
        )
        client.begin_analyze.return_value = poller

        result = tool._analyze("https://example.com/doc.pdf", "url")
        assert result["contents"] == [{}]

    def test_invalid_source_type_raises(self) -> None:
        tool, _ = _make_tool()
        with pytest.raises(ValueError, match="Invalid source type"):
            tool._analyze("something", "ftp")


# ---------------------------------------------------------------------------
# _format_result tests
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_markdown_only_format(self) -> None:
        tool, _ = _make_tool()
        result = {"contents": [{"markdown": "# Title\n\nBody text"}]}
        formatted = tool._format_result(result)
        assert "# Title" in formatted
        assert "Body text" in formatted

    def test_fields_with_confidence(self) -> None:
        tool, _ = _make_tool()
        result = {
            "contents": [
                {
                    "fields": {
                        "amount": {
                            "type": "string",
                            "value": "$500",
                            "confidence": 0.99,
                        }
                    }
                }
            ]
        }
        formatted = tool._format_result(result)
        assert "amount: $500 (confidence: 0.99)" in formatted

    def test_fields_without_confidence(self) -> None:
        tool, _ = _make_tool()
        result = {
            "contents": [
                {
                    "fields": {
                        "name": {"type": "string", "value": "Alice", "confidence": None}
                    }
                }
            ]
        }
        formatted = tool._format_result(result)
        assert "name: Alice" in formatted
        assert "confidence" not in formatted

    def test_array_field_format(self) -> None:
        tool, _ = _make_tool()
        result = {
            "contents": [
                {
                    "fields": {
                        "LineItems": [
                            {
                                "value": {"Description": "Widget", "Quantity": 2},
                                "confidence": 0.95,
                            },
                            {
                                "value": {"Description": "Gadget", "Quantity": 1},
                                "confidence": 0.88,
                            },
                        ]
                    }
                }
            ]
        }
        formatted = tool._format_result(result)
        assert "LineItems:" in formatted
        assert "Widget" in formatted
        assert "Gadget" in formatted
        assert "confidence: 0.95" in formatted
        assert "confidence: 0.88" in formatted

    def test_empty_contents(self) -> None:
        tool, _ = _make_tool()
        result: Dict[str, Any] = {"contents": []}
        assert tool._format_result(result) == "No content extracted."


# ---------------------------------------------------------------------------
# _run (end-to-end) tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_url(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Extracted text")]
        )
        client.begin_analyze.return_value = poller

        output = tool._run("https://example.com/doc.pdf", source_type="url")
        assert "Extracted text" in output

    def test_run_path(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Path result")]
        )
        client.begin_analyze_binary.return_value = poller

        with patch("builtins.open", mock_open(read_data=b"file-bytes")):
            output = tool._run("/tmp/report.pdf", source_type="path")

        assert "Path result" in output
        client.begin_analyze_binary.assert_called_once()
        client.begin_analyze.assert_not_called()

    def test_run_base64(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Base64 result")]
        )
        client.begin_analyze_binary.return_value = poller

        raw = base64.b64encode(b"pdf-bytes").decode()
        output = tool._run(raw, source_type="base64")

        assert "Base64 result" in output
        client.begin_analyze_binary.assert_called_once()
        client.begin_analyze.assert_not_called()

    def test_run_empty_result(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result([])
        client.begin_analyze.return_value = poller

        output = tool._run("https://example.com/empty.pdf", source_type="url")
        assert output == "No content was extracted from the input."

    def test_run_via_invoke(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Result from invoke")]
        )
        client.begin_analyze.return_value = poller

        output = tool.invoke(
            {"source": "https://example.com/doc.pdf", "source_type": "url"}
        )
        assert "Result from invoke" in output
