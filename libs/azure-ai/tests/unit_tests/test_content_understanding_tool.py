"""Unit tests for AzureAIContentUnderstandingTool."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_content(
    markdown: Optional[str] = "# Invoice\n\nTotal: $100",
    fields: Optional[Dict[str, Mock]] = None,
) -> Mock:
    content = Mock()
    content.markdown = markdown
    content.fields = fields
    return content


def _make_result(
    contents: list[Mock],
    warnings: Any = None,
    analyzer_id: Optional[str] = None,
) -> Mock:
    result = Mock()
    result.contents = contents
    result.warnings = warnings
    result.analyzer_id = analyzer_id
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
# _analyze tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_url_uses_begin_analyze(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result([_make_content(markdown="# Hello")])
        client.begin_analyze.return_value = poller

        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="mocked output",
        ):
            result = tool._analyze("https://example.com/doc.pdf", "url")

        client.begin_analyze.assert_called_once()
        client.begin_analyze_binary.assert_not_called()
        assert isinstance(result, str)

    def test_path_uses_begin_analyze_binary(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Binary result")]
        )
        client.begin_analyze_binary.return_value = poller

        with patch("builtins.open", mock_open(read_data=b"file-bytes")):
            with patch(
                "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
                return_value="mocked output",
            ):
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
        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="mocked output",
        ):
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
            with patch(
                "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
                return_value="mocked output",
            ):
                tool._analyze("/tmp/test.pdf", "path")

        client.begin_analyze.assert_called_once()
        client.begin_analyze_binary.assert_not_called()
        _, kwargs = client.begin_analyze.call_args
        assert kwargs["model_deployments"] == deployments

    def test_calls_to_llm_input_with_source_metadata(self) -> None:
        """Verify to_llm_input is called with the result and source metadata."""
        tool, client = _make_tool()
        mock_result = _make_result([_make_content(markdown="Invoice")])
        poller = MagicMock()
        poller.result.return_value = mock_result
        client.begin_analyze.return_value = poller

        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="formatted output",
        ) as mock_to_llm:
            result = tool._analyze("https://example.com/inv.pdf", "url")

        mock_to_llm.assert_called_once_with(
            mock_result, metadata={"source": "https://example.com/inv.pdf"}
        )
        assert result == "formatted output"

    def test_empty_contents(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result([])
        client.begin_analyze.return_value = poller

        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="",
        ):
            result = tool._analyze("https://example.com/empty.pdf", "url")
        assert result == ""

    def test_invalid_source_type_raises(self) -> None:
        tool, _ = _make_tool()
        with pytest.raises(ValueError, match="Invalid source type"):
            tool._analyze("something", "ftp")


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

        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="Extracted text",
        ):
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
            with patch(
                "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
                return_value="Path result",
            ):
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
        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="Base64 result",
        ):
            output = tool._run(raw, source_type="base64")

        assert "Base64 result" in output
        client.begin_analyze_binary.assert_called_once()
        client.begin_analyze.assert_not_called()

    def test_run_empty_result(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result([])
        client.begin_analyze.return_value = poller

        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="",
        ):
            output = tool._run("https://example.com/empty.pdf", source_type="url")
        assert output == "No content was extracted from the input."

    def test_run_via_invoke(self) -> None:
        tool, client = _make_tool()
        poller = MagicMock()
        poller.result.return_value = _make_result(
            [_make_content(markdown="Result from invoke")]
        )
        client.begin_analyze.return_value = poller

        with patch(
            "langchain_azure_ai.tools.services.content_understanding.to_llm_input",
            return_value="Result from invoke",
        ):
            output = tool.invoke(
                {"source": "https://example.com/doc.pdf", "source_type": "url"}
            )
        assert "Result from invoke" in output
