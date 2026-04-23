"""Unit tests for AzureOpenAITranscriptionsTool."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

_ENDPOINT = "https://westeurope.api.cognitive.microsoft.com/"
_MODEL = "whisper-1"


def _make_tool(**extra: Any) -> tuple[Any, Any]:
    """Instantiate AzureOpenAITranscriptionsTool with a mocked OpenAI client."""
    from langchain_azure_ai.tools import AzureOpenAITranscriptionsTool

    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        tool = AzureOpenAITranscriptionsTool(
            credential="test-key",
            endpoint=_ENDPOINT,
            model=_MODEL,
            **extra,
        )
        return tool, mock_client


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    """Test tool instantiation and configuration."""

    def test_defaults(self) -> None:
        """Test default values."""
        tool, _ = _make_tool()
        assert tool.name == "azure_openai_transcriptions"
        assert tool.model == _MODEL
        assert tool.endpoint == _ENDPOINT

    def test_client_created_with_string_key(self) -> None:
        """Test OpenAI client initialization with string API key."""
        from langchain_azure_ai.tools import AzureOpenAITranscriptionsTool

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            tool = AzureOpenAITranscriptionsTool(
                credential="my-key",
                endpoint=_ENDPOINT,
                model=_MODEL,
            )
            mock_openai.assert_called_once_with(
                base_url=_ENDPOINT,
                api_key="my-key",
                default_headers={"x-ms-useragent": "langchain-azure-ai"},
            )
            assert tool._client == mock_client

    def test_client_created_with_azure_key_credential(self) -> None:
        """Test OpenAI client initialization with AzureKeyCredential."""
        from azure.core.credentials import AzureKeyCredential

        from langchain_azure_ai.tools import AzureOpenAITranscriptionsTool

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            tool = AzureOpenAITranscriptionsTool(
                credential=AzureKeyCredential("my-key"),
                endpoint=_ENDPOINT,
                model=_MODEL,
            )
            mock_openai.assert_called_once_with(
                base_url=_ENDPOINT,
                api_key="my-key",
                default_headers={"x-ms-useragent": "langchain-azure-ai"},
            )
            assert tool._client == mock_client

    def test_client_created_with_token_credential(self) -> None:
        """Test OpenAI client initialization with TokenCredential."""
        from azure.core.credentials import TokenCredential

        from langchain_azure_ai.tools import AzureOpenAITranscriptionsTool

        mock_credential = MagicMock(spec=TokenCredential)
        mock_token = MagicMock()
        mock_token.token = "bearer-token-value"
        mock_credential.get_token.return_value = mock_token

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            tool = AzureOpenAITranscriptionsTool(
                credential=mock_credential,
                endpoint=_ENDPOINT,
                model=_MODEL,
            )
            mock_openai.assert_called_once_with(
                base_url=_ENDPOINT,
                api_key="bearer-token-value",
                default_headers={"x-ms-useragent": "langchain-azure-ai"},
            )
            mock_credential.get_token.assert_called_once_with(
                "https://cognitiveservices.azure.com/.default"
            )
            assert tool._client == mock_client

    def test_missing_openai_package(self) -> None:
        """Test that ImportError is raised when openai package is not installed."""
        with patch(
            "openai.OpenAI",
            side_effect=ImportError("No module named 'openai'"),
        ):
            from langchain_azure_ai.tools import AzureOpenAITranscriptionsTool

            with pytest.raises(ImportError, match="openai"):
                AzureOpenAITranscriptionsTool(
                    credential="test-key",
                    endpoint=_ENDPOINT,
                    model=_MODEL,
                )


# ---------------------------------------------------------------------------
# Transcription tests
# ---------------------------------------------------------------------------


class TestTranscription:
    """Test audio transcription functionality."""

    def test_transcribe_local_file(self) -> None:
        """Test transcribing a local audio file."""
        tool, mock_client = _make_tool()

        mock_response = MagicMock()
        mock_response.text = "Hello, this is a test audio"
        mock_client.audio.transcriptions.create.return_value = mock_response

        with (
            patch("builtins.open", mock_open(read_data=b"audio data")),
            patch("os.path.isfile", return_value=True),
        ):
            result = tool._run("path/to/audio.wav")

        assert result == "Hello, this is a test audio"
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == _MODEL
        assert "language" not in call_kwargs

    def test_transcribe_with_language(self) -> None:
        """Test transcribing with language specification."""
        tool, mock_client = _make_tool()

        mock_response = MagicMock()
        mock_response.text = "Bonjour, ceci est un test"
        mock_client.audio.transcriptions.create.return_value = mock_response

        with (
            patch("builtins.open", mock_open(read_data=b"audio data")),
            patch("os.path.isfile", return_value=True),
        ):
            result = tool._run("path/to/audio.wav", language="fr")

        assert result == "Bonjour, ceci est un test"
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "fr"

    def test_run_wraps_exception(self) -> None:
        """Test that exceptions are wrapped in RuntimeError."""
        tool, mock_client = _make_tool()

        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        with (
            patch("builtins.open", mock_open(read_data=b"audio data")),
            patch("os.path.isfile", return_value=True),
        ):
            with pytest.raises(RuntimeError, match="Error while running"):
                tool._run("path/to/audio.wav")


# ---------------------------------------------------------------------------
# File source detection and handling tests
# ---------------------------------------------------------------------------


class TestFileSourceDetection:
    """Test detection of local vs. remote audio sources."""

    def test_local_file_detected(self) -> None:
        """Test that local file paths are correctly identified."""
        tool, _ = _make_tool()
        with patch("os.path.isfile", return_value=True):
            assert tool._get_audio_file_path("path/to/audio.wav") == "path/to/audio.wav"

    def test_absolute_local_path_detected(self) -> None:
        """Test that absolute local paths are correctly identified."""
        tool, _ = _make_tool()
        with patch("os.path.isfile", return_value=True):
            assert tool._get_audio_file_path("/tmp/audio.wav") == "/tmp/audio.wav"

    def test_remote_http_url_downloaded(self) -> None:
        """Test that HTTP URLs are downloaded."""
        tool, _ = _make_tool()

        with (
            patch(
                "langchain_azure_ai.tools._openai_tools.detect_file_src_type",
                return_value="remote",
            ),
            patch(
                "langchain_azure_ai.tools._openai_tools.download_audio_from_url"
            ) as mock_download,
        ):
            mock_download.return_value = "/tmp/downloaded_audio.wav"
            result = tool._get_audio_file_path("http://example.com/audio.wav")

        assert result == "/tmp/downloaded_audio.wav"
        mock_download.assert_called_once_with("http://example.com/audio.wav")

    def test_remote_https_url_downloaded(self) -> None:
        """Test that HTTPS URLs are downloaded."""
        tool, _ = _make_tool()

        with (
            patch(
                "langchain_azure_ai.tools._openai_tools.detect_file_src_type",
                return_value="remote",
            ),
            patch(
                "langchain_azure_ai.tools._openai_tools.download_audio_from_url"
            ) as mock_download,
        ):
            mock_download.return_value = "/tmp/downloaded_audio.wav"
            result = tool._get_audio_file_path("https://example.com/audio.wav")

        assert result == "/tmp/downloaded_audio.wav"
        mock_download.assert_called_once_with("https://example.com/audio.wav")

    def test_invalid_path_raises_error(self) -> None:
        """Test that invalid paths raise ValueError."""
        tool, _ = _make_tool()

        with pytest.raises(ValueError, match="Invalid audio path"):
            tool._get_audio_file_path("not:a:valid:path")


# ---------------------------------------------------------------------------
# Input schema tests
# ---------------------------------------------------------------------------


class TestInputSchema:
    """Test the SpeechToTextInput schema."""

    def test_audio_path_required(self) -> None:
        """Test that audio_path is required."""
        from langchain_azure_ai.tools import SpeechToTextInput

        with pytest.raises(ValueError):
            SpeechToTextInput()  # type: ignore[call-arg]

    def test_audio_path_only(self) -> None:
        """Test creating input with only audio_path."""
        from langchain_azure_ai.tools import SpeechToTextInput

        input_schema = SpeechToTextInput(audio_path="path/to/audio.wav")
        assert input_schema.audio_path == "path/to/audio.wav"
        assert input_schema.language is None

    def test_audio_path_with_language(self) -> None:
        """Test creating input with both audio_path and language."""
        from langchain_azure_ai.tools import SpeechToTextInput

        input_schema = SpeechToTextInput(
            audio_path="path/to/audio.wav",
            language="es",
        )
        assert input_schema.audio_path == "path/to/audio.wav"
        assert input_schema.language == "es"


# ---------------------------------------------------------------------------
# Integration-like tests with invoke
# ---------------------------------------------------------------------------


class TestInvoke:
    """Test tool invocation through the invoke method."""

    def test_invoke_with_dict_input(self) -> None:
        """Test invoking tool with dictionary input."""
        tool, mock_client = _make_tool()

        mock_response = MagicMock()
        mock_response.text = "Transcribed text"
        mock_client.audio.transcriptions.create.return_value = mock_response

        with (
            patch("builtins.open", mock_open(read_data=b"audio data")),
            patch("os.path.isfile", return_value=True),
        ):
            result = tool.invoke({"audio_path": "path/to/audio.wav"})

        assert result == "Transcribed text"

    def test_invoke_with_language(self) -> None:
        """Test invoking tool with language parameter."""
        tool, mock_client = _make_tool()

        mock_response = MagicMock()
        mock_response.text = "Texto transcrito"
        mock_client.audio.transcriptions.create.return_value = mock_response

        with (
            patch("builtins.open", mock_open(read_data=b"audio data")),
            patch("os.path.isfile", return_value=True),
        ):
            result = tool.invoke({"audio_path": "path/to/audio.wav", "language": "es"})

        assert result == "Texto transcrito"
