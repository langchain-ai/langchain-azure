"""Unit tests for AzureAISpeechToTextTool."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build a minimal azure.cognitiveservices.speech stub so the module can be
# imported without the real SDK installed.
# ---------------------------------------------------------------------------


def _make_speechsdk_stub() -> ModuleType:
    speechsdk = ModuleType("azure.cognitiveservices.speech")

    class _SpeechConfig:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs
            self.speech_recognition_language: str = "en-US"

    class _AudioConfig:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

    class _SpeechRecognizer:
        def __init__(self, **kwargs: Any) -> None:
            self.recognized = MagicMock()
            self.session_stopped = MagicMock()
            self.canceled = MagicMock()

        def start_continuous_recognition_async(self) -> None:
            pass

        def stop_continuous_recognition_async(self) -> None:
            pass

    speechsdk.SpeechConfig = _SpeechConfig  # type: ignore[attr-defined]
    speechsdk.AudioConfig = _AudioConfig  # type: ignore[attr-defined]
    speechsdk.SpeechRecognizer = _SpeechRecognizer  # type: ignore[attr-defined]
    return speechsdk


# Register the stub *before* the tool module is imported so the top-level
# ``import azure.cognitiveservices.speech as speechsdk`` succeeds.
_stub = _make_speechsdk_stub()
sys.modules.setdefault("azure.cognitiveservices", ModuleType("azure.cognitiveservices"))
sys.modules["azure.cognitiveservices.speech"] = _stub

_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(**extra: Any) -> Any:
    """Instantiate AzureAISpeechToTextTool with a mocked SpeechConfig.

    Passing ``endpoint`` explicitly ensures ``AIServicesService.validate_environment``
    does not read ``AZURE_AI_PROJECT_ENDPOINT`` from the environment.
    """
    from langchain_azure_ai.tools.services.speech_to_text import (
        AzureAISpeechToTextTool,
    )

    with patch(
        "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
    ) as mock_sdk:
        mock_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_config
        tool = AzureAISpeechToTextTool(
            credential="test-key",
            endpoint=_ENDPOINT,
            **extra,
        )
        tool._speech_config = mock_config
        return tool, mock_config, mock_sdk


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        tool, _, _ = _make_tool()
        assert tool.name == "azure_ai_speech_to_text"
        assert tool.speech_language == "en-US"
        assert tool.endpoint == _ENDPOINT

    def test_custom_language(self) -> None:
        tool, _, _ = _make_tool(speech_language="pt-BR")
        assert tool.speech_language == "pt-BR"

    def test_speech_config_created_with_string_key(self) -> None:
        from langchain_azure_ai.tools.services.speech_to_text import (
            AzureAISpeechToTextTool,
        )

        with patch(
            "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
        ) as mock_sdk:
            mock_sdk.SpeechConfig.return_value = MagicMock()
            AzureAISpeechToTextTool(credential="my-key", endpoint=_ENDPOINT)
            mock_sdk.SpeechConfig.assert_called_once_with(
                subscription="my-key", endpoint=_ENDPOINT
            )

    def test_speech_config_created_with_azure_key_credential(self) -> None:
        from azure.core.credentials import AzureKeyCredential

        from langchain_azure_ai.tools.services.speech_to_text import (
            AzureAISpeechToTextTool,
        )

        with patch(
            "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
        ) as mock_sdk:
            mock_sdk.SpeechConfig.return_value = MagicMock()
            AzureAISpeechToTextTool(
                credential=AzureKeyCredential("my-key"), endpoint=_ENDPOINT
            )
            mock_sdk.SpeechConfig.assert_called_once_with(
                subscription="my-key", endpoint=_ENDPOINT
            )

    def test_speech_config_created_with_token_credential(self) -> None:
        from azure.core.credentials import TokenCredential

        from langchain_azure_ai.tools.services.speech_to_text import (
            AzureAISpeechToTextTool,
        )

        mock_credential = MagicMock(spec=TokenCredential)
        mock_token = MagicMock()
        mock_token.token = "bearer-token-value"
        mock_credential.get_token.return_value = mock_token

        with patch(
            "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
        ) as mock_sdk:
            mock_sdk.SpeechConfig.return_value = MagicMock()
            AzureAISpeechToTextTool(credential=mock_credential, endpoint=_ENDPOINT)
            mock_sdk.SpeechConfig.assert_called_once_with(
                auth_token="bearer-token-value", endpoint=_ENDPOINT
            )

    def test_endpoint_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_azure_ai.tools.services.speech_to_text import (
            AzureAISpeechToTextTool,
        )

        monkeypatch.setenv("AZURE_AI_INFERENCE_ENDPOINT", _ENDPOINT)
        monkeypatch.setenv("AZURE_AI_INFERENCE_CREDENTIAL", "my-key")
        # Ensure the project-endpoint path is not triggered in developer environments
        # where AZURE_AI_PROJECT_ENDPOINT may be set.
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        with patch(
            "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
        ) as mock_sdk:
            mock_sdk.SpeechConfig.return_value = MagicMock()
            tool = AzureAISpeechToTextTool()
            assert tool.endpoint == _ENDPOINT


# ---------------------------------------------------------------------------
# Source-type detection tests
# ---------------------------------------------------------------------------


class TestDetectSourceType:
    def test_http_url(self) -> None:
        tool, _, _ = _make_tool()
        assert tool._detect_source_type("http://example.com/audio.wav") == "remote"

    def test_https_url(self) -> None:
        tool, _, _ = _make_tool()
        assert tool._detect_source_type("https://example.com/audio.wav") == "remote"

    def test_local_path(self) -> None:
        tool, _, _ = _make_tool()
        assert tool._detect_source_type("/tmp/audio.wav") == "local"

    def test_relative_path(self) -> None:
        tool, _, _ = _make_tool()
        assert tool._detect_source_type("audio.wav") == "local"


# ---------------------------------------------------------------------------
# _speech_to_text and _run tests
# ---------------------------------------------------------------------------


class TestSpeechToText:
    def test_local_file(self) -> None:
        tool, mock_config, _ = _make_tool()

        with (
            patch(
                "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
            ) as mock_sdk,
            patch.object(tool, "_continuous_recognize", return_value="hello world"),
        ):
            mock_sdk.AudioConfig.return_value = MagicMock()
            mock_sdk.SpeechRecognizer.return_value = MagicMock()
            result = tool._speech_to_text("/tmp/test.wav")

        assert result == "hello world"
        mock_sdk.AudioConfig.assert_called_once_with(filename="/tmp/test.wav")
        assert mock_config.speech_recognition_language == "en-US"

    def test_remote_url_downloads_and_transcribes(self) -> None:
        tool, _, _ = _make_tool()

        with (
            patch(
                "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
            ) as mock_sdk,
            patch.object(
                tool, "_download_audio", return_value="/tmp/downloaded.wav"
            ) as mock_dl,
            patch.object(
                tool, "_continuous_recognize", return_value="downloaded audio text"
            ),
        ):
            mock_sdk.AudioConfig.return_value = MagicMock()
            mock_sdk.SpeechRecognizer.return_value = MagicMock()
            result = tool._speech_to_text("https://example.com/audio.wav")

        assert result == "downloaded audio text"
        mock_dl.assert_called_once_with("https://example.com/audio.wav")
        mock_sdk.AudioConfig.assert_called_once_with(filename="/tmp/downloaded.wav")

    def test_run_returns_transcription(self) -> None:
        tool, _, _ = _make_tool()

        with patch.object(tool, "_speech_to_text", return_value="transcribed text"):
            result = tool._run("path/to/audio.wav")

        assert result == "transcribed text"

    def test_run_wraps_exception(self) -> None:
        tool, _, _ = _make_tool()

        with patch.object(tool, "_speech_to_text", side_effect=Exception("SDK error")):
            with pytest.raises(RuntimeError, match="Error while running"):
                tool._run("path/to/audio.wav")

    def test_speech_language_is_set_before_recognition(self) -> None:
        tool, mock_config, _ = _make_tool(speech_language="fr-FR")

        with (
            patch(
                "langchain_azure_ai.tools.services.speech_to_text.speechsdk"
            ) as mock_sdk,
            patch.object(tool, "_continuous_recognize", return_value="bonjour"),
        ):
            mock_sdk.AudioConfig.return_value = MagicMock()
            mock_sdk.SpeechRecognizer.return_value = MagicMock()
            tool._speech_to_text("/tmp/audio.wav")

        assert mock_config.speech_recognition_language == "fr-FR"
