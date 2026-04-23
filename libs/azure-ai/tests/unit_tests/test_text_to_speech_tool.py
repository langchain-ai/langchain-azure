"""Unit tests for AzureAITextToSpeechTool."""

from __future__ import annotations

import os
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
            self.speech_synthesis_language: str = "en-US"

    class _SpeechSynthesizer:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

        def speak_text(self, text: str) -> Any:
            return MagicMock()

    class _AudioDataStream:
        def __init__(self, result: Any) -> None:
            self.result = result

        def save_to_wav_file(self, path: str) -> None:
            with open(path, "wb") as f:
                f.write(b"RIFF")

    class _ResultReason:
        SynthesizingAudioCompleted = "SynthesizingAudioCompleted"
        Canceled = "Canceled"

    class _CancellationReason:
        Error = "Error"
        EndOfStream = "EndOfStream"

    speechsdk.SpeechConfig = _SpeechConfig  # type: ignore[attr-defined]
    speechsdk.SpeechSynthesizer = _SpeechSynthesizer  # type: ignore[attr-defined]
    speechsdk.AudioDataStream = _AudioDataStream  # type: ignore[attr-defined]
    speechsdk.ResultReason = _ResultReason  # type: ignore[attr-defined]
    speechsdk.CancellationReason = _CancellationReason  # type: ignore[attr-defined]
    return speechsdk


_stub = _make_speechsdk_stub()
sys.modules.setdefault("azure.cognitiveservices", ModuleType("azure.cognitiveservices"))
sys.modules["azure.cognitiveservices.speech"] = _stub

_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"


def _make_tool(**extra: Any) -> Any:
    """Instantiate AzureAITextToSpeechTool with a mocked SpeechConfig."""
    from langchain_azure_ai.tools.services.text_to_speech import AzureAITextToSpeechTool

    with patch(
        "langchain_azure_ai.tools.services.text_to_speech.speechsdk"
    ) as mock_sdk:
        mock_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_config
        tool = AzureAITextToSpeechTool(
            credential="test-key",
            endpoint=_ENDPOINT,
            **extra,
        )
        tool._speech_config = mock_config
        return tool, mock_config, mock_sdk


class TestConstruction:
    def test_defaults(self) -> None:
        tool, _, _ = _make_tool()
        assert tool.name == "azure_ai_text_to_speech"
        assert tool.speech_language == "en-US"
        assert tool.endpoint == _ENDPOINT

    def test_custom_language(self) -> None:
        tool, _, _ = _make_tool(speech_language="es-ES")
        assert tool.speech_language == "es-ES"

    def test_speech_config_created_with_string_key(self) -> None:
        from langchain_azure_ai.tools.services.text_to_speech import (
            AzureAITextToSpeechTool,
        )

        with patch(
            "langchain_azure_ai.tools.services.text_to_speech.speechsdk"
        ) as mock_sdk:
            mock_sdk.SpeechConfig.return_value = MagicMock()
            AzureAITextToSpeechTool(credential="my-key", endpoint=_ENDPOINT)
            mock_sdk.SpeechConfig.assert_called_once_with(
                subscription="my-key", endpoint=_ENDPOINT
            )


class TestTextToSpeech:
    def test_text_to_speech_success_returns_wav_path(self) -> None:
        tool, mock_config, _ = _make_tool()

        mock_result = MagicMock()
        mock_result.reason = "SynthesizingAudioCompleted"

        with patch(
            "langchain_azure_ai.tools.services.text_to_speech.speechsdk"
        ) as mock_sdk:
            mock_stream = MagicMock()
            mock_sdk.ResultReason.SynthesizingAudioCompleted = (
                "SynthesizingAudioCompleted"
            )
            mock_sdk.ResultReason.Canceled = "Canceled"
            mock_sdk.AudioDataStream.return_value = mock_stream
            mock_synthesizer = MagicMock()
            mock_synthesizer.speak_text.return_value = mock_result
            mock_sdk.SpeechSynthesizer.return_value = mock_synthesizer

            output_path = tool._text_to_speech("hello")

        assert output_path.endswith(".wav")
        assert os.path.exists(output_path)
        mock_stream.save_to_wav_file.assert_called_once_with(output_path)
        assert mock_config.speech_synthesis_language == "en-US"
        os.remove(output_path)

    def test_text_to_speech_cancel_with_error_raises(self) -> None:
        tool, _, _ = _make_tool()

        cancellation = MagicMock()
        cancellation.reason = "Error"
        cancellation.error_details = "service unavailable"

        mock_result = MagicMock()
        mock_result.reason = "Canceled"
        mock_result.cancellation_details = cancellation

        with patch(
            "langchain_azure_ai.tools.services.text_to_speech.speechsdk"
        ) as mock_sdk:
            mock_sdk.ResultReason.SynthesizingAudioCompleted = (
                "SynthesizingAudioCompleted"
            )
            mock_sdk.ResultReason.Canceled = "Canceled"
            mock_sdk.CancellationReason.Error = "Error"
            mock_synthesizer = MagicMock()
            mock_synthesizer.speak_text.return_value = mock_result
            mock_sdk.SpeechSynthesizer.return_value = mock_synthesizer

            with pytest.raises(RuntimeError, match="Speech synthesis error"):
                tool._text_to_speech("hello")

    def test_text_to_speech_cancel_no_error_returns_message(self) -> None:
        tool, _, _ = _make_tool()

        cancellation = MagicMock()
        cancellation.reason = "EndOfStream"
        cancellation.error_details = ""

        mock_result = MagicMock()
        mock_result.reason = "Canceled"
        mock_result.cancellation_details = cancellation

        with patch(
            "langchain_azure_ai.tools.services.text_to_speech.speechsdk"
        ) as mock_sdk:
            mock_sdk.ResultReason.SynthesizingAudioCompleted = (
                "SynthesizingAudioCompleted"
            )
            mock_sdk.ResultReason.Canceled = "Canceled"
            mock_sdk.CancellationReason.Error = "Error"
            mock_synthesizer = MagicMock()
            mock_synthesizer.speak_text.return_value = mock_result
            mock_sdk.SpeechSynthesizer.return_value = mock_synthesizer

            msg = tool._text_to_speech("hello")

        assert msg == "Speech synthesis canceled."

    def test_run_wraps_exception(self) -> None:
        tool, _, _ = _make_tool()

        with patch.object(tool, "_text_to_speech", side_effect=Exception("SDK error")):
            with pytest.raises(RuntimeError, match="Error while running"):
                tool._run("hello")
