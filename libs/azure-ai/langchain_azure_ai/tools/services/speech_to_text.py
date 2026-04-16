"""Tool that queries the Azure AI Services Speech to Text API."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.request
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, model_validator

from langchain_azure_ai._resources import AIServicesService

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    raise ImportError(
        "To use Azure AI Speech to Text tool, please install the "
        "'azure-cognitiveservices-speech' package: "
        "`pip install azure-cognitiveservices-speech` or install the 'tools' "
        "extra: `pip install langchain-azure-ai[tools]`"
    )

logger = logging.getLogger(__name__)


class AzureAISpeechToTextTool(BaseTool, AIServicesService):
    """Tool that transcribes audio to text using the Azure AI Speech service.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text

    Supply ``endpoint`` (the cognitive-services endpoint for your Speech
    resource, e.g. ``"https://eastus.api.cognitive.microsoft.com/"``) and a
    valid ``credential`` (subscription key string or
    :class:`~azure.core.credentials.TokenCredential`). You may also use the
    ``project_endpoint`` parameter (or ``AZURE_AI_PROJECT_ENDPOINT`` env var)
    to resolve the endpoint automatically from an Azure AI Foundry project.

    .. tip::

        The endpoint URL for a regional Speech resource follows the pattern
        ``https://<region>.api.cognitive.microsoft.com/``. For example,
        East US → ``https://eastus.api.cognitive.microsoft.com/``.

    Environment variables
    ---------------------
    - ``AZURE_AI_INFERENCE_CREDENTIAL`` — subscription key fallback.
    - ``AZURE_AI_INFERENCE_ENDPOINT`` — cognitive-services endpoint URL.
    - ``AZURE_AI_PROJECT_ENDPOINT`` — Azure AI Foundry project endpoint (used
      to resolve the cognitive-services endpoint automatically).

    Example:
        .. code-block:: python

            from langchain_azure_ai.tools.services.speech_to_text import (
                AzureAISpeechToTextTool,
            )

            tool = AzureAISpeechToTextTool(
                credential="<subscription-key>",
                endpoint="https://eastus.api.cognitive.microsoft.com/",
                speech_language="en-US",
            )
            text = tool.run("path/to/audio.wav")
    """

    _speech_config: speechsdk.SpeechConfig = PrivateAttr()

    name: str = "azure_ai_speech_to_text"
    """The name of the tool."""

    description: str = (
        "Transcribes audio files to text using Azure AI Speech service. "
        "Accepts a path to a local audio file or a URL pointing to an audio file. "
        "Supports a wide range of audio formats including WAV, MP3, OGG, and FLAC. "
        "Use this when you need to convert spoken audio into written text. "
        "Input: path to a local audio file or a URL. Output: transcribed text."
    )
    """The description of the tool."""

    speech_language: str = "en-US"
    """The language of the speech in BCP-47 format (e.g. ``"en-US"``).
    Defaults to ``"en-US"``."""

    @model_validator(mode="after")
    def initialize_speech_config(self) -> AzureAISpeechToTextTool:
        """Initialize the Azure Speech SDK configuration from the resolved endpoint."""
        from azure.core.credentials import AzureKeyCredential

        credential = self.credential
        if isinstance(credential, (str, AzureKeyCredential)):
            key = credential if isinstance(credential, str) else credential.key
            self._speech_config = speechsdk.SpeechConfig(
                subscription=key,
                endpoint=self.endpoint,
            )
        else:
            # TokenCredential — acquire a bearer token for Cognitive Services
            token = credential.get_token(  # type: ignore[union-attr]
                "https://cognitiveservices.azure.com/.default"
            ).token
            self._speech_config = speechsdk.SpeechConfig(
                auth_token=token,
                endpoint=self.endpoint,
            )
        return self

    def _detect_source_type(self, audio_path: str) -> str:
        """Return ``"remote"`` for HTTP/HTTPS URLs, ``"local"`` otherwise."""
        parsed = urlparse(audio_path)
        if parsed.scheme in ("http", "https"):
            return "remote"
        return "local"

    def _download_audio(self, url: str) -> str:
        """Download audio from *url* to a temporary file and return its path."""
        name = url.split("?")[0].rstrip("/").split("/")[-1]
        suffix = ("." + name.rsplit(".", 1)[-1]) if "." in name else ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with urllib.request.urlopen(url) as response:  # noqa: S310
            with open(tmp_path, "wb") as f:
                f.write(response.read())
        return tmp_path

    def _continuous_recognize(self, speech_recognizer: Any) -> str:
        """Run continuous recognition and return the full transcription."""
        done = False
        text = ""

        def stop_cb(evt: Any) -> None:
            """Stop continuous recognition when the session ends or is cancelled."""
            speech_recognizer.stop_continuous_recognition_async()
            nonlocal done
            done = True

        def retrieve_cb(evt: Any) -> None:
            """Accumulate recognised text segments."""
            nonlocal text
            text += evt.result.text

        speech_recognizer.recognized.connect(retrieve_cb)
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        speech_recognizer.start_continuous_recognition_async()
        while not done:
            time.sleep(0.5)

        return text

    def _speech_to_text(self, audio_path: str) -> str:
        """Transcribe *audio_path* (local file or URL) to text."""
        src_type = self._detect_source_type(audio_path)
        if src_type == "local":
            audio_config = speechsdk.AudioConfig(filename=audio_path)
        else:
            tmp_path = self._download_audio(audio_path)
            audio_config = speechsdk.AudioConfig(filename=tmp_path)

        self._speech_config.speech_recognition_language = self.speech_language
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )
        return self._continuous_recognize(speech_recognizer)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            return self._speech_to_text(query)
        except Exception as e:
            raise RuntimeError(f"Error while running {self.name}: {e}") from e
