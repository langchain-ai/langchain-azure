"""Tool that queries the Azure AI Services Text to Speech API."""

from __future__ import annotations

import logging
import tempfile
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, model_validator

from langchain_azure_ai._resources import AIServicesService

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    raise ImportError(
        "To use Azure AI Text to Speech tool, please install the "
        "'azure-cognitiveservices-speech' package: "
        "`pip install azure-cognitiveservices-speech` or install the 'tools' "
        "extra: `pip install langchain-azure-ai[tools]`"
    )

logger = logging.getLogger(__name__)


class AzureAITextToSpeechTool(BaseTool, AIServicesService):
    """Tool that converts text to speech using Azure AI Speech service.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-text-to-speech

    Supply ``endpoint`` (the cognitive-services endpoint for your Speech
    resource, e.g. ``"https://eastus.api.cognitive.microsoft.com/"``) and a
    valid ``credential`` (subscription key string or
    :class:`~azure.core.credentials.TokenCredential`). You may also use the
    ``project_endpoint`` parameter (or ``AZURE_AI_PROJECT_ENDPOINT`` env var)
    to resolve the endpoint automatically from an Azure AI Foundry project.

    Environment variables
    ---------------------
    - ``AZURE_AI_INFERENCE_CREDENTIAL`` — subscription key fallback.
    - ``AZURE_AI_INFERENCE_ENDPOINT`` — cognitive-services endpoint URL.
    - ``AZURE_AI_PROJECT_ENDPOINT`` — Azure AI Foundry project endpoint (used
      to resolve the cognitive-services endpoint automatically).
    """

    _speech_config: speechsdk.SpeechConfig = PrivateAttr()

    name: str = "azure_ai_text_to_speech"
    """The name of the tool."""

    description: str = (
        "Converts text to spoken audio using Azure AI Speech service. "
        "Input should be text to synthesize. Returns a local path to a WAV "
        "audio file containing the synthesized speech."
    )
    """The description of the tool."""

    speech_language: str = "en-US"
    """The language of the synthesized speech in BCP-47 format (e.g. ``"en-US"``).
    Defaults to ``"en-US"``."""

    @model_validator(mode="after")
    def initialize_speech_config(self) -> AzureAITextToSpeechTool:
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

    def _text_to_speech(self, text: str) -> str:
        """Synthesize *text* into a WAV file and return the output path."""
        self._speech_config.speech_synthesis_language = self.speech_language
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=None,
        )
        result = speech_synthesizer.speak_text(text)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            stream = speechsdk.AudioDataStream(result)
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".wav", delete=False
            ) as f:
                stream.save_to_wav_file(f.name)
                return f.name

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.debug("Speech synthesis canceled: %s", cancellation_details.reason)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                raise RuntimeError(
                    f"Speech synthesis error: {cancellation_details.error_details}"
                )
            return "Speech synthesis canceled."

        return f"Speech synthesis failed: {result.reason}"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            return self._text_to_speech(query)
        except Exception as e:
            raise RuntimeError(f"Error while running {self.name}: {e}") from e
