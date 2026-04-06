"""Tool that queries the Azure AI Content Understanding API."""

from __future__ import annotations

import base64
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, PrivateAttr, SkipValidation, model_validator

from langchain_azure_ai._resources import AIServicesService

try:
    from azure.ai.contentunderstanding import ContentUnderstandingClient
    from azure.ai.contentunderstanding.models import AnalysisInput
    from azure.core.credentials import AzureKeyCredential
except ImportError as ex:
    raise ImportError(
        "To use Azure AI Content Understanding tool, please install the "
        "'azure-ai-contentunderstanding' package: "
        "`pip install azure-ai-contentunderstanding`"
    ) from ex

logger = logging.getLogger(__name__)


class ContentUnderstandingInput(BaseModel):
    """The input for the Azure AI Content Understanding tool."""

    source_type: Literal["url", "path", "base64"] = "url"
    """The type of the content source: 'url', 'path', or 'base64'."""

    source: str
    """The content source — a URL, local file path, or base64-encoded string."""


class AzureAIContentUnderstandingTool(BaseTool, AIServicesService):
    """Tool that queries the Azure AI Content Understanding API.

    Content Understanding is a multimodal AI service that extracts structured
    content from documents, images, audio, and video files.  It returns
    markdown representations and optionally extracted fields defined by the
    chosen analyzer.

    **Examples:**

    Analyze a document from a URL using the default prebuilt analyzer:

    ```python
    from langchain_azure_ai.tools import AzureAIContentUnderstandingTool

    tool = AzureAIContentUnderstandingTool(
        endpoint="https://[your-service].cognitiveservices.azure.com",
        credential="your-api-key",
    )

    result = tool.invoke(
        {"source": "https://example.com/invoice.pdf", "source_type": "url"}
    )
    print(result)
    ```

    Analyze a local audio file with a different analyzer:

    ```python
    tool = AzureAIContentUnderstandingTool(
        endpoint="https://[your-service].cognitiveservices.azure.com",
        credential="your-api-key",
        analyzer_id="prebuilt-audioSearch",
    )

    result = tool.invoke(
        {"source": "/path/to/recording.wav", "source_type": "path"}
    )
    ```

    Use with custom model deployments:

    ```python
    tool = AzureAIContentUnderstandingTool(
        endpoint="https://[your-service].cognitiveservices.azure.com",
        credential="your-api-key",
        analyzer_id="my-custom-analyzer",
        model_deployments={"gpt-4.1": "myGpt41"},
    )
    ```

    Use with Microsoft Entra ID (AAD) authentication:

    ```python
    from azure.identity import DefaultAzureCredential

    tool = AzureAIContentUnderstandingTool(
        endpoint="https://[your-service].cognitiveservices.azure.com",
        credential=DefaultAzureCredential(),
    )
    ```

    If no credential is provided, ``DefaultAzureCredential()`` is used
    automatically:

    ```python
    tool = AzureAIContentUnderstandingTool(
        endpoint="https://[your-service].cognitiveservices.azure.com",
    )
    ```
    """

    _client: ContentUnderstandingClient = PrivateAttr()

    name: str = "azure_ai_content_understanding"
    """The name of the tool."""

    description: str = (
        "Extracts structured content from documents, images, audio, and video "
        "using Azure AI Content Understanding. Returns markdown representations "
        "and extracted fields. Supports PDFs, Office files, images (JPG, PNG), "
        "audio (WAV, MP3), and video (MP4, MOV). Use prebuilt analyzers like "
        "'prebuilt-documentSearch' for documents, 'prebuilt-imageSearch' for images, "
        "'prebuilt-audioSearch' for audio, or 'prebuilt-videoSearch' for video. "
        "Accepts file paths, URLs, or base64 strings."
    )
    """The description of the tool."""

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = (
        ContentUnderstandingInput
    )
    """The input args schema for the tool."""

    analyzer_id: str = "prebuilt-documentSearch"
    """The analyzer ID to use. Defaults to the prebuilt document search analyzer."""

    model_deployments: Optional[Dict[str, str]] = None
    """Override default mapping of model names to deployment names.
    Example: {"gpt-4.1": "myGpt41", "text-embedding-3-large": "myEmbedding"}."""

    @model_validator(mode="after")
    def initialize_client(self) -> AzureAIContentUnderstandingTool:
        """Initialize the Azure AI Content Understanding client."""
        # Supports both API key (str) and Entra ID (TokenCredential) auth.
        # A string credential is wrapped as AzureKeyCredential; TokenCredential
        # (e.g. DefaultAzureCredential) is passed through directly.
        credential = (
            AzureKeyCredential(self.credential)
            if isinstance(self.credential, str)
            else self.credential
        )

        self._client = ContentUnderstandingClient(
            endpoint=self.endpoint,  # type: ignore[arg-type]
            credential=credential,  # type: ignore[arg-type]
            **self.client_kwargs,  # type: ignore[arg-type]
        )
        return self

    @staticmethod
    def _resolve_field_value(field: Any) -> Any:
        """Extract the plain Python value from a ``ContentField``.

        Uses the SDK's ``.value`` convenience property. For object fields,
        recursively resolves nested values. For array fields, returns a
        list of ``{value, confidence}`` dicts.
        """
        t = field.type
        raw = field.value

        if t == "object" and raw is not None:
            return {
                k: AzureAIContentUnderstandingTool._resolve_field_value(v)
                for k, v in raw.items()
            }
        if t == "array" and raw is not None:
            return [
                {
                    "value": AzureAIContentUnderstandingTool._resolve_field_value(item),
                    "confidence": getattr(item, "confidence", None),
                }
                for item in raw
            ]
        # date/time .value already returns str; all others return native types
        return raw

    def _get_binary_data(self, source: str, source_type: str) -> Optional[bytes]:
        """Return raw bytes for binary upload, or ``None`` for URL inputs."""
        if source_type == "base64" or (
            source_type == "url" and source.startswith("data:")
        ):
            if source.startswith("data:"):
                base64_content = source.split(",", 1)[1]
            else:
                base64_content = source
            return base64.b64decode(base64_content)

        if source_type == "path":
            with open(source, "rb") as f:
                return f.read()

        return None

    def _analyze(self, source: str, source_type: str) -> Dict:
        """Run analysis and return a structured result dict.

        Uses ``begin_analyze_binary`` for file/bytes inputs to avoid the
        ~33% overhead of base64 encoding.  Falls back to ``begin_analyze``
        (JSON body) when a URL is provided or when ``model_deployments``
        is set (only supported on the JSON path).
        """
        binary_data = self._get_binary_data(source, source_type)

        if binary_data is not None and not self.model_deployments:
            poller = self._client.begin_analyze_binary(
                analyzer_id=self.analyzer_id,
                binary_input=binary_data,
            )
        else:
            if binary_data is not None:
                analysis_input = AnalysisInput(data=binary_data)
            elif source_type == "url":
                analysis_input = AnalysisInput(url=source)
            else:
                raise ValueError(f"Invalid source type: {source_type}")

            poller = self._client.begin_analyze(
                analyzer_id=self.analyzer_id,
                inputs=[analysis_input],
                model_deployments=self.model_deployments,
            )

        result = poller.result()

        if isinstance(result.warnings, list):
            for warning in result.warnings:
                logger.warning("CU analysis warning: %s", warning.message)

        res_dict: Dict[str, Any] = {}
        contents_output: List[Dict[str, Any]] = []

        for content in result.contents:
            entry: Dict[str, Any] = {}
            if content.markdown is not None:
                entry["markdown"] = content.markdown
            if content.fields:
                entry["fields"] = self._flatten_fields(content.fields)
            contents_output.append(entry)

        res_dict["contents"] = contents_output
        return res_dict

    def _flatten_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CU ``ContentField`` objects to plain dicts with confidence."""
        result: Dict[str, Any] = {}
        for name, field in fields.items():
            field_type = field.type
            if field_type == "array" and field.value is not None:
                result[name] = [
                    {
                        "value": self._resolve_field_value(item),
                        "confidence": getattr(item, "confidence", None),
                    }
                    for item in field.value
                ]
            else:
                result[name] = {
                    "type": field_type,
                    "value": self._resolve_field_value(field),
                    "confidence": getattr(field, "confidence", None),
                }
        return result

    def _format_result(self, result: Dict) -> str:
        """Format the analysis result into a readable string."""
        sections: List[str] = []
        for i, content in enumerate(result.get("contents", [])):
            if "markdown" in content:
                sections.append(content["markdown"])
            if "fields" in content:
                for name, info in content["fields"].items():
                    if isinstance(info, list):
                        # Array field — list of {value, confidence} dicts
                        items = []
                        for item in info:
                            v = item.get("value", "")
                            c = item.get("confidence")
                            c_str = f" (confidence: {c:.2f})" if c is not None else ""
                            items.append(f"  - {v}{c_str}")
                        sections.append(f"{name}:\n" + "\n".join(items))
                    else:
                        val = info.get("value", "")
                        conf = info.get("confidence")
                        conf_str = (
                            f" (confidence: {conf:.2f})" if conf is not None else ""
                        )
                        sections.append(f"{name}: {val}{conf_str}")

        return "\n".join(sections) if sections else "No content extracted."

    def _run(
        self,
        source: str,
        source_type: str = "url",
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        result = self._analyze(source, source_type=source_type)
        if not result.get("contents"):
            return "No content was extracted from the input."
        return self._format_result(result)
