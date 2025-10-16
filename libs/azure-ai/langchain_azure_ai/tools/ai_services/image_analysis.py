"""Tool that queries the Azure AI Services Image Analysis API."""

from __future__ import annotations

import base64
import json
import logging
from typing import Annotated, Any, Dict, Literal, Optional

from azure.core.exceptions import HttpResponseError
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ArgsSchema, BaseTool
from langchain_core.utils import pre_init
from pydantic import BaseModel, PrivateAttr, SkipValidation, model_validator

from langchain_azure_ai._resources import AIServicesService

try:
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
except ImportError:
    raise ImportError(
        "To use Azure AI Image Analysis tool, please install the"
        "'azure-ai-vision-imageanalysis' package: "
        "`pip install azure-ai-vision-imageanalysis` or install the 'tools' "
        "extra: `pip install langchain-azure-ai[tools]`"
    )

logger = logging.getLogger(__name__)


class ImageInput(BaseModel):
    """The input document for the Azure AI Image Analysis tool."""

    source_type: Literal["url", "path", "base64"] = "url"
    """The type of the image source, either 'url', 'path', or 'base64'."""

    source: str
    """The image source, either a URL, a local file path, or a base64 string."""


class AzureAIImageAnalysisTool(BaseTool, AIServicesService):
    """Tool that queries the Azure AI Services Image Analysis API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40
    """

    _client: ImageAnalysisClient = PrivateAttr()

    name: str = "azure_ai_image_analysis"

    description: str = (
        "Analyzes images to extract visual insights including object detection, "
        "text recognition (OCR), captions, tags, people detection, and smart crops. "
        "Accepts image file paths or URLs. Use this when you need to understand "
        "image content, extract text from images, identify objects or people, "
        "or generate image descriptions."
    )

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = ImageInput
    """The input args schema for the tool."""

    visual_features: Optional[VisualFeatures] = None

    @pre_init
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the environment is set up correctly."""
        values = super().validate_environment(values)

        try:
            if values["visual_features"] is None:
                values["visual_features"] = [
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS,
                    VisualFeatures.READ,
                    VisualFeatures.SMART_CROPS,
                    VisualFeatures.PEOPLE,
                ]
            else:
                for feature in values["visual_features"]:
                    if not any(item.value == feature for item in VisualFeatures):
                        raise ValueError(
                            f"Invalid visual feature: {feature}. "
                            f"Valid features are: {[f.value for f in VisualFeatures]}"
                        )
        except ImportError:
            raise ImportError(
                "azure-ai-vision-imageanalysis is not installed. "
                "Run `pip install azure-ai-vision-imageanalysis` to install."
            )

        return values

    @model_validator(mode="after")
    def initialize_client(self) -> AzureAIImageAnalysisTool:
        """Initialize the Azure AI Image Analysis client."""
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.core.credentials import AzureKeyCredential

        credential = (
            AzureKeyCredential(self.credential)
            if isinstance(self.credential, str)
            else self.credential
        )

        self._client = ImageAnalysisClient(
            endpoint=self.endpoint,  # type: ignore[arg-type]
            credential=credential,  # type: ignore[arg-type]
            **self.client_kwargs,
        )
        return self

    def _image_analysis(
        self, source: str, source_type: Literal["url", "path", "base64"]
    ) -> Dict:
        """Analyze an image using the Image Analysis client."""
        try:
            if source_type == "base64":
                image_data = base64.b64decode(source)

                result = self._client.analyze(
                    image_data=image_data,
                    visual_features=self.visual_features,  # type: ignore[arg-type]
                )
            elif source_type == "path":
                with open(source, "rb") as f:
                    image_data = f.read()

                result = self._client.analyze(
                    image_data=image_data,
                    visual_features=self.visual_features,  # type: ignore[arg-type]
                )
            elif source_type == "url":
                result = self._client.analyze_from_url(
                    image_url=source,
                    visual_features=self.visual_features,  # type: ignore[arg-type]
                )
            else:
                raise ValueError(f"Invalid image path: {source}")
        except HttpResponseError as e:
            return {
                "status_code": e.status_code,
                "error_code": e.error.code if e.error else None,
                "error_message": e.error.message if e.error else None,
                "error_details": e.error.details if e.error else None,
            }

        res_dict = result.as_dict()

        return res_dict

    def _format_image_analysis_result(self, results: Dict) -> str:
        output = {}

        if "tagsResult" in results:
            output["tags"] = results["tagsResult"]["values"]

        if "objectsResult" in results:
            output["objects"] = results["objectsResult"]["values"]

        if "readResult" in results:
            output["read"] = []
            for line in [block for block in results["readResult"]["blocks"]]:
                output["read"].append(", ".join(text["text"] for text in line["lines"]))

        if "peopleResult" in results:
            output["people"] = results["peopleResult"]["values"]

        if "smartCropsResult" in results:
            output["smartCrops"] = results["smartCropsResult"]["values"]

        if "captionResult" in results:
            output["captions"] = results["captionResult"]["captions"]

        return json.dumps(output, indent=2)

    def _run(
        self,
        source: str,
        source_type: Literal["url", "path", "base64"],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            image_analysis_result = self._image_analysis(source, source_type)
            if not image_analysis_result:
                return "No good image analysis result was found"

            return self._format_image_analysis_result(image_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running {self.name}: {e}")
