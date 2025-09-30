"""Tool that queries the Azure AI Services Image Analysis API."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.exceptions import HttpResponseError
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import pre_init
from pydantic import PrivateAttr, model_validator

from langchain_azure_ai._resources import AIServicesService
from langchain_azure_ai.utils.utils import detect_file_src_type

logger = logging.getLogger(__name__)


class AzureAIImageAnalysisTool(BaseTool, AIServicesService):
    """Tool that queries the Azure AI Services Image Analysis API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40
    """

    _client: ImageAnalysisClient = PrivateAttr()

    name: str = "azure_ai_image_analysis"

    description: str = (
        "A wrapper around Azure AI Services Image Analysis. "
        "Useful for when you need to analyze images. "
        "Input should be a url to an image."
    )

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

    def _image_analysis(self, image_path: str) -> Dict:
        image_src_type = detect_file_src_type(image_path)
        print(f"Image source type detected: {image_src_type}")

        try:
            if image_src_type == "local":
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = self._client.analyze(
                    image_data=image_data,
                    visual_features=self.visual_features,  # type: ignore[arg-type]
                )
            elif image_src_type == "remote":
                result = self._client.analyze_from_url(
                    image_url=image_path,
                    visual_features=self.visual_features,  # type: ignore[arg-type]
                )
            else:
                raise ValueError(f"Invalid image path: {image_path}")
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
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # try:
        print(f"Running {self.name} with query: {query}")

        image_analysis_result = self._image_analysis(query)
        if not image_analysis_result:
            return "No good image analysis result was found"

        return self._format_image_analysis_result(image_analysis_result)
        # except Exception as e:
        #    raise RuntimeError(f"Error while running {self.name}: {e}")
