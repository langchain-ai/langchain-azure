"""Tool for generating images using OpenAI-compatible image generation endpoints."""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, PrivateAttr, SkipValidation, model_validator

from langchain_azure_ai._resources import ModelInferenceService

logger = logging.getLogger(__name__)

_MAX_PROMPT_PREFIX_LENGTH = 40


class ImageGenerationInput(BaseModel):
    """Input schema for the image generation tool."""

    prompt: str
    """The text prompt describing the image to generate."""

    n: int = 1
    """The number of images to generate."""

    size: Optional[str] = "1024x1024"
    """The size of the generated image (e.g. '1024x1024', '1024x1792', '1792x1024')."""

    quality: Optional[str] = None
    """The quality of the image. Use 'hd' for higher quality (model-dependent)."""

    style: Optional[str] = None
    """The style of the image, e.g. 'vivid' or 'natural' (model-dependent)."""


class OpenAIModelImageGenTool(BaseTool, ModelInferenceService):
    """Tool that generates images using an OpenAI-compatible image generation API.

    This tool connects to model deployments in Azure AI Foundry or Azure OpenAI that
    expose an OpenAI-compatible ``/images/generations`` endpoint, such as
    ``gpt-image-1`` or ``dall-e-3``.

    Example:
        .. code-block:: python

            from langchain_azure_ai.tools import OpenAIModelImageGenTool

            tool = OpenAIModelImageGenTool(
                endpoint="https://<resource>.openai.azure.com/openai/v1/",
                credential="<api-key>",
                model="gpt-image-1",
            )
            result = tool.invoke({"prompt": "A cute baby polar bear"})
    """

    _client: Any = PrivateAttr()

    name: str = "image_generation"
    """The name of the tool."""

    description: str = (
        "Generates images from text descriptions using an image generation model. "
        "Accepts a prompt describing the desired image and optional parameters such "
        "as the number of images, size, quality, and style. "
        "Returns file paths of the saved images when an output directory is "
        "configured, or base64-encoded PNG data otherwise."
    )
    """The description of the tool."""

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = (
        ImageGenerationInput
    )
    """The input args schema for the tool."""

    model: str
    """The deployment name of the image generation model to use."""

    output_directory: Optional[str] = None
    """Directory where generated images will be saved. If ``None``, the base64-encoded
    image data is returned directly."""

    @model_validator(mode="after")
    def initialize_client(self) -> OpenAIModelImageGenTool:
        """Initialize the OpenAI client for image generation."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "To use the ImageGenModelTool, please install the 'openai' package: "
                "`pip install openai` or install the 'langchain-openai' package: "
                "`pip install langchain-openai`"
            ) from exc

        credential = self.credential
        if hasattr(credential, "key"):
            # AzureKeyCredential
            api_key = credential.key  # type: ignore[union-attr]
        elif isinstance(credential, str):
            api_key = credential
        else:
            # TokenCredential – obtain a bearer token for the Azure OpenAI scope
            token = credential.get_token(  # type: ignore[union-attr]
                "https://cognitiveservices.azure.com/.default"
            )
            api_key = token.token

        self._client = OpenAI(
            base_url=self.endpoint,
            api_key=api_key,
        )
        return self

    def _build_generate_kwargs(
        self,
        prompt: str,
        n: int,
        size: Optional[str],
        quality: Optional[str],
        style: Optional[str],
    ) -> Dict[str, Any]:
        """Build keyword arguments for the images.generate call."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }
        if size is not None:
            kwargs["size"] = size
        if quality is not None:
            kwargs["quality"] = quality
        if style is not None:
            kwargs["style"] = style
        return kwargs

    def _save_images(self, b64_data_list: List[str], prompt: str) -> List[str]:
        """Decode and save images to *output_directory*.

        Returns a list of absolute file paths.
        """
        import base64

        assert self.output_directory is not None
        os.makedirs(self.output_directory, exist_ok=True)
        paths: List[str] = []
        safe_prefix = "".join(
            c if c.isalnum() else "_" for c in prompt[:_MAX_PROMPT_PREFIX_LENGTH]
        )
        for idx, b64_data in enumerate(b64_data_list):
            filename = f"{safe_prefix}_{idx}.png"
            file_path = os.path.join(self.output_directory, filename)
            image_bytes = base64.b64decode(b64_data)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            paths.append(file_path)
            logger.debug("Saved generated image to %s", file_path)
        return paths

    def _run(
        self,
        prompt: str,
        n: int = 1,
        size: Optional[str] = "1024x1024",
        quality: Optional[str] = None,
        style: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate images and return file paths or base64 data."""
        kwargs = self._build_generate_kwargs(prompt, n, size, quality, style)
        response = self._client.images.generate(**kwargs)

        b64_data_list: List[str] = []
        for item in response.data:
            if item.b64_json:
                b64_data_list.append(item.b64_json)

        if not b64_data_list:
            return "No images were generated."

        if self.output_directory is not None:
            paths = self._save_images(b64_data_list, prompt)
            if len(paths) == 1:
                return f"Image saved to: {paths[0]}"
            return "Images saved to:\n" + "\n".join(paths)

        if len(b64_data_list) == 1:
            return f"Generated image (base64 PNG): {b64_data_list[0]}"
        return "Generated images (base64 PNG):\n" + "\n".join(
            f"Image {i}: {data}" for i, data in enumerate(b64_data_list, 1)
        )
