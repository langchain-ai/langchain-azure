"""Unit tests for AzureAIImageAnalysisTool."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.ai.vision.imageanalysis.models import ImageAnalysisResult, VisualFeatures
from azure.core.exceptions import HttpResponseError
from pydantic import ValidationError

from langchain_azure_ai.tools.services.image_analysis import AzureAIImageAnalysisTool

_BOUNDING_BOX = {"x": 0, "y": 0, "w": 10, "h": 10}
_IMAGE_URL = "https://example.com/image.png"

# Wire-format payload, parsed by the real SDK model so that the tests exercise the
# keys the service returns rather than the keys the tool expects.
_FULL_RESPONSE: dict[str, Any] = {
    "metadata": {"height": 10, "width": 10},
    "modelVersion": "2023-10-01",
    "captionResult": {"confidence": 0.9, "text": "a dog on a beach"},
    "denseCaptionsResult": {
        "values": [
            {"confidence": 0.8, "text": "a dog", "boundingBox": _BOUNDING_BOX},
        ]
    },
    "tagsResult": {"values": [{"name": "dog", "confidence": 0.7}]},
    "objectsResult": {
        "values": [
            {
                "boundingBox": _BOUNDING_BOX,
                "tags": [{"name": "dog", "confidence": 0.6}],
            }
        ]
    },
    "readResult": {
        "blocks": [
            {
                "lines": [
                    {
                        "text": "hello",
                        "boundingPolygon": [{"x": 0, "y": 0}],
                        "words": [],
                    },
                    {
                        "text": "world",
                        "boundingPolygon": [{"x": 0, "y": 0}],
                        "words": [],
                    },
                ]
            }
        ]
    },
    "peopleResult": {"values": [{"boundingBox": _BOUNDING_BOX, "confidence": 0.5}]},
    "smartCropsResult": {
        "values": [{"aspectRatio": 1.0, "boundingBox": _BOUNDING_BOX}]
    },
}


def _make_tool(**extra: Any) -> tuple[AzureAIImageAnalysisTool, MagicMock]:
    """Create an AzureAIImageAnalysisTool whose service client is mocked."""
    tool = AzureAIImageAnalysisTool(
        endpoint="https://test.cognitiveservices.azure.com",
        credential="test-key",
        **extra,
    )
    client = MagicMock()
    tool._client = client
    return tool, client


class TestConstruction:
    def test_defaults_to_all_visual_features(self) -> None:
        tool, _ = _make_tool()
        assert tool.visual_features == [
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.READ,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.PEOPLE,
        ]

    @pytest.mark.parametrize("feature", [VisualFeatures.CAPTION, "caption"])
    def test_accepts_custom_visual_features(self, feature: Any) -> None:
        tool, _ = _make_tool(visual_features=[feature])
        assert tool.visual_features == [VisualFeatures.CAPTION]

    def test_rejects_unknown_visual_feature(self) -> None:
        with pytest.raises(ValidationError, match="Invalid visual feature"):
            _make_tool(visual_features=["unknown"])


class TestRun:
    def test_url_source(self) -> None:
        tool, client = _make_tool(visual_features=[VisualFeatures.TAGS])
        client.analyze_from_url.return_value = ImageAnalysisResult(_FULL_RESPONSE)

        tool.invoke({"source": _IMAGE_URL, "source_type": "url"})

        client.analyze_from_url.assert_called_once_with(
            image_url=_IMAGE_URL, visual_features=[VisualFeatures.TAGS]
        )

    def test_path_source(self, tmp_path: Path) -> None:
        tool, client = _make_tool(visual_features=[VisualFeatures.TAGS])
        client.analyze.return_value = ImageAnalysisResult(_FULL_RESPONSE)
        image = tmp_path / "image.png"
        image.write_bytes(b"image-bytes")

        tool.invoke({"source": str(image), "source_type": "path"})

        client.analyze.assert_called_once_with(
            image_data=b"image-bytes", visual_features=[VisualFeatures.TAGS]
        )

    def test_base64_source(self) -> None:
        tool, client = _make_tool(visual_features=[VisualFeatures.TAGS])
        client.analyze.return_value = ImageAnalysisResult(_FULL_RESPONSE)
        source = base64.b64encode(b"image-bytes").decode()

        tool.invoke({"source": source, "source_type": "base64"})

        client.analyze.assert_called_once_with(
            image_data=b"image-bytes", visual_features=[VisualFeatures.TAGS]
        )

    def test_formats_every_visual_feature(self) -> None:
        tool, client = _make_tool()
        client.analyze_from_url.return_value = ImageAnalysisResult(_FULL_RESPONSE)

        output = json.loads(tool.invoke({"source": _IMAGE_URL, "source_type": "url"}))

        assert output["caption"]["text"] == "a dog on a beach"
        assert output["denseCaptions"][0]["text"] == "a dog"
        assert output["tags"][0]["name"] == "dog"
        assert output["objects"][0]["tags"][0]["name"] == "dog"
        assert output["read"] == ["hello, world"]
        assert output["people"][0]["confidence"] == 0.5
        assert output["smartCrops"][0]["aspectRatio"] == 1.0

    def test_omits_features_missing_from_the_response(self) -> None:
        tool, client = _make_tool(visual_features=[VisualFeatures.TAGS])
        client.analyze_from_url.return_value = ImageAnalysisResult(
            {
                "metadata": _FULL_RESPONSE["metadata"],
                "modelVersion": _FULL_RESPONSE["modelVersion"],
                "tagsResult": _FULL_RESPONSE["tagsResult"],
            }
        )

        output = json.loads(tool.invoke({"source": _IMAGE_URL, "source_type": "url"}))

        assert list(output) == ["tags"]

    def test_service_error_is_raised(self) -> None:
        tool, client = _make_tool()
        error = HttpResponseError(message="Access denied")
        client.analyze_from_url.side_effect = error

        with pytest.raises(RuntimeError, match="Access denied") as exc_info:
            tool.invoke({"source": _IMAGE_URL, "source_type": "url"})

        assert exc_info.value.__cause__ is error
