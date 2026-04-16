"""Unit tests for AzureOpenAIModelImageGenTool."""

import base64
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_data(b64: str) -> Any:
    """Create a mock image data object."""
    data = MagicMock()
    data.b64_json = b64
    return data


def _make_response(*b64_values: str) -> Any:
    """Create a mock OpenAI images.generate response."""
    response = MagicMock()
    response.data = [_make_image_data(v) for v in b64_values]
    return response


def _b64_png() -> str:
    """Return a trivial base64 string to use as fake PNG data."""
    return base64.b64encode(b"fake-png-bytes").decode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_openai_client() -> Generator[tuple[Any, Any, Any], None, None]:
    """Patch openai.OpenAI so no real HTTP calls are made."""
    with (
        patch("langchain_azure_ai.tools.image_gen.os") as mock_os,
        patch("openai.OpenAI") as mock_cls,
    ):
        client_instance = MagicMock()
        mock_cls.return_value = client_instance
        yield mock_cls, client_instance, mock_os


def _make_tool(**extra: Any) -> tuple[Any, Any]:
    """Create an AzureOpenAIModelImageGenTool with a mocked OpenAI client."""
    from langchain_azure_ai.tools import AzureOpenAIModelImageGenTool

    with patch("openai.OpenAI") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance
        tool = AzureOpenAIModelImageGenTool(
            endpoint="https://test.openai.azure.com/openai/v1/",
            credential="test-api-key",
            model="gpt-image-1",
            **extra,
        )
        # Attach the mock for later assertions
        tool._client = client_instance
        return tool, client_instance


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIModelImageGenToolConstruction:
    """Tests for AzureOpenAIModelImageGenTool constructor."""

    def test_basic_construction(self) -> None:
        tool, _ = _make_tool()
        assert tool.name == "image_generation"
        assert tool.model == "gpt-image-1"
        assert tool.output_directory is None

    def test_construction_with_output_directory(self, tmp_path: Any) -> None:
        tool, _ = _make_tool(output_directory=str(tmp_path))
        assert tool.output_directory == str(tmp_path)

    def test_description_is_set(self) -> None:
        tool, _ = _make_tool()
        assert "image" in tool.description.lower()

    def test_args_schema_fields(self) -> None:
        from langchain_azure_ai.tools import ImageGenerationInput

        fields = ImageGenerationInput.model_fields
        assert "prompt" in fields
        assert "n" in fields
        assert "size" in fields
        assert "quality" in fields
        assert "style" in fields


# ---------------------------------------------------------------------------
# _build_generate_kwargs tests
# ---------------------------------------------------------------------------


class TestBuildGenerateKwargs:
    """Tests for _build_generate_kwargs helper."""

    def test_minimal(self) -> None:
        tool, _ = _make_tool()
        kwargs = tool._build_generate_kwargs("A cat", 1, "1024x1024", None, None)
        assert kwargs["model"] == "gpt-image-1"
        assert kwargs["prompt"] == "A cat"
        assert kwargs["n"] == 1
        assert kwargs["size"] == "1024x1024"
        assert kwargs["response_format"] == "b64_json"
        assert "quality" not in kwargs
        assert "style" not in kwargs

    def test_with_quality_and_style(self) -> None:
        tool, _ = _make_tool()
        kwargs = tool._build_generate_kwargs("A cat", 1, None, "hd", "vivid")
        assert kwargs["quality"] == "hd"
        assert kwargs["style"] == "vivid"
        assert "size" not in kwargs


# ---------------------------------------------------------------------------
# _run tests – no output_directory
# ---------------------------------------------------------------------------


class TestRunWithoutOutputDirectory:
    """Tests for _run when output_directory is None."""

    def test_single_image_returns_base64(self) -> None:
        tool, client = _make_tool()
        b64 = _b64_png()
        client.images.generate.return_value = _make_response(b64)

        result = tool._run(prompt="A polar bear")

        assert "base64" in result.lower()
        assert b64 in result

    def test_multiple_images_returns_all(self) -> None:
        tool, client = _make_tool()
        b64_a, b64_b = _b64_png(), base64.b64encode(b"other-bytes").decode()
        client.images.generate.return_value = _make_response(b64_a, b64_b)

        result = tool._run(prompt="A polar bear", n=2)

        assert b64_a in result
        assert b64_b in result

    def test_no_images_returns_message(self) -> None:
        tool, client = _make_tool()
        empty_response = MagicMock()
        empty_response.data = []
        client.images.generate.return_value = empty_response

        result = tool._run(prompt="A polar bear")

        assert "no images" in result.lower()


# ---------------------------------------------------------------------------
# _run tests – with output_directory
# ---------------------------------------------------------------------------


class TestRunWithOutputDirectory:
    """Tests for _run when output_directory is set."""

    def test_saves_image_and_returns_path(self, tmp_path: Any) -> None:
        tool, client = _make_tool(output_directory=str(tmp_path))
        b64 = base64.b64encode(b"\x89PNG fake data").decode()
        client.images.generate.return_value = _make_response(b64)

        result = tool._run(prompt="A polar bear")

        assert "saved to" in result.lower()
        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == b"\x89PNG fake data"

    def test_saves_multiple_images(self, tmp_path: Any) -> None:
        tool, client = _make_tool(output_directory=str(tmp_path))
        b64_a = base64.b64encode(b"data-a").decode()
        b64_b = base64.b64encode(b"data-b").decode()
        client.images.generate.return_value = _make_response(b64_a, b64_b)

        result = tool._run(prompt="Bears", n=2)

        saved_files = sorted(tmp_path.glob("*.png"))
        assert len(saved_files) == 2
        assert "images saved" in result.lower()

    def test_filename_uses_prompt_prefix(self, tmp_path: Any) -> None:
        tool, client = _make_tool(output_directory=str(tmp_path))
        b64 = base64.b64encode(b"data").decode()
        client.images.generate.return_value = _make_response(b64)

        tool._run(prompt="Hello world")

        files = list(tmp_path.glob("*.png"))
        assert len(files) == 1
        assert "Hello_world" in files[0].name


# ---------------------------------------------------------------------------
# Import / export tests
# ---------------------------------------------------------------------------


class TestPublicExport:
    """Tests that AzureOpenAIModelImageGenTool is exported from the tools namespace."""

    def test_importable_from_tools(self) -> None:
        from langchain_azure_ai.tools import AzureOpenAIModelImageGenTool  # noqa: F401

        assert AzureOpenAIModelImageGenTool is not None

    def test_in_all(self) -> None:
        import langchain_azure_ai.tools as tools_module

        assert "AzureOpenAIModelImageGenTool" in tools_module.__all__
