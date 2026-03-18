"""Unit tests for langchain_azure_ai.tools.builtin."""

import pytest

from langchain_azure_ai.tools.builtin import (
    BuiltinTool,
    CodeInterpreterTool,
    ComputerUseTool,
    FileSearchTool,
    ImageGenerationTool,
    McpTool,
    WebSearchTool,
)


# ---------------------------------------------------------------------------
# BuiltinTool base class
# ---------------------------------------------------------------------------


class TestBuiltinTool:
    def test_is_dict_subclass(self) -> None:
        tool = BuiltinTool(type="custom", option="val")
        assert isinstance(tool, dict)

    def test_dict_conversion(self) -> None:
        tool = BuiltinTool(type="custom", option="val")
        assert dict(tool) == {"type": "custom", "option": "val"}

    def test_subclass_can_extend(self) -> None:
        class MyTool(BuiltinTool):
            def __init__(self, option: str = "default") -> None:
                super().__init__(type="my_tool", option=option)

        tool = MyTool()
        assert tool["type"] == "my_tool"
        assert tool["option"] == "default"


# ---------------------------------------------------------------------------
# CodeInterpreterTool
# ---------------------------------------------------------------------------


class TestCodeInterpreterTool:
    def test_defaults(self) -> None:
        tool = CodeInterpreterTool()
        assert tool["type"] == "code_interpreter"
        assert tool["container"] == {"type": "auto"}

    def test_with_file_ids(self) -> None:
        tool = CodeInterpreterTool(file_ids=["file_abc", "file_xyz"])
        assert tool["container"]["file_ids"] == ["file_abc", "file_xyz"]

    def test_with_memory_limit(self) -> None:
        tool = CodeInterpreterTool(memory_limit="4g")
        assert tool["container"]["memory_limit"] == "4g"

    def test_with_all_options(self) -> None:
        policy = {"type": "disabled"}
        tool = CodeInterpreterTool(
            file_ids=["f1"], memory_limit="16g", network_policy=policy
        )
        assert tool["container"]["file_ids"] == ["f1"]
        assert tool["container"]["memory_limit"] == "16g"
        assert tool["container"]["network_policy"] == policy

    def test_none_options_excluded(self) -> None:
        tool = CodeInterpreterTool()
        assert "file_ids" not in tool["container"]
        assert "memory_limit" not in tool["container"]
        assert "network_policy" not in tool["container"]

    def test_is_builtin_tool(self) -> None:
        assert isinstance(CodeInterpreterTool(), BuiltinTool)

    def test_is_dict(self) -> None:
        assert isinstance(CodeInterpreterTool(), dict)


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def test_defaults(self) -> None:
        tool = WebSearchTool()
        assert tool["type"] == "web_search"
        assert "search_context_size" not in tool
        assert "user_location" not in tool
        assert "filters" not in tool

    def test_with_search_context_size(self) -> None:
        tool = WebSearchTool(search_context_size="high")
        assert tool["search_context_size"] == "high"

    def test_with_user_location(self) -> None:
        location = {"type": "approximate", "city": "Seattle", "country": "US"}
        tool = WebSearchTool(user_location=location)
        assert tool["user_location"] == location

    def test_with_filters(self) -> None:
        filters = {"allowed_domains": ["example.com"]}
        tool = WebSearchTool(filters=filters)
        assert tool["filters"] == filters

    def test_with_all_options(self) -> None:
        tool = WebSearchTool(
            search_context_size="low",
            user_location={"type": "approximate", "country": "DE"},
            filters={"allowed_domains": ["bbc.com"]},
        )
        assert tool["search_context_size"] == "low"
        assert tool["user_location"]["country"] == "DE"
        assert tool["filters"]["allowed_domains"] == ["bbc.com"]

    def test_is_builtin_tool(self) -> None:
        assert isinstance(WebSearchTool(), BuiltinTool)


# ---------------------------------------------------------------------------
# FileSearchTool
# ---------------------------------------------------------------------------


class TestFileSearchTool:
    def test_required_vector_store_ids(self) -> None:
        tool = FileSearchTool(vector_store_ids=["vs_001"])
        assert tool["type"] == "file_search"
        assert tool["vector_store_ids"] == ["vs_001"]

    def test_multiple_vector_store_ids(self) -> None:
        tool = FileSearchTool(vector_store_ids=["vs_001", "vs_002"])
        assert len(tool["vector_store_ids"]) == 2

    def test_with_max_num_results(self) -> None:
        tool = FileSearchTool(vector_store_ids=["vs_001"], max_num_results=20)
        assert tool["max_num_results"] == 20

    def test_with_filters(self) -> None:
        f = {"type": "eq", "key": "category", "value": "science"}
        tool = FileSearchTool(vector_store_ids=["vs_001"], filters=f)
        assert tool["filters"] == f

    def test_with_ranking_options(self) -> None:
        ro = {"ranker": "default-2024-11-15", "score_threshold": 0.8}
        tool = FileSearchTool(vector_store_ids=["vs_001"], ranking_options=ro)
        assert tool["ranking_options"] == ro

    def test_none_options_excluded(self) -> None:
        tool = FileSearchTool(vector_store_ids=["vs_001"])
        assert "max_num_results" not in tool
        assert "filters" not in tool
        assert "ranking_options" not in tool

    def test_is_builtin_tool(self) -> None:
        assert isinstance(FileSearchTool(vector_store_ids=["vs_001"]), BuiltinTool)


# ---------------------------------------------------------------------------
# ImageGenerationTool
# ---------------------------------------------------------------------------


class TestImageGenerationTool:
    def test_defaults(self) -> None:
        tool = ImageGenerationTool()
        assert tool["type"] == "image_generation"
        # Only type key should be present
        assert set(tool.keys()) == {"type"}

    def test_with_model(self) -> None:
        tool = ImageGenerationTool(model="gpt-image-1")
        assert tool["model"] == "gpt-image-1"

    def test_with_quality_and_size(self) -> None:
        tool = ImageGenerationTool(quality="high", size="1024x1024")
        assert tool["quality"] == "high"
        assert tool["size"] == "1024x1024"

    def test_with_all_options(self) -> None:
        tool = ImageGenerationTool(
            model="gpt-image-1",
            action="generate",
            background="opaque",
            moderation="low",
            output_format="webp",
            output_compression=80,
            quality="medium",
            size="1536x1024",
            partial_images=1,
        )
        assert tool["action"] == "generate"
        assert tool["background"] == "opaque"
        assert tool["output_format"] == "webp"
        assert tool["output_compression"] == 80
        assert tool["partial_images"] == 1

    def test_none_options_excluded(self) -> None:
        tool = ImageGenerationTool()
        for key in (
            "model", "action", "background", "quality", "size",
            "moderation", "output_format", "output_compression", "partial_images",
        ):
            assert key not in tool

    def test_is_builtin_tool(self) -> None:
        assert isinstance(ImageGenerationTool(), BuiltinTool)


# ---------------------------------------------------------------------------
# ComputerUseTool
# ---------------------------------------------------------------------------


class TestComputerUseTool:
    def test_type(self) -> None:
        tool = ComputerUseTool()
        assert tool["type"] == "computer_use_preview"

    def test_only_type_key(self) -> None:
        tool = ComputerUseTool()
        assert set(tool.keys()) == {"type"}

    def test_is_builtin_tool(self) -> None:
        assert isinstance(ComputerUseTool(), BuiltinTool)

    def test_is_dict(self) -> None:
        assert isinstance(ComputerUseTool(), dict)


# ---------------------------------------------------------------------------
# McpTool
# ---------------------------------------------------------------------------


class TestMcpTool:
    def test_required_server_label(self) -> None:
        tool = McpTool(server_label="my_server", server_url="https://example.com")
        assert tool["type"] == "mcp"
        assert tool["server_label"] == "my_server"
        assert tool["server_url"] == "https://example.com"

    def test_with_connector_id(self) -> None:
        tool = McpTool(server_label="gmail", connector_id="connector_gmail")
        assert tool["connector_id"] == "connector_gmail"
        assert "server_url" not in tool

    def test_with_allowed_tools(self) -> None:
        tool = McpTool(
            server_label="srv",
            server_url="https://srv.example.com",
            allowed_tools=["search", "read"],
        )
        assert tool["allowed_tools"] == ["search", "read"]

    def test_with_headers(self) -> None:
        tool = McpTool(
            server_label="srv",
            server_url="https://srv.example.com",
            headers={"Authorization": "Bearer token"},
        )
        assert tool["headers"] == {"Authorization": "Bearer token"}

    def test_with_require_approval(self) -> None:
        tool = McpTool(
            server_label="srv",
            server_url="https://srv.example.com",
            require_approval="always",
        )
        assert tool["require_approval"] == "always"

    def test_with_all_options(self) -> None:
        tool = McpTool(
            server_label="srv",
            server_url="https://srv.example.com",
            allowed_tools=["tool1"],
            headers={"X-Key": "val"},
            require_approval="never",
            server_description="My MCP server",
            authorization="oauth_token",
        )
        assert tool["allowed_tools"] == ["tool1"]
        assert tool["headers"] == {"X-Key": "val"}
        assert tool["require_approval"] == "never"
        assert tool["server_description"] == "My MCP server"
        assert tool["authorization"] == "oauth_token"

    def test_none_options_excluded(self) -> None:
        tool = McpTool(server_label="srv", server_url="https://srv.example.com")
        for key in (
            "allowed_tools", "headers", "require_approval",
            "server_description", "authorization", "connector_id",
        ):
            assert key not in tool

    def test_is_builtin_tool(self) -> None:
        assert isinstance(
            McpTool(server_label="s", server_url="https://s.com"), BuiltinTool
        )


# ---------------------------------------------------------------------------
# convert_to_openai_tool compatibility
# ---------------------------------------------------------------------------


class TestConvertToOpenAIToolCompatibility:
    """Verify all builtin tools pass through convert_to_openai_tool unchanged."""

    def test_code_interpreter(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool = CodeInterpreterTool(file_ids=["f1"])
        result = convert_to_openai_tool(tool)
        assert result == dict(tool)

    def test_web_search(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool = WebSearchTool(search_context_size="medium")
        result = convert_to_openai_tool(tool)
        assert result == dict(tool)

    def test_file_search(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool = FileSearchTool(vector_store_ids=["vs_1"])
        result = convert_to_openai_tool(tool)
        assert result == dict(tool)

    def test_image_generation(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool = ImageGenerationTool(quality="high")
        result = convert_to_openai_tool(tool)
        assert result == dict(tool)

    def test_computer_use(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool = ComputerUseTool()
        result = convert_to_openai_tool(tool)
        assert result == dict(tool)

    def test_mcp(self) -> None:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool = McpTool(server_label="s", server_url="https://s.com")
        result = convert_to_openai_tool(tool)
        assert result == dict(tool)
