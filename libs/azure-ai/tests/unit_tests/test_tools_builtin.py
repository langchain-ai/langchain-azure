"""Unit tests for langchain_azure_ai.tools.builtin."""

import warnings

import pytest

from langchain_azure_ai._api.base import ExperimentalWarning
from langchain_azure_ai.tools.builtin import (
    BuiltinTool,
    CodeInterpreterTool,
    ComputerUseTool,
    FileSearchFilters,
    FileSearchTool,
    ImageGenerationInputImageMask,
    ImageGenerationTool,
    McpAllowedTools,
    McpRequireApproval,
    McpTool,
    RankingOptions,
    UserLocation,
    WebSearchFilters,
    WebSearchTool,
)

# ---------------------------------------------------------------------------
# SDK type re-exports
# ---------------------------------------------------------------------------


class TestSDKTypeReExports:
    """Verify that SDK types are re-exported from langchain_azure_ai.tools.builtin."""

    def test_file_search_filters_is_type_alias(self) -> None:
        from openai.types.responses.file_search_tool_param import (
            Filters as _FileSearchFilters,
        )

        assert FileSearchFilters is _FileSearchFilters

    def test_image_generation_input_image_mask_is_typeddict(self) -> None:
        from openai.types.responses.tool_param import (
            ImageGenerationInputImageMask as _ImageGenerationInputImageMask,
        )

        assert ImageGenerationInputImageMask is _ImageGenerationInputImageMask

    def test_mcp_allowed_tools_is_type_alias(self) -> None:
        from openai.types.responses.tool_param import (
            McpAllowedTools as _McpAllowedTools,
        )

        assert McpAllowedTools is _McpAllowedTools

    def test_mcp_require_approval_is_type_alias(self) -> None:
        from openai.types.responses.tool_param import (
            McpRequireApproval as _McpRequireApproval,
        )

        assert McpRequireApproval is _McpRequireApproval

    def test_ranking_options_is_typeddict(self) -> None:
        from openai.types.responses.file_search_tool_param import (
            RankingOptions as _RankingOptions,
        )

        assert RankingOptions is _RankingOptions

    def test_user_location_is_typeddict(self) -> None:
        from openai.types.responses.web_search_tool_param import (
            UserLocation as _UserLocation,
        )

        assert UserLocation is _UserLocation

    def test_web_search_filters_is_typeddict(self) -> None:
        from openai.types.responses.web_search_tool_param import (
            Filters as _WebSearchFilters,
        )

        assert WebSearchFilters is _WebSearchFilters


# ---------------------------------------------------------------------------
# ExperimentalWarning emission
# ---------------------------------------------------------------------------


class TestExperimentalWarnings:
    """Verify that each concrete tool class emits
    ExperimentalWarning on instantiation."""

    @pytest.mark.parametrize(
        "tool_cls, kwargs",
        [
            (CodeInterpreterTool, {}),
            (WebSearchTool, {}),
            (FileSearchTool, {"vector_store_ids": ["vs_1"]}),
            (ImageGenerationTool, {}),
            (ComputerUseTool, {}),
            (McpTool, {"server_label": "s"}),
        ],
    )
    def test_emits_experimental_warning(  # type: ignore[no-untyped-def]
        self, tool_cls, kwargs
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tool_cls(**kwargs)
        experimental_warnings = [
            w for w in caught if issubclass(w.category, ExperimentalWarning)
        ]
        assert (
            experimental_warnings
        ), f"{tool_cls.__name__} did not emit an ExperimentalWarning"

    def test_builtin_tool_base_no_warning(self) -> None:
        """BuiltinTool base class itself is NOT marked experimental."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BuiltinTool(type="custom")
        experimental_warnings = [
            w for w in caught if issubclass(w.category, ExperimentalWarning)
        ]
        assert not experimental_warnings


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

    def test_request_headers_default_empty(self) -> None:
        tool = BuiltinTool(type="custom")
        assert tool.request_headers == {}

    def test_request_headers_not_in_dict_payload(self) -> None:
        """request_headers is an instance attribute, not part of the dict."""
        tool = BuiltinTool(type="custom")
        assert "request_headers" not in dict(tool)
        assert "_request_headers" not in dict(tool)

    def test_subclass_can_extend(self) -> None:
        class MyTool(BuiltinTool):
            def __init__(self, option: str = "default") -> None:
                super().__init__(type="my_tool", option=option)

        tool = MyTool()
        assert tool["type"] == "my_tool"
        assert tool["option"] == "default"

    def test_subclass_can_set_request_headers(self) -> None:
        class SecureTool(BuiltinTool):
            def __init__(self, api_key: str) -> None:
                super().__init__(type="secure_tool")
                self._request_headers = {"X-Api-Key": api_key}

        tool = SecureTool(api_key="my-secret")
        assert tool.request_headers == {"X-Api-Key": "my-secret"}
        # Headers must NOT leak into the dict payload
        assert "X-Api-Key" not in dict(tool)
        assert "_request_headers" not in dict(tool)


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

    def test_with_user_location_dict(self) -> None:
        location = {"type": "approximate", "city": "Seattle", "country": "US"}
        tool = WebSearchTool(user_location=location)  # type: ignore[arg-type]
        assert tool["user_location"] == location

    def test_with_user_location_sdk_type(self) -> None:
        location = UserLocation(type="approximate", city="Berlin", country="DE")
        tool = WebSearchTool(user_location=location)
        assert tool["user_location"]["city"] == "Berlin"

    def test_with_filters_dict(self) -> None:
        filters = {"allowed_domains": ["example.com"]}
        tool = WebSearchTool(filters=filters)  # type: ignore[arg-type]
        assert tool["filters"] == filters

    def test_with_filters_sdk_type(self) -> None:
        filters = WebSearchFilters(allowed_domains=["pubmed.ncbi.nlm.nih.gov"])
        tool = WebSearchTool(filters=filters)
        assert tool["filters"]["allowed_domains"] == ["pubmed.ncbi.nlm.nih.gov"]

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
        tool = FileSearchTool(vector_store_ids=["vs_001"], filters=f)  # type: ignore[arg-type]
        assert tool["filters"] == f

    def test_with_ranking_options_dict(self) -> None:
        ro = {"ranker": "default-2024-11-15", "score_threshold": 0.8}
        tool = FileSearchTool(vector_store_ids=["vs_001"], ranking_options=ro)  # type: ignore[arg-type]
        assert tool["ranking_options"] == ro

    def test_with_ranking_options_sdk_type(self) -> None:
        ro = RankingOptions(ranker="default-2024-11-15", score_threshold=0.9)
        tool = FileSearchTool(vector_store_ids=["vs_001"], ranking_options=ro)
        assert tool["ranking_options"]["score_threshold"] == 0.9

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

    def test_with_input_image_mask_sdk_type(self) -> None:
        mask = ImageGenerationInputImageMask(file_id="file_mask_001")
        tool = ImageGenerationTool(input_image_mask=mask)
        assert tool["input_image_mask"]["file_id"] == "file_mask_001"

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
            "model",
            "action",
            "background",
            "input_fidelity",
            "input_image_mask",
            "quality",
            "size",
            "moderation",
            "output_format",
            "output_compression",
            "partial_images",
        ):
            assert key not in tool

    # ---- model_deployment / request_headers ----------------------------

    def test_request_headers_empty_by_default(self) -> None:
        tool = ImageGenerationTool()
        assert tool.request_headers == {}

    def test_request_headers_with_model_deployment(self) -> None:
        tool = ImageGenerationTool(model_deployment="my-img-deploy")
        assert tool.request_headers == {
            "x-ms-oai-image-generation-deployment": "my-img-deploy"
        }

    def test_model_deployment_not_in_dict_payload(self) -> None:
        """model_deployment is a request header, not part of the tool dict."""
        tool = ImageGenerationTool(model_deployment="my-img-deploy")
        assert "model_deployment" not in dict(tool)
        assert "x-ms-oai-image-generation-deployment" not in dict(tool)

    def test_model_deployment_alongside_model(self) -> None:
        """model (tool payload) and model_deployment (header) are independent."""
        tool = ImageGenerationTool(
            model="gpt-image-1",
            model_deployment="my-gpt-image-1-deploy",
        )
        assert tool["model"] == "gpt-image-1"
        assert tool.request_headers == {
            "x-ms-oai-image-generation-deployment": "my-gpt-image-1-deploy"
        }

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

    def test_with_allowed_tools_list(self) -> None:
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
            headers={"X-Api-Key": "my-api-key"},
        )
        assert tool["headers"] == {"X-Api-Key": "my-api-key"}

    def test_with_require_approval_literal(self) -> None:
        tool = McpTool(
            server_label="srv",
            server_url="https://srv.example.com",
            require_approval="always",
        )
        assert tool["require_approval"] == "always"

    def test_with_require_approval_sdk_filter(self) -> None:
        from openai.types.responses.tool_param import (
            McpRequireApprovalMcpToolApprovalFilter,
            McpRequireApprovalMcpToolApprovalFilterNever,
        )

        approval = McpRequireApprovalMcpToolApprovalFilter(
            never=McpRequireApprovalMcpToolApprovalFilterNever(tool_names=["read_file"])
        )
        tool = McpTool(
            server_label="srv",
            server_url="https://srv.example.com",
            require_approval=approval,
        )
        assert tool["require_approval"]["never"]["tool_names"] == ["read_file"]

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
            "allowed_tools",
            "headers",
            "require_approval",
            "server_description",
            "authorization",
            "connector_id",
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
