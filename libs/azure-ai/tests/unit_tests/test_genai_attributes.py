"""Tests for GenAI attribute mapping."""

import json
import pytest

from langchain_azure_ai.tracing.genai_attributes import (
    GenAIAttributes,
    create_mlflow_compat_usage,
    format_messages,
    format_system_instructions,
    infer_provider,
    infer_server_address_port,
    normalize_token_usage,
    truncate_and_redact,
)


class MockMessage:
    """Mock LangChain message for testing."""
    
    def __init__(self, type_: str, content: str, **kwargs):
        self.type = type_
        self.content = content
        self.additional_kwargs = kwargs.get("additional_kwargs", {})


def test_genai_attributes_constants():
    """Test that GenAI attribute constants are defined."""
    assert GenAIAttributes.OPERATION_NAME == "gen_ai.operation.name"
    assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
    assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"


def test_format_messages_basic():
    """Test basic message formatting."""
    messages = [
        MockMessage("human", "Hello"),
        MockMessage("ai", "Hi there!")
    ]
    
    formatted, truncated, hash_val = format_messages(messages)
    
    parsed = json.loads(formatted)
    assert len(parsed) == 2
    assert parsed[0]["role"] == "human"
    assert parsed[0]["parts"][0]["text"] == "Hello"
    assert parsed[1]["role"] == "ai"
    assert parsed[1]["parts"][0]["text"] == "Hi there!"
    assert truncated is False
    assert hash_val is None


def test_format_messages_with_redaction():
    """Test message formatting with redaction."""
    messages = [MockMessage("human", "Sensitive data")]
    
    formatted, truncated, hash_val = format_messages(messages, redact=True)
    
    parsed = json.loads(formatted)
    assert parsed[0]["parts"][0]["text"] == "[REDACTED]"
    assert hash_val is not None
    assert len(hash_val) == 64  # SHA256 hash


def test_format_messages_with_truncation():
    """Test message formatting with truncation."""
    messages = [MockMessage("human", "A" * 1000)]
    
    formatted, truncated, hash_val = format_messages(messages, max_chars=100)
    
    parsed = json.loads(formatted)
    assert len(parsed[0]["parts"][0]["text"]) <= 104  # 100 + "..."
    assert truncated is True


def test_format_messages_with_role_filter():
    """Test message formatting with role filtering."""
    messages = [
        MockMessage("system", "System prompt"),
        MockMessage("human", "User message"),
        MockMessage("ai", "AI response")
    ]
    
    formatted, _, _ = format_messages(messages, role_filter=["human", "ai"])
    
    parsed = json.loads(formatted)
    assert len(parsed) == 2
    assert all(msg["role"] in ["human", "ai"] for msg in parsed)


def test_format_system_instructions():
    """Test system instructions formatting."""
    messages = [
        MockMessage("system", "You are a helpful assistant"),
        MockMessage("human", "Hello")
    ]
    
    instructions = format_system_instructions(messages)
    assert instructions == "You are a helpful assistant"


def test_format_system_instructions_redacted():
    """Test system instructions with redaction."""
    messages = [MockMessage("system", "Secret instructions")]
    
    instructions = format_system_instructions(messages, redact=True)
    assert instructions == "[REDACTED]"


def test_format_system_instructions_none():
    """Test system instructions when no system messages."""
    messages = [MockMessage("human", "Hello")]
    
    instructions = format_system_instructions(messages)
    assert instructions is None


def test_normalize_token_usage_standard():
    """Test token usage normalization with standard format."""
    usage_data = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
    
    normalized = normalize_token_usage(usage_data)
    
    assert normalized["input_tokens"] == 10
    assert normalized["output_tokens"] == 20
    assert normalized["total_tokens"] == 30


def test_normalize_token_usage_alternate_names():
    """Test token usage normalization with alternate field names."""
    usage_data = {
        "input_tokens": 15,
        "output_tokens": 25
    }
    
    normalized = normalize_token_usage(usage_data)
    
    assert normalized["input_tokens"] == 15
    assert normalized["output_tokens"] == 25
    assert normalized["total_tokens"] == 40  # Computed


def test_normalize_token_usage_empty():
    """Test token usage normalization with empty data."""
    normalized = normalize_token_usage({})
    
    assert normalized["input_tokens"] == 0
    assert normalized["output_tokens"] == 0
    assert normalized["total_tokens"] == 0


def test_infer_provider_azure():
    """Test provider inference for Azure OpenAI."""
    serialized = {
        "id": ["langchain", "chat_models", "AzureChatOpenAI"]
    }
    
    provider = infer_provider(serialized=serialized)
    assert provider == "azure.ai.openai"


def test_infer_provider_openai():
    """Test provider inference for OpenAI."""
    serialized = {
        "id": ["langchain", "chat_models", "ChatOpenAI"]
    }
    
    provider = infer_provider(serialized=serialized)
    assert provider == "openai"


def test_infer_provider_from_url():
    """Test provider inference from base URL."""
    invocation_params = {
        "base_url": "https://api.openai.com/v1"
    }
    
    provider = infer_provider(invocation_params=invocation_params)
    assert provider == "openai"


def test_infer_provider_from_metadata():
    """Test provider inference from metadata."""
    metadata = {"provider": "custom_provider"}
    
    provider = infer_provider(metadata=metadata)
    assert provider == "custom_provider"


def test_infer_server_address_port():
    """Test server address and port inference."""
    invocation_params = {
        "base_url": "https://api.openai.com:443/v1"
    }
    
    address, port = infer_server_address_port(invocation_params=invocation_params)
    
    assert address == "api.openai.com"
    assert port == 443


def test_infer_server_address_port_default():
    """Test server address inference with default port."""
    invocation_params = {
        "base_url": "https://api.example.com/v1"
    }
    
    address, port = infer_server_address_port(invocation_params=invocation_params)
    
    assert address == "api.example.com"
    # Port is None for default HTTPS port


def test_truncate_and_redact_basic():
    """Test basic truncation."""
    value, truncated = truncate_and_redact("Hello world", max_chars=5)
    
    assert value == "Hello..."
    assert truncated is True


def test_truncate_and_redact_no_truncation():
    """Test no truncation when under limit."""
    value, truncated = truncate_and_redact("Hi", max_chars=10)
    
    assert value == "Hi"
    assert truncated is False


def test_truncate_and_redact_redacted():
    """Test redaction."""
    value, truncated = truncate_and_redact("Sensitive", redact=True)
    
    assert value == "[REDACTED]"
    assert truncated is False


def test_create_mlflow_compat_usage():
    """Test MLflow compatibility usage format."""
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30
    }
    
    mlflow_json = create_mlflow_compat_usage(usage)
    
    parsed = json.loads(mlflow_json)
    assert parsed["promptTokens"] == 10
    assert parsed["completionTokens"] == 20
    assert parsed["totalTokens"] == 30
