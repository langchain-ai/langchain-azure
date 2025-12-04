# Azure AI OpenTelemetry Tracer - Autolog Implementation

## Overview

This implementation provides an MLflow-like autolog pattern for Azure AI OpenTelemetry tracing with LangChain/LangGraph. After calling `autolog()`, all LangChain operations are automatically traced without needing to pass callbacks explicitly.

## Quick Start

```python
from langchain_azure_ai.tracing import AzureAIOpenTelemetryTracer

# Configure once at startup
AzureAIOpenTelemetryTracer.set_app_insights(
    "InstrumentationKey=...;IngestionEndpoint=..."
)

AzureAIOpenTelemetryTracer.set_config({
    "provider_name": "azure.ai.openai",
    "redact_messages": False,
    "patch_mode": "monkey"  # Enable auto-registration
})

# Enable automatic tracing
AzureAIOpenTelemetryTracer.autolog()

# All LangChain operations are now automatically traced!
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("Tell me about {topic}") | ChatOpenAI()
result = chain.invoke({"topic": "Python"})  # Automatically traced!
```

## Features Implemented

### Phase 1: Core Infrastructure ✅
- **Static API Methods**: `set_app_insights()`, `set_config()`, `autolog()`, `is_active()`, `shutdown()`, etc.
- **Auto-Registration**: Monkey patching of `Runnable.invoke/ainvoke` for automatic tracer injection
- **Configuration Management**: 20+ configuration options with environment variable support
- **Thread Safety**: Lock-based synchronization for all shared state

### Phase 2: GenAI Attributes & Token Aggregation ✅
- **GenAI Semantic Conventions**: 40+ attribute constants covering all required fields
- **Message Formatting**: Structured message format with redaction and truncation support
- **Token Usage**: Real-time tracking and automatic aggregation to root spans
- **Provider Inference**: Automatic detection of Azure OpenAI, OpenAI, Anthropic, etc.
- **MLflow Compatibility**: Optional MLflow-compatible attributes

### Phase 3: Post-Processing ✅
- **RunTree Enrichment**: Walk LangSmith RunTree to enrich spans
- **Operation Normalization**: Map LangChain operations to GenAI spec names
- **Token Aggregation**: Compute totals across nested LLM calls
- **Synthetic Spans**: Optional creation of spans for missing data

## Configuration Options

```python
{
    # Provider
    "provider_name": None,  # e.g., "azure.ai.openai"
    "tracer_name": "azure_ai_genai_tracer",
    
    # Features
    "emit_mlflow_compat": True,
    "aggregate_usage": True,
    "enable_post_processor": True,
    "normalize_operation_names": True,
    
    # Privacy
    "redact_messages": False,
    "redact_tool_arguments": False,
    "redact_tool_results": False,
    "max_message_characters": None,
    
    # Tool metadata
    "include_tool_definitions": False,
    "log_stream_chunks": False,
    
    # Performance
    "sampling_rate": 1.0,  # 0.0-1.0
    
    # Span hierarchy
    "prefer_last_chat_parent": True,
    "honor_external_parent": True,
    "enable_span_links": True,
    
    # Auto-registration mode
    "patch_mode": "callback",  # "callback" | "hybrid" | "monkey"
    
    # Session tracking
    "thread_id_attribute": "gen_ai.conversation.id",
    
    # Post-processing
    "post_process_root_tag": "langchain.azure.ai",
    "store_run_tree_snapshot": False,
}
```

## Environment Variables

Configuration can be overridden via environment variables:

```bash
export AZURE_GENAI_REDACT_MESSAGES=true
export AZURE_GENAI_SAMPLING_RATE=0.5
export AZURE_GENAI_PROVIDER_NAME=azure.ai.openai
export AZURE_GENAI_PATCH_MODE=monkey
export APPLICATION_INSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
```

## API Reference

### Static Methods

#### `set_app_insights(connection_string: str)`
Set the Application Insights connection string.

#### `set_config(config: Dict[str, Any])`
Update global configuration. Validates configuration before applying.

#### `get_config() -> Dict[str, Any]`
Get current configuration as a dictionary.

#### `autolog(**overrides)`
Enable automatic tracing. Accepts configuration overrides.

**Important**: Set `patch_mode="monkey"` for true auto-registration without callbacks.

#### `is_active() -> bool`
Check if autolog is currently active.

#### `add_tags(tags: Dict[str, Any])`
Add global tags to all spans.

#### `update_redaction_rules(**flags)`
Update redaction configuration dynamically.

#### `force_flush()`
Force immediate flush of all pending telemetry.

#### `shutdown()`
Disable autolog, flush data, and cleanup resources.

#### `get_tracer_instance() -> Optional[AzureAIOpenTelemetryTracer]`
Get the active tracer instance (for explicit use if needed).

## GenAI Semantic Conventions

The tracer automatically captures all required GenAI attributes:

### Request Attributes
- `gen_ai.request.model`
- `gen_ai.request.temperature`
- `gen_ai.request.max_tokens`
- `gen_ai.request.top_p`, `top_k`
- `gen_ai.request.frequency_penalty`, `presence_penalty`
- `gen_ai.request.stop_sequences`

### Messages
- `gen_ai.input.messages` - Structured JSON format
- `gen_ai.system_instructions` - System prompts
- `gen_ai.output.messages` - AI responses

### Tool Attributes
- `gen_ai.tool.name`, `gen_ai.tool.type`
- `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`
- `gen_ai.tool.call.result`

### Usage/Tokens
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.usage.total_tokens`
- Aggregated totals: `gen_ai.usage.*.total`

### Response Metadata
- `gen_ai.response.id`
- `gen_ai.response.model`
- `gen_ai.response.finish_reasons`

### Agent Metadata
- `gen_ai.agent.name`, `gen_ai.agent.id`
- `gen_ai.conversation.id` - From thread_id/session_id

## Token Aggregation

Token usage is automatically tracked and aggregated:

```python
# Individual LLM call
span.set_attribute("gen_ai.usage.input_tokens", 10)
span.set_attribute("gen_ai.usage.output_tokens", 20)

# Root agent span gets aggregated totals
root_span.set_attribute("gen_ai.usage.input_tokens.total", 50)
root_span.set_attribute("gen_ai.usage.output_tokens.total", 80)
root_span.set_attribute("gen_ai.usage.total_tokens.total", 130)
```

## Redaction and Privacy

Control what gets recorded:

```python
AzureAIOpenTelemetryTracer.set_config({
    "redact_messages": True,  # Redact message content
    "redact_tool_arguments": True,  # Redact tool inputs
    "redact_tool_results": True,  # Redact tool outputs
    "max_message_characters": 1000,  # Truncate long messages
})
```

When redacted, hashes are included:
- `gen_ai.input.messages.hash`
- `gen_ai.output.messages.hash`

## Post-Processing

Post-processing enriches spans after execution completes:

```python
from langchain_azure_ai.tracing.post_processor import create_post_processor

processor = create_post_processor({
    "enable_post_processor": True,
    "aggregate_usage": True,
    "emit_mlflow_compat": True,
    "normalize_operation_names": True
})

# Automatically triggered after root agent completes
# Or manually: result = processor.process(run_tree, span_registry)
```

## Backward Compatibility

The old API still works:

```python
# Old way (still supported)
tracer = AzureAIOpenTelemetryTracer(connection_string="...")
result = chain.invoke(input, config={"callbacks": [tracer]})

# New way (with autolog)
AzureAIOpenTelemetryTracer.autolog()
result = chain.invoke(input)  # No callbacks needed!
```

## Architecture

```
langchain_azure_ai/
├── tracing/
│   ├── __init__.py              # Main exports
│   ├── tracer.py                # Enhanced tracer with static API
│   ├── config.py                # Configuration management
│   ├── callback_manager.py      # Auto-registration via monkey patching
│   ├── genai_attributes.py      # Attribute mapping utilities
│   ├── token_aggregator.py      # Token usage tracking
│   └── post_processor.py        # RunTree enrichment
└── callbacks/tracers/
    └── inference_tracing.py     # Original tracer (backward compat)
```

## Testing

Comprehensive test coverage:

```bash
# Run all tracing tests
pytest tests/unit_tests/test_tracing_config.py
pytest tests/unit_tests/test_static_api.py
pytest tests/unit_tests/test_genai_attributes.py
pytest tests/unit_tests/test_token_aggregation.py
pytest tests/unit_tests/test_post_processor.py
```

## Performance

- **Minimal overhead**: Monkey patching only affects tracer injection
- **Thread-safe**: All operations use appropriate locks
- **Sampling support**: Control overhead with `sampling_rate`
- **Lazy evaluation**: Attributes only computed when needed

## Troubleshooting

### Autolog not working
- Ensure `patch_mode="monkey"` or `"hybrid"`
- Check that `is_active()` returns `True`
- Verify no errors in logs

### Missing spans
- Enable post-processor: `enable_post_processor=True`
- Use `emit_missing_spans=True` for synthetic spans

### Token totals incorrect
- Enable aggregation: `aggregate_usage=True`
- Check that LLMs return token usage in responses

## Example: Complete Application

```python
import os
from langchain_azure_ai.tracing import AzureAIOpenTelemetryTracer
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Configure tracing at startup
def setup_tracing():
    AzureAIOpenTelemetryTracer.set_app_insights(
        os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    )
    
    AzureAIOpenTelemetryTracer.set_config({
        "provider_name": "azure.ai.openai",
        "redact_messages": False,
        "aggregate_usage": True,
        "emit_mlflow_compat": True,
        "patch_mode": "monkey"
    })
    
    AzureAIOpenTelemetryTracer.add_tags({
        "environment": "production",
        "application": "my-app",
        "version": "1.0.0"
    })
    
    AzureAIOpenTelemetryTracer.autolog()
    print("✅ Automatic tracing enabled!")

# Main application
def main():
    setup_tracing()
    
    # Create chain
    llm = AzureChatOpenAI(
        deployment_name="gpt-4",
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. {input}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Use chain - automatically traced!
    result = chain.invoke({"input": "Tell me a joke"})
    print(result)
    
    # Cleanup on shutdown
    AzureAIOpenTelemetryTracer.shutdown()

if __name__ == "__main__":
    main()
```

## Implementation Stats

- **Total Lines**: ~2145 lines across all modules
- **Test Coverage**: Comprehensive unit tests for all functionality
- **Modules**: 6 core modules (config, tracer, callback_manager, genai_attributes, token_aggregator, post_processor)
- **Test Files**: 5 test files with 170+ test cases
- **Configuration Options**: 20+ configurable parameters
- **GenAI Attributes**: 40+ semantic convention attributes

## License

Same as langchain-azure parent project.
