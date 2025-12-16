# Phase 1: Core Infrastructure Implementation Plan

## Overview
This document outlines the implementation plan for Phase 1 of the comprehensive autolog refactoring.

## Architecture

### New Module Structure
```
langchain_azure_ai/
├── tracing/
│   ├── __init__.py                 # Export main tracer class
│   ├── tracer.py                   # Main AzureAIOpenTelemetryTracer with static API
│   ├── config.py                   # Configuration management and defaults
│   ├── callback_manager.py         # Auto-registration via context variables
│   ├── genai_attributes.py         # GenAI semantic conventions mapping
│   ├── token_aggregator.py         # Real-time token usage aggregation
│   ├── post_processor.py           # RunTree post-processing
│   └── monkey_patch.py             # Optional monkey patching support
└── callbacks/
    └── tracers/
        └── inference_tracing.py    # Backward compatibility wrapper (deprecated)
```

### Key Components

#### 1. Configuration Management (`config.py`)
```python
DEFAULT_CONFIG = {
    "provider_name": None,
    "emit_mlflow_compat": True,
    "aggregate_usage": True,
    "enable_post_processor": True,
    "normalize_operation_names": True,
    "redact_messages": False,
    "redact_tool_arguments": False,
    "redact_tool_results": False,
    "include_tool_definitions": False,
    "sampling_rate": 1.0,
    "prefer_last_chat_parent": True,
    "honor_external_parent": True,
    "patch_mode": "callback",  # callback | hybrid | monkey
    "log_stream_chunks": False,
    "max_message_characters": None,
    "thread_id_attribute": "gen_ai.conversation.id",
    "enable_span_links": True,
    "post_process_root_tag": "langchain.azure.ai",
    "store_run_tree_snapshot": False,
    "tracer_name": "azure_ai_genai_tracer",
}
```

#### 2. Auto-Registration Strategy

Since LangChain doesn't provide a global callback registry API, we'll use a hybrid approach:

**Option A: Context Variables + Wrapper** (Preferred for Phase 1)
- Use Python contextvars to store the active tracer
- Wrap key LangChain entry points (Runnable.invoke, etc.) to inject tracer from context
- This is less invasive than full monkey patching

**Option B: Callback Manager Hook** (If available)
- Check if newer LangChain versions support callback injection hooks
- Use official API if available

**Option C: Monkey Patching** (Fallback for Phase 4)
- Full method replacement for maximum compatibility
- Most invasive but guaranteed to work

#### 3. Static API Methods

```python
class AzureAIOpenTelemetryTracer:
    # Class-level state
    _GLOBAL_CONFIG: Dict[str, Any] = {}
    _APP_INSIGHTS_CONNECTION_STRING: Optional[str] = None
    _ACTIVE: bool = False
    _GLOBAL_TRACER_INSTANCE: Optional[AzureAIOpenTelemetryTracer] = None
    _SPAN_REGISTRY: Dict[str, Any] = {}
    _LOCK: Lock = Lock()
    
    @classmethod
    def set_app_insights(cls, connection_string: str) -> None:
        """Set Application Insights connection string."""
        
    @classmethod
    def set_config(cls, config: Dict[str, Any]) -> None:
        """Update global configuration."""
        
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get current configuration."""
        
    @classmethod
    def autolog(cls, **overrides) -> None:
        """Enable automatic tracing."""
        # 1. Merge config with overrides
        # 2. Configure Azure Monitor if connection string set
        # 3. Create global tracer instance
        # 4. Register with LangChain (context vars or monkey patch)
        # 5. Set _ACTIVE = True
        
    @classmethod
    def is_active(cls) -> bool:
        """Check if autolog is active."""
        
    @classmethod
    def add_tags(cls, tags: Dict[str, Any]) -> None:
        """Add global tags to all spans."""
        
    @classmethod
    def update_redaction_rules(cls, **flags) -> None:
        """Update redaction configuration."""
        
    @classmethod
    def force_flush(cls) -> None:
        """Force flush all pending spans."""
        
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown tracer and disable auto-logging."""
```

## Implementation Steps

### Step 1: Create Base Infrastructure (Current)
1. Create `tracing/` module structure
2. Implement `config.py` with configuration management
3. Create `tracer.py` skeleton with static API methods
4. Add basic tests for configuration

### Step 2: Implement Auto-Registration
1. Create `callback_manager.py` with context variable approach
2. Implement wrapper for Runnable.invoke
3. Test auto-registration works without explicit callback passing

### Step 3: Add GenAI Attributes
1. Create `genai_attributes.py` module
2. Implement attribute mapping functions
3. Update tracer to use attribute mapper
4. Add tests for attribute presence

### Step 4: Token Aggregation
1. Create `token_aggregator.py`
2. Implement real-time aggregation
3. Add aggregation to root spans
4. Test aggregation logic

## Testing Strategy

### Unit Tests
- `test_config_management.py` - Configuration merging and validation
- `test_static_api.py` - Static methods behavior
- `test_auto_registration.py` - Context-based registration
- `test_genai_attributes.py` - Attribute mapping
- `test_token_aggregation.py` - Usage aggregation

### Integration Tests
- `test_chain_execution.py` - Full chain execution with autolog
- `test_backward_compat.py` - Ensure old API still works

## Backward Compatibility

The old API must continue to work:
```python
# Old way (still supported)
tracer = AzureAIOpenTelemetryTracer(connection_string="...")
result = chain.invoke(input, config={"callbacks": [tracer]})

# New way (autolog)
AzureAIOpenTelemetryTracer.set_app_insights("...")
AzureAIOpenTelemetryTracer.autolog()
result = chain.invoke(input)  # Tracer auto-attached
```

## Next Steps

After Phase 1 is complete and tested:
- Phase 2: Complete GenAI attribute implementation
- Phase 3: Post-processing and RunTree integration
- Phase 4: Monkey patching for maximum compatibility
