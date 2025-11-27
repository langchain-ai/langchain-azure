"""Tests for post-processor."""

from langchain_azure_ai.tracing.post_processor import (
    PostProcessConfig,
    PostProcessResult,
    RunTreePostProcessor,
    create_post_processor,
)


class MockRun:
    """Mock RunTree run for testing."""
    
    def __init__(self, id, run_type="llm", child_runs=None, outputs=None, extra=None):
        self.id = id
        self.run_type = run_type
        self.child_runs = child_runs or []
        self.outputs = outputs or {}
        self.extra = extra or {}


class MockSpan:
    """Mock OpenTelemetry span for testing."""
    
    def __init__(self, span_id):
        self.span_id = span_id
        self.attributes = {}
    
    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_post_process_config_defaults():
    """Test PostProcessConfig default values."""
    config = PostProcessConfig()
    
    assert config.enable_post_processor is True
    assert config.normalize_operation_names is True
    assert config.emit_mlflow_compat is True
    assert config.aggregate_usage is True
    assert config.emit_missing_spans is False


def test_post_process_result_init():
    """Test PostProcessResult initialization."""
    result = PostProcessResult()
    
    assert result.enriched_count == 0
    assert result.synthetic_count == 0
    assert result.aggregated_tokens == {}
    assert result.errors == []


def test_run_tree_post_processor_creation():
    """Test creating a post-processor."""
    processor = RunTreePostProcessor()
    
    assert processor.config is not None
    assert processor.config.enable_post_processor is True


def test_run_tree_post_processor_with_config():
    """Test creating post-processor with custom config."""
    config = PostProcessConfig(
        enable_post_processor=False,
        normalize_operation_names=False
    )
    processor = RunTreePostProcessor(config)
    
    assert processor.config.enable_post_processor is False
    assert processor.config.normalize_operation_names is False


def test_process_disabled():
    """Test process when post-processor is disabled."""
    config = PostProcessConfig(enable_post_processor=False)
    processor = RunTreePostProcessor(config)
    
    run = MockRun("run1")
    result = processor.process(run, {})
    
    assert result.enriched_count == 0
    assert result.synthetic_count == 0


def test_process_with_span_registry():
    """Test processing with span registry."""
    processor = RunTreePostProcessor()
    
    run = MockRun("run1")
    span = MockSpan("run1")
    span_registry = {"run1": span}
    
    result = processor.process(run, span_registry)
    
    assert result.enriched_count == 1
    assert result.synthetic_count == 0


def test_process_with_child_runs():
    """Test processing with nested runs."""
    processor = RunTreePostProcessor()
    
    child1 = MockRun("child1")
    child2 = MockRun("child2")
    root = MockRun("root", child_runs=[child1, child2])
    
    span_registry = {
        "root": MockSpan("root"),
        "child1": MockSpan("child1"),
        "child2": MockSpan("child2")
    }
    
    result = processor.process(root, span_registry)
    
    assert result.enriched_count == 3  # root + 2 children


def test_process_with_missing_spans():
    """Test processing with missing spans and emit_missing_spans enabled."""
    config = PostProcessConfig(emit_missing_spans=True)
    processor = RunTreePostProcessor(config)
    
    run = MockRun("run1")
    span_registry = {}  # No spans
    
    result = processor.process(run, span_registry)
    
    assert result.enriched_count == 0
    assert result.synthetic_count == 1


def test_aggregate_token_usage():
    """Test token usage aggregation."""
    processor = RunTreePostProcessor()
    
    # Create runs with token usage
    child1 = MockRun(
        "child1",
        outputs={
            "llm_output": {
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        }
    )
    
    child2 = MockRun(
        "child2",
        outputs={
            "llm_output": {
                "token_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 15,
                    "total_tokens": 20
                }
            }
        }
    )
    
    root = MockRun("root", child_runs=[child1, child2])
    
    result = processor.process(root, {})
    
    # Check aggregated totals
    assert result.aggregated_tokens["total_input_tokens"] == 15
    assert result.aggregated_tokens["total_output_tokens"] == 35
    assert result.aggregated_tokens["total_tokens"] == 50


def test_aggregate_token_usage_disabled():
    """Test token usage aggregation when disabled."""
    config = PostProcessConfig(aggregate_usage=False)
    processor = RunTreePostProcessor(config)
    
    child = MockRun(
        "child",
        outputs={
            "llm_output": {
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20
                }
            }
        }
    )
    
    root = MockRun("root", child_runs=[child])
    
    result = processor.process(root, {})
    
    # Should be empty since aggregation disabled
    assert result.aggregated_tokens == {}


def test_process_with_errors():
    """Test that errors are captured gracefully."""
    processor = RunTreePostProcessor()
    
    # Pass invalid run_tree (None)
    result = processor.process(None, {})
    
    # Should capture error but not raise
    assert len(result.errors) > 0


def test_create_post_processor_with_dict_config():
    """Test creating post-processor from dict config."""
    config = {
        "enable_post_processor": False,
        "normalize_operation_names": False,
        "emit_mlflow_compat": False
    }
    
    processor = create_post_processor(config)
    
    assert processor.config.enable_post_processor is False
    assert processor.config.normalize_operation_names is False
    assert processor.config.emit_mlflow_compat is False


def test_create_post_processor_defaults():
    """Test creating post-processor with no config."""
    processor = create_post_processor()
    
    assert processor.config.enable_post_processor is True
    assert processor.config.normalize_operation_names is True


def test_walk_run_tree_depth():
    """Test walking deeply nested run tree."""
    processor = RunTreePostProcessor()
    
    # Create nested structure: root -> child1 -> child2 -> child3
    child3 = MockRun("child3")
    child2 = MockRun("child2", child_runs=[child3])
    child1 = MockRun("child1", child_runs=[child2])
    root = MockRun("root", child_runs=[child1])
    
    span_registry = {
        "root": MockSpan("root"),
        "child1": MockSpan("child1"),
        "child2": MockSpan("child2"),
        "child3": MockSpan("child3")
    }
    
    result = processor.process(root, span_registry)
    
    assert result.enriched_count == 4


def test_normalize_operation_names():
    """Test operation name normalization."""
    processor = RunTreePostProcessor()
    
    # Different run types should map to GenAI operations
    run_types = ["llm", "chat_model", "chain", "tool", "retriever"]
    
    for run_type in run_types:
        run = MockRun("run", run_type=run_type)
        span = MockSpan("run")
        span_registry = {"run": span}
        
        result = processor.process(run, span_registry)
        
        assert result.enriched_count == 1


def test_add_missing_attributes_with_thread_id():
    """Test adding conversation ID from thread_id metadata."""
    processor = RunTreePostProcessor()
    
    run = MockRun(
        "run",
        extra={"thread_id": "conversation-123"}
    )
    span = MockSpan("run")
    span_registry = {"run": span}
    
    result = processor.process(run, span_registry)
    
    assert result.enriched_count == 1
