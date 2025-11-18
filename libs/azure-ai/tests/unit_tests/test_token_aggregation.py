"""Tests for token usage aggregation."""

import pytest

from langchain_azure_ai.tracing.token_aggregator import (
    AggregatedUsage,
    TokenAggregator,
    TokenUsage,
    get_aggregator,
)


def test_token_usage_creation():
    """Test TokenUsage creation."""
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30


def test_token_usage_addition():
    """Test adding two TokenUsage objects."""
    usage1 = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = TokenUsage(input_tokens=5, output_tokens=15, total_tokens=20)
    
    total = usage1 + usage2
    
    assert total.input_tokens == 15
    assert total.output_tokens == 35
    assert total.total_tokens == 50


def test_token_usage_to_dict():
    """Test converting TokenUsage to dictionary."""
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    
    dict_form = usage.to_dict()
    
    assert dict_form == {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30
    }


def test_aggregated_usage_add_span():
    """Test adding span usage to aggregated usage."""
    agg = AggregatedUsage()
    
    usage1 = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15)
    
    agg.add_span_usage("span1", usage1)
    agg.add_span_usage("span2", usage2)
    
    totals = agg.get_totals()
    assert totals["total_input_tokens"] == 15
    assert totals["total_output_tokens"] == 30
    assert totals["total_tokens"] == 45


def test_aggregated_usage_reset():
    """Test resetting aggregated usage."""
    agg = AggregatedUsage()
    
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    agg.add_span_usage("span1", usage)
    
    assert agg.get_totals()["total_tokens"] == 30
    
    agg.reset()
    
    totals = agg.get_totals()
    assert totals["total_input_tokens"] == 0
    assert totals["total_output_tokens"] == 0
    assert totals["total_tokens"] == 0


def test_token_aggregator_register_root():
    """Test registering a root run."""
    aggregator = TokenAggregator()
    
    aggregator.register_run("root1", parent_run_id=None)
    
    # Should be able to add usage
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    aggregator.add_usage("root1", usage)
    
    root_usage = aggregator.get_root_usage("root1")
    assert root_usage["total_tokens"] == 30


def test_token_aggregator_register_child():
    """Test registering child runs."""
    aggregator = TokenAggregator()
    
    aggregator.register_run("root1", parent_run_id=None)
    aggregator.register_run("child1", parent_run_id="root1")
    aggregator.register_run("child2", parent_run_id="child1")
    
    # Add usage to children
    usage1 = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15)
    
    aggregator.add_usage("child1", usage1)
    aggregator.add_usage("child2", usage2)
    
    # Both should aggregate to root
    root_usage = aggregator.get_root_usage("root1")
    assert root_usage["total_tokens"] == 45
    
    # Can also query via child IDs
    child_usage = aggregator.get_root_usage("child1")
    assert child_usage["total_tokens"] == 45


def test_token_aggregator_multiple_roots():
    """Test multiple independent root runs."""
    aggregator = TokenAggregator()
    
    aggregator.register_run("root1", parent_run_id=None)
    aggregator.register_run("root2", parent_run_id=None)
    
    usage1 = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15)
    
    aggregator.add_usage("root1", usage1)
    aggregator.add_usage("root2", usage2)
    
    # Should be independent
    root1_usage = aggregator.get_root_usage("root1")
    root2_usage = aggregator.get_root_usage("root2")
    
    assert root1_usage["total_tokens"] == 30
    assert root2_usage["total_tokens"] == 15


def test_token_aggregator_cleanup():
    """Test cleaning up completed runs."""
    aggregator = TokenAggregator()
    
    aggregator.register_run("root1", parent_run_id=None)
    aggregator.register_run("child1", parent_run_id="root1")
    
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    aggregator.add_usage("child1", usage)
    
    # Before cleanup
    assert aggregator.get_root_usage("root1") is not None
    
    # Cleanup
    aggregator.cleanup_run("root1")
    
    # After cleanup
    assert aggregator.get_root_usage("root1") is None
    assert aggregator.get_root_usage("child1") is None


def test_token_aggregator_get_usage_for_span():
    """Test getting usage for a specific span."""
    aggregator = TokenAggregator()
    
    aggregator.register_run("root1", parent_run_id=None)
    aggregator.register_run("child1", parent_run_id="root1")
    
    usage1 = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15)
    
    aggregator.add_usage("root1", usage1)
    aggregator.add_usage("child1", usage2)
    
    # Get specific span usage
    span_usage = aggregator.get_usage_for_span("child1")
    
    assert span_usage is not None
    assert span_usage.total_tokens == 15


def test_token_aggregator_unregistered_run():
    """Test adding usage for unregistered run."""
    aggregator = TokenAggregator()
    
    usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    
    # Should not raise, but log warning
    aggregator.add_usage("unregistered", usage)
    
    # Should return None
    assert aggregator.get_root_usage("unregistered") is None


def test_get_global_aggregator():
    """Test getting global aggregator instance."""
    aggregator1 = get_aggregator()
    aggregator2 = get_aggregator()
    
    # Should be same instance
    assert aggregator1 is aggregator2


def test_token_aggregator_thread_safety():
    """Test thread-safe operations."""
    import threading
    
    aggregator = TokenAggregator()
    aggregator.register_run("root1", parent_run_id=None)
    
    errors = []
    
    def add_usage(span_id, amount):
        try:
            usage = TokenUsage(input_tokens=amount, output_tokens=amount, total_tokens=amount*2)
            aggregator.add_usage(span_id, usage)
        except Exception as e:
            errors.append(e)
    
    # Register children
    for i in range(10):
        aggregator.register_run(f"child{i}", parent_run_id="root1")
    
    # Add usage concurrently
    threads = [
        threading.Thread(target=add_usage, args=(f"child{i}", i+1))
        for i in range(10)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0
    
    # Check total
    root_usage = aggregator.get_root_usage("root1")
    expected_total = sum((i+1)*2 for i in range(10))
    assert root_usage["total_tokens"] == expected_total
