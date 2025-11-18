"""Post-processor for RunTree enrichment and attribute normalization.

This module provides optional post-processing of LangChain RunTree data
to enrich spans with additional attributes and aggregate metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    
    enable_post_processor: bool = True
    normalize_operation_names: bool = True
    emit_mlflow_compat: bool = True
    aggregate_usage: bool = True
    emit_missing_spans: bool = False  # Create synthetic spans for missing data
    store_run_tree_snapshot: bool = False  # Store RunTree for debugging


@dataclass
class PostProcessResult:
    """Result of post-processing."""
    
    enriched_count: int = 0  # Number of spans enriched
    synthetic_count: int = 0  # Number of synthetic spans created
    aggregated_tokens: Dict[str, int] = None  # Aggregated token totals
    errors: List[str] = None  # Any errors encountered
    
    def __post_init__(self):
        if self.aggregated_tokens is None:
            self.aggregated_tokens = {}
        if self.errors is None:
            self.errors = []


class RunTreePostProcessor:
    """Post-processor for enriching spans with RunTree data."""
    
    def __init__(self, config: Optional[PostProcessConfig] = None):
        """Initialize the post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config or PostProcessConfig()
    
    def process(
        self,
        run_tree: Any,
        span_registry: Optional[Dict[str, Any]] = None
    ) -> PostProcessResult:
        """Process a RunTree and enrich associated spans.
        
        Args:
            run_tree: LangSmith RunTree object
            span_registry: Registry mapping run IDs to spans
            
        Returns:
            PostProcessResult with enrichment statistics
        """
        result = PostProcessResult()
        
        if not self.config.enable_post_processor:
            LOGGER.debug("Post-processor disabled, skipping")
            return result
        
        try:
            # Walk the RunTree
            self._walk_run_tree(run_tree, span_registry or {}, result)
            
            # Aggregate token usage if enabled
            if self.config.aggregate_usage:
                self._aggregate_token_usage(run_tree, span_registry or {}, result)
            
        except Exception as e:
            error_msg = f"Error during post-processing: {e}"
            LOGGER.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
        
        return result
    
    def _walk_run_tree(
        self,
        run: Any,
        span_registry: Dict[str, Any],
        result: PostProcessResult,
        depth: int = 0
    ) -> None:
        """Walk the RunTree DFS and enrich spans.
        
        Args:
            run: Current run node
            span_registry: Registry of spans
            result: Result object to update
            depth: Current depth in tree
        """
        if not run:
            return
        
        run_id = str(getattr(run, "id", None))
        
        # Find corresponding span
        span = span_registry.get(run_id)
        
        if span:
            # Enrich existing span
            self._enrich_span(span, run)
            result.enriched_count += 1
        elif self.config.emit_missing_spans:
            # Create synthetic span
            self._create_synthetic_span(run)
            result.synthetic_count += 1
        
        # Process children
        child_runs = getattr(run, "child_runs", [])
        for child in child_runs:
            self._walk_run_tree(child, span_registry, result, depth + 1)
    
    def _enrich_span(self, span: Any, run: Any) -> None:
        """Enrich a span with RunTree data.
        
        Args:
            span: OpenTelemetry span
            run: RunTree run node
        """
        try:
            # Normalize operation name if configured
            if self.config.normalize_operation_names:
                self._normalize_operation_name(span, run)
            
            # Add any missing attributes from run metadata
            self._add_missing_attributes(span, run)
            
        except Exception as e:
            LOGGER.warning(f"Error enriching span {run.id}: {e}")
    
    def _normalize_operation_name(self, span: Any, run: Any) -> None:
        """Normalize operation name to GenAI spec.
        
        Args:
            span: OpenTelemetry span
            run: RunTree run node
        """
        # Map LangChain run types to GenAI operation names
        run_type = getattr(run, "run_type", None)
        
        operation_map = {
            "llm": "chat",
            "chat_model": "chat",
            "chain": "invoke_agent",
            "tool": "execute_tool",
            "retriever": "retrieve"
        }
        
        if run_type in operation_map:
            normalized_op = operation_map[run_type]
            # Would set attribute here if span is mutable
            # span.set_attribute("gen_ai.operation.name", normalized_op)
    
    def _add_missing_attributes(self, span: Any, run: Any) -> None:
        """Add any missing attributes from run data.
        
        Args:
            span: OpenTelemetry span
            run: RunTree run node
        """
        # Extract metadata
        metadata = getattr(run, "extra", {}) or {}
        
        # Add conversation ID if present
        if "thread_id" in metadata or "session_id" in metadata:
            conversation_id = metadata.get("thread_id") or metadata.get("session_id")
            # Would set attribute here
            # span.set_attribute("gen_ai.conversation.id", conversation_id)
    
    def _create_synthetic_span(self, run: Any) -> None:
        """Create a synthetic span for a run without a real span.
        
        Args:
            run: RunTree run node
        """
        # In a real implementation, would create an OTel span here
        LOGGER.debug(f"Would create synthetic span for run {run.id}")
    
    def _aggregate_token_usage(
        self,
        run_tree: Any,
        span_registry: Dict[str, Any],
        result: PostProcessResult
    ) -> None:
        """Aggregate token usage across the tree.
        
        Args:
            run_tree: Root RunTree
            span_registry: Registry of spans
            result: Result object to update
        """
        # Collect all token usage from child runs
        total_input = 0
        total_output = 0
        total_total = 0
        
        def collect_usage(run: Any) -> None:
            nonlocal total_input, total_output, total_total
            
            # Get usage from run outputs
            outputs = getattr(run, "outputs", {}) or {}
            llm_output = outputs.get("llm_output", {})
            token_usage = llm_output.get("token_usage", {})
            
            if token_usage:
                total_input += token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                total_output += token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
                total_total += token_usage.get("total_tokens", 0)
            
            # Recurse to children
            for child in getattr(run, "child_runs", []):
                collect_usage(child)
        
        collect_usage(run_tree)
        
        # Store aggregated totals
        result.aggregated_tokens = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_total if total_total > 0 else total_input + total_output
        }
        
        # Attach to root span if available
        root_run_id = str(getattr(run_tree, "id", None))
        root_span = span_registry.get(root_run_id)
        
        if root_span and self.config.emit_mlflow_compat:
            # Would set MLflow attributes here
            # import json
            # mlflow_usage = json.dumps({
            #     "promptTokens": total_input,
            #     "completionTokens": total_output,
            #     "totalTokens": total_total
            # })
            # root_span.set_attribute("mlflow.trace.tokenUsage", mlflow_usage)
            pass


def create_post_processor(config: Optional[Dict[str, Any]] = None) -> RunTreePostProcessor:
    """Create a post-processor with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RunTreePostProcessor
    """
    if config:
        post_config = PostProcessConfig(
            enable_post_processor=config.get("enable_post_processor", True),
            normalize_operation_names=config.get("normalize_operation_names", True),
            emit_mlflow_compat=config.get("emit_mlflow_compat", True),
            aggregate_usage=config.get("aggregate_usage", True),
            emit_missing_spans=config.get("emit_missing_spans", False),
            store_run_tree_snapshot=config.get("store_run_tree_snapshot", False)
        )
    else:
        post_config = PostProcessConfig()
    
    return RunTreePostProcessor(post_config)
