"""Token usage aggregation for GenAI spans.

This module provides real-time token usage tracking and aggregation
across nested LLM calls to compute totals at the root agent span.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single operation."""
    
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two token usage objects."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class AggregatedUsage:
    """Aggregated token usage across multiple operations."""
    
    # Per-span usage
    span_usage: Dict[str, TokenUsage] = field(default_factory=dict)
    
    # Aggregated totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # Lock for thread-safe updates
    _lock: Lock = field(default_factory=Lock)
    
    def add_span_usage(self, span_id: str, usage: TokenUsage) -> None:
        """Add usage for a specific span.
        
        Args:
            span_id: Span identifier
            usage: Token usage for this span
        """
        with self._lock:
            self.span_usage[span_id] = usage
            
            # Update totals
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.total_tokens += usage.total_tokens
    
    def get_totals(self) -> Dict[str, int]:
        """Get aggregated totals.
        
        Returns:
            Dictionary with total_input_tokens, total_output_tokens, total_tokens
        """
        with self._lock:
            return {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens
            }
    
    def reset(self) -> None:
        """Reset all usage data."""
        with self._lock:
            self.span_usage.clear()
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_tokens = 0


class TokenAggregator:
    """Manages token usage aggregation across execution hierarchies."""
    
    def __init__(self):
        """Initialize the token aggregator."""
        # Map from root run ID to aggregated usage
        self._root_usage: Dict[str, AggregatedUsage] = {}
        
        # Map from any run ID to its root run ID
        self._run_to_root: Dict[str, str] = {}
        
        self._lock = Lock()
    
    def register_run(self, run_id: str, parent_run_id: Optional[str] = None) -> None:
        """Register a run and its parent relationship.
        
        Args:
            run_id: The run ID
            parent_run_id: Parent run ID, or None if this is a root
        """
        with self._lock:
            if parent_run_id is None:
                # This is a root run
                root_id = run_id
                self._run_to_root[run_id] = root_id
                if root_id not in self._root_usage:
                    self._root_usage[root_id] = AggregatedUsage()
            else:
                # Find the root of the parent
                root_id = self._run_to_root.get(parent_run_id)
                if root_id:
                    self._run_to_root[run_id] = root_id
                else:
                    # Parent not registered yet, register this as potential root
                    # (will be corrected later)
                    self._run_to_root[run_id] = run_id
                    if run_id not in self._root_usage:
                        self._root_usage[run_id] = AggregatedUsage()
    
    def add_usage(self, run_id: str, usage: TokenUsage) -> None:
        """Add token usage for a run.
        
        This automatically aggregates the usage to the root run.
        
        Args:
            run_id: The run ID
            usage: Token usage to add
        """
        with self._lock:
            # Find root
            root_id = self._run_to_root.get(run_id)
            if not root_id:
                LOGGER.warning(f"Run {run_id} not registered, cannot add usage")
                return
            
            # Get or create aggregated usage for root
            if root_id not in self._root_usage:
                self._root_usage[root_id] = AggregatedUsage()
            
            # Add to root's aggregated usage
            self._root_usage[root_id].add_span_usage(run_id, usage)
    
    def get_root_usage(self, run_id: str) -> Optional[Dict[str, int]]:
        """Get aggregated usage for a root run.
        
        Args:
            run_id: Run ID (can be root or child)
            
        Returns:
            Dictionary with aggregated totals, or None if not found
        """
        with self._lock:
            root_id = self._run_to_root.get(run_id)
            if not root_id:
                return None
            
            usage = self._root_usage.get(root_id)
            if not usage:
                return None
            
            return usage.get_totals()
    
    def cleanup_run(self, run_id: str) -> None:
        """Clean up data for a completed root run.
        
        Args:
            run_id: Root run ID to clean up
        """
        with self._lock:
            # Remove the root usage
            if run_id in self._root_usage:
                del self._root_usage[run_id]
            
            # Remove all run-to-root mappings for this root
            to_remove = [
                rid for rid, root in self._run_to_root.items()
                if root == run_id
            ]
            for rid in to_remove:
                del self._run_to_root[rid]
    
    def get_usage_for_span(self, run_id: str) -> Optional[TokenUsage]:
        """Get usage for a specific span/run.
        
        Args:
            run_id: The run ID
            
        Returns:
            TokenUsage for this span, or None if not found
        """
        with self._lock:
            root_id = self._run_to_root.get(run_id)
            if not root_id:
                return None
            
            usage = self._root_usage.get(root_id)
            if not usage:
                return None
            
            return usage.span_usage.get(run_id)


# Global token aggregator instance
_global_aggregator = TokenAggregator()


def get_aggregator() -> TokenAggregator:
    """Get the global token aggregator instance.
    
    Returns:
        Global TokenAggregator instance
    """
    return _global_aggregator
