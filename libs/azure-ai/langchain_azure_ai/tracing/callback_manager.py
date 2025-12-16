"""Callback manager for auto-registration of Azure AI OpenTelemetry tracer.

This module provides the mechanism to automatically attach the tracer to LangChain
operations without requiring explicit callback passing.
"""

from __future__ import annotations

import functools
import logging
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, TypeVar

LOGGER = logging.getLogger(__name__)

# Context variable to store the active tracer instance
_active_tracer_context: ContextVar[Optional[Any]] = ContextVar(
    "azure_ai_active_tracer", default=None
)

# Track whether monkey patching has been applied
_monkey_patch_applied: bool = False

# Store original methods for restoration
_original_runnable_invoke: Optional[Callable] = None
_original_runnable_ainvoke: Optional[Callable] = None

F = TypeVar("F", bound=Callable[..., Any])


def set_active_tracer(tracer: Optional[Any]) -> None:
    """Set the active tracer in context.
    
    Args:
        tracer: Tracer instance to activate, or None to deactivate
    """
    _active_tracer_context.set(tracer)


def get_active_tracer() -> Optional[Any]:
    """Get the currently active tracer from context.
    
    Returns:
        Active tracer instance or None
    """
    return _active_tracer_context.get()


def apply_monkey_patches() -> None:
    """Apply monkey patches to LangChain for automatic tracer injection.
    
    This patches key LangChain entry points to automatically inject the
    active tracer into callbacks without requiring explicit passing.
    """
    global _monkey_patch_applied, _original_runnable_invoke, _original_runnable_ainvoke
    
    if _monkey_patch_applied:
        LOGGER.debug("Monkey patches already applied, skipping")
        return
    
    try:
        from langchain_core.runnables import Runnable
        
        # Store original methods for restoration
        _original_runnable_invoke = Runnable.invoke
        _original_runnable_ainvoke = getattr(Runnable, "ainvoke", None)
        
        @functools.wraps(_original_runnable_invoke)
        def patched_invoke(self: Any, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
            """Patched invoke that auto-injects tracer."""
            tracer = get_active_tracer()
            
            if tracer is not None:
                # Inject tracer into callbacks
                if config is None:
                    config = {}
                
                callbacks = config.get("callbacks", [])
                if not isinstance(callbacks, list):
                    callbacks = [callbacks] if callbacks else []
                
                # Only add if not already present
                if tracer not in callbacks:
                    callbacks = callbacks + [tracer]
                    config = {**config, "callbacks": callbacks}
            
            return _original_runnable_invoke(self, input, config, **kwargs)
        
        # Apply patches
        Runnable.invoke = patched_invoke
        
        # Patch async invoke if it exists
        if _original_runnable_ainvoke is not None:
            @functools.wraps(_original_runnable_ainvoke)
            async def patched_ainvoke(self: Any, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
                """Patched ainvoke that auto-injects tracer."""
                tracer = get_active_tracer()
                
                if tracer is not None:
                    if config is None:
                        config = {}
                    
                    callbacks = config.get("callbacks", [])
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks] if callbacks else []
                    
                    if tracer not in callbacks:
                        callbacks = callbacks + [tracer]
                        config = {**config, "callbacks": callbacks}
                
                return await _original_runnable_ainvoke(self, input, config, **kwargs)
            
            Runnable.ainvoke = patched_ainvoke
        
        _monkey_patch_applied = True
        LOGGER.info("Successfully applied monkey patches for auto-registration")
        
    except ImportError as e:
        LOGGER.warning(f"Could not apply monkey patches: {e}")
        raise RuntimeError(
            "Failed to apply monkey patches. LangChain Core may not be installed."
        ) from e
        
    except ImportError as e:
        LOGGER.warning(f"Could not apply monkey patches: {e}")
        raise RuntimeError(
            "Failed to apply monkey patches. LangChain Core may not be installed."
        ) from e
    except Exception as e:
        LOGGER.error(f"Unexpected error applying monkey patches: {e}")
        raise


def remove_monkey_patches() -> None:
    """Remove monkey patches and restore original LangChain methods.
    
    This restores the original Runnable.invoke and Runnable.ainvoke methods.
    Note: If other code has also patched these methods after we did,
    this may not fully restore the original behavior.
    """
    global _monkey_patch_applied, _original_runnable_invoke, _original_runnable_ainvoke
    
    if not _monkey_patch_applied:
        return
    
    try:
        from langchain_core.runnables import Runnable
        
        # Restore original methods if we stored them
        if _original_runnable_invoke is not None:
            Runnable.invoke = _original_runnable_invoke
            _original_runnable_invoke = None
        
        if _original_runnable_ainvoke is not None:
            Runnable.ainvoke = _original_runnable_ainvoke
            _original_runnable_ainvoke = None
        
        _monkey_patch_applied = False
        LOGGER.info("Successfully removed monkey patches")
        
    except Exception as e:
        LOGGER.warning(f"Error removing monkey patches: {e}")


def is_monkey_patched() -> bool:
    """Check if monkey patches have been applied.
    
    Returns:
        True if monkey patches are active
    """
    return _monkey_patch_applied
