from __future__ import annotations

import sys
from functools import wraps
from typing import Any


def ensure_langgraph_py310_async_context_compat() -> None:
    """Patch LangGraph async node execution for Python 3.10.

    LangGraph's ``RunnableCallable.ainvoke`` relies on task context
    propagation when tracing is enabled. On Python < 3.11, sync graph
    nodes executed through ``run_in_executor`` lose the task-local
    runnable config, which breaks ``langgraph.types.interrupt()``.

    We patch the async wrapper once so the current runnable config stays
    set while the coroutine is awaited. ``run_in_executor`` then copies
    that context into the worker thread, allowing ``interrupt()`` to see
    the per-task scratchpad config instead of failing or replaying the
    graph.
    """
    if sys.version_info >= (3, 11):
        return

    import langgraph._internal._runnable as runnable

    if getattr(
        runnable.RunnableCallable.ainvoke,
        "__langchain_azure_ai_py310_context_patch__",
        False,
    ):
        return

    original_ainvoke = runnable.RunnableCallable.ainvoke

    @wraps(original_ainvoke)
    async def patched_ainvoke(
        self: Any,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        if config is None or runnable.ASYNCIO_ACCEPTS_CONTEXT:
            return await original_ainvoke(self, input, config, **kwargs)

        token = runnable.var_child_runnable_config.set(config)
        try:
            return await original_ainvoke(self, input, config, **kwargs)
        finally:
            runnable.var_child_runnable_config.reset(token)

    setattr(
        patched_ainvoke,
        "__langchain_azure_ai_py310_context_patch__",
        True,
    )
    runnable.RunnableCallable.ainvoke = patched_ainvoke
