"""Identity of Responses pauses that share a LangGraph checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import INTERRUPT, RESUME, CheckpointTuple
from langgraph.types import Interrupt

from .._checkpoint._checkpoint_reuse import _observe_checkpoint_reads
from .hosting_runnable_config import HostingRunnableConfig

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import StateSnapshot

    from .checkpoint_ref import CheckpointRef

_PAUSE_GENERATIONS = "langgraph_pause_generations"
_PARENT_PAUSE = "__langchain_azure_parent_pause"


def _is_current_checkpoint(
    state: StateSnapshot, reference: CheckpointRef | None
) -> bool:
    return (
        reference is not None
        and HostingRunnableConfig(state.config).checkpoint_ref == reference
        and (state.config.get("configurable") or {}).get("checkpoint_ns", "") == ""
        and state.created_at is not None
    )


def _latest_config(config: RunnableConfig) -> RunnableConfig:
    configurable = dict(config.get("configurable") or {})
    configurable.pop("checkpoint_id", None)
    configurable.pop("checkpoint_map", None)
    return {**config, "configurable": configurable}


async def _read_pause_state(
    graph: CompiledStateGraph, config: RunnableConfig, *, latest: bool = False
) -> tuple[StateSnapshot, dict[str, int]]:
    if latest:
        config = _latest_config(config)
        if graph.config and {"checkpoint_id", "checkpoint_map"}.intersection(
            graph.config.get("configurable") or {}
        ):
            graph = graph.copy({"config": _latest_config(graph.config)})
    with _observe_checkpoint_reads(latest=latest) as reads:
        state = await graph.aget_state(config, subgraphs=True)
    return state, _pause_generations(reads, state.interrupts)


def _pause_generations(
    reads: Sequence[CheckpointTuple], pending: Sequence[Interrupt]
) -> dict[str, int]:
    """Use recorded answers to distinguish sequential pauses with the same ID."""
    ids = {interrupt.id for interrupt in pending}
    generations: dict[str, int] = {}
    for read in reads:
        writes = read.pending_writes or ()
        resume_counts = {
            task_id: len(value) if isinstance(value, list) else 1
            for task_id, channel, value in writes
            if channel == RESUME
        }
        for task_id, channel, value in writes:
            if channel != INTERRUPT:
                continue
            for interrupt in value if isinstance(value, Sequence) else (value,):
                if isinstance(interrupt, Interrupt) and interrupt.id in ids:
                    generations[interrupt.id] = max(
                        generations.get(interrupt.id, 0), resume_counts.get(task_id, 0)
                    )
    return generations
