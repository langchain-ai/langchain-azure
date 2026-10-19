"""Persist functional call identities needed to copy paused task writes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from inspect import isawaitable
from threading import Lock
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableBinding, RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.types import PregelExecutableTask

from .checkpoint_ref import CheckpointRef

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.pregel._algo import Call

_IDENTITY_NAMESPACE = "__langgraph_response_tasks__"
_NULL_TASK_ID = "00000000-0000-0000-0000-000000000000"


class FunctionalTaskError(ValueError):
    """A persisted functional call cannot be safely associated with its result."""


@dataclass(frozen=True)
class TaskFork:
    """A task's new ID and its verified source and destination namespaces."""

    task_id: str
    source_namespace: str
    namespace: str


def _path(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise FunctionalTaskError("The functional task path is invalid.")
    result: list[Any] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            result.append(_path(item))
        elif isinstance(item, (str, int)):
            result.append(item)
        else:
            raise FunctionalTaskError("The functional task path is invalid.")
    return tuple(result)


@dataclass(frozen=True)
class _Identity:
    name: str
    parent_ns: str
    path: tuple[Any, ...]
    step: int
    version: int = 1

    @classmethod
    def parse(cls, data: Any) -> _Identity:
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data.get("version") != 1
            or not isinstance(data.get("name"), str)
            or not data["name"]
            or not isinstance(data.get("parent_ns"), str)
            or not data["parent_ns"]
            or not isinstance(data.get("step"), int)
            or isinstance(data["step"], bool)
        ):
            raise FunctionalTaskError("The functional task identity is invalid.")
        path = _path(data.get("path"))
        if (
            len(path) != 4
            or path[0] != "__pregel_push"
            or not isinstance(path[1], tuple)
            or not isinstance(path[2], int)
            or isinstance(path[2], bool)
            or path[2] < 0
            or path[3] is not True
        ):
            raise FunctionalTaskError("The functional task identity is invalid.")
        return cls(data["name"], data["parent_ns"], path, data["step"])

    def task_id(self, checkpoint: Checkpoint) -> str:
        try:
            from langgraph.pregel._algo import Call, prepare_single_task
        except ImportError as exc:
            raise FunctionalTaskError(
                "This LangGraph version cannot restore functional task identities."
            ) from exc

        def placeholder() -> None:
            raise AssertionError("Task identity reconstruction must not execute work.")

        placeholder.__name__ = self.name
        # Delegate identity generation, including checkpoint-format differences,
        # to LangGraph. The caller verifies the original ID before using a copy.
        prepared = prepare_single_task(
            (
                "__pregel_push",
                self.path[1],
                self.path[2],
                "",
                Call(
                    placeholder,
                    ((), {}),
                    retry_policy=None,
                    cache_policy=None,
                    callbacks=None,
                ),
            ),
            None,
            checkpoint=checkpoint,
            checkpoint_id_bytes=bytes.fromhex(checkpoint["id"].replace("-", "")),
            checkpoint_null_version=None,
            pending_writes=[],
            processes={},
            channels={},
            managed={},
            config={"configurable": {"checkpoint_ns": self.parent_ns}},
            step=self.step,
            stop=self.step + 2,
            for_execution=False,
        )
        if prepared is None:
            raise FunctionalTaskError("The functional task identity is unavailable.")
        return prepared.id


def _index_config(config: RunnableConfig, task_id: str) -> RunnableConfig:
    configurable = config.get("configurable") or {}
    ref = CheckpointRef.from_dict(configurable)
    if ref is None or configurable.get("checkpoint_ns", "") != "":
        raise FunctionalTaskError("The functional task checkpoint is invalid.")
    return {
        **config,
        "configurable": {
            **configurable,
            "checkpoint_ns": (f"{_IDENTITY_NAMESPACE}:{ref.checkpoint_id}:{task_id}"),
        },
    }


def _read_index(
    config: RunnableConfig, saved: CheckpointTuple | None
) -> _Identity | None:
    if saved is None:
        return None
    requested = config["configurable"]
    actual = saved.config.get("configurable") or {}
    if any(
        actual.get(key) != requested.get(key)
        for key in ("thread_id", "checkpoint_ns", "checkpoint_id")
    ) or saved.checkpoint.get("id") != requested.get("checkpoint_id"):
        raise FunctionalTaskError("The functional task identity checkpoint is invalid.")
    value = saved.checkpoint["channel_values"].get("identity")
    try:
        data = json.loads(value) if isinstance(value, str) else None
    except ValueError as exc:
        raise FunctionalTaskError("The functional task identity is invalid.") from exc
    return _Identity.parse(data)


def _index_checkpoint(config: RunnableConfig, identity: _Identity) -> Checkpoint:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = config["configurable"]["checkpoint_id"]
    checkpoint["channel_values"] = {"identity": json.dumps(asdict(identity))}
    checkpoint["channel_versions"] = {"identity": 1}
    return checkpoint


def _check_existing(identity: _Identity, existing: _Identity | None) -> bool:
    if existing is not None and existing != identity:
        raise FunctionalTaskError("The functional task identities do not agree.")
    return existing is not None


async def _save_index(
    saver: BaseCheckpointSaver[Any],
    config: RunnableConfig,
    task_id: str,
    identity: _Identity,
) -> None:
    config = _index_config(config, task_id)
    if not _check_existing(
        identity, _read_index(config, await saver.aget_tuple(config))
    ):
        await saver.aput(
            {
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": None,
                },
            },
            _index_checkpoint(config, identity),
            {"source": "update", "step": identity.step, "parents": {}},
            {"identity": 1},
        )


class FunctionalTaskRecorder:
    """Capture scheduled functional calls, including calls served from cache."""

    def __init__(self) -> None:
        self._identities: dict[str, _Identity] = {}
        self._checkpoints: dict[str, Checkpoint] = {}
        self._lock = Lock()

    def record_checkpoint(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        """Retain checkpoint identity and format without retaining graph state."""
        if (config.get("configurable") or {}).get("checkpoint_ns", "") != "":
            return
        header = empty_checkpoint()
        header["id"] = checkpoint["id"]
        header["v"] = checkpoint["v"]
        with self._lock:
            self._checkpoints[checkpoint["id"]] = header

    def bind(self, graph: CompiledStateGraph) -> CompiledStateGraph:
        """Return a graph copy that observes calls before cache lookup.

        Args:
            graph: Graph whose node runnables need request-scoped observation.

        Returns:
            A graph copy with the original execution, writers, and tracing.
        """
        # The complete node establishes context even when its function is untraced.
        return graph.copy(
            {
                "nodes": {
                    name: node.copy(
                        {
                            "bound": RunnableBinding(
                                bound=node.node,
                                config_factories=[self.configure],
                            ),
                            "writers": [],
                        }
                    )
                    if node.node is not None
                    else node
                    for name, node in graph.nodes.items()
                }
            }
        )

    def configure(self, config: RunnableConfig) -> RunnableConfig:
        """Observe LangGraph scheduling without changing task or cache execution."""
        call = (config.get("configurable") or {}).get("__pregel_call")
        if not isinstance(call, partial) or not callable(
            call.keywords.get("schedule_task")
        ):
            raise FunctionalTaskError(
                "This LangGraph version cannot record functional task identities."
            )
        schedule = call.keywords["schedule_task"]

        def record(
            parent: PregelExecutableTask, index: int, call: Call | None = None
        ) -> Any:
            if call is None:
                return schedule(parent, index, call)
            self._record_call(parent, index, call)
            scheduled = schedule(parent, index, call)
            if isawaitable(scheduled):
                return self._persist_reused_async(parent, scheduled)
            if scheduled is not None and any(
                channel == "__return__" for channel, _ in scheduled.writes
            ):
                saver, write_config, task_path = self._reused_write_config(
                    parent, scheduled
                )
                saver.put_writes(
                    write_config, scheduled.writes, scheduled.id, task_path
                )
            return scheduled

        return {"configurable": {"__pregel_call": partial(call, schedule_task=record)}}

    @staticmethod
    def _reused_write_config(
        parent: PregelExecutableTask, task: PregelExecutableTask
    ) -> tuple[BaseCheckpointSaver[Any], RunnableConfig, str]:
        from langgraph.pregel._algo import task_path_str

        configurable = parent.config.get("configurable") or {}
        saver = configurable.get("__pregel_checkpointer")
        if not isinstance(saver, BaseCheckpointSaver):
            raise FunctionalTaskError("The functional task saver is unavailable.")
        config: RunnableConfig = {
            **parent.config,
            "configurable": {
                **configurable,
                "checkpoint_ns": "",
                "checkpoint_id": (configurable.get("checkpoint_map") or {}).get(""),
            },
        }
        return saver, config, task_path_str(task.path)

    async def _persist_reused_async(
        self,
        parent: PregelExecutableTask,
        scheduled: Awaitable[PregelExecutableTask | None],
    ) -> PregelExecutableTask | None:
        task = await scheduled
        if task is not None and any(
            channel == "__return__" for channel, _ in task.writes
        ):
            # Older LangGraph versions emit cache hits without persisting them.
            saver, config, task_path = self._reused_write_config(parent, task)
            await saver.aput_writes(config, task.writes, task.id, task_path)
        return task

    def _record_call(
        self, parent: PregelExecutableTask, index: int, call: Call
    ) -> None:
        from langgraph.pregel._call import get_runnable_for_task

        configurable = parent.config.get("configurable") or {}
        checkpoint_id = (configurable.get("checkpoint_map") or {}).get("")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise FunctionalTaskError("The functional task checkpoint is invalid.")
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise FunctionalTaskError("The functional task checkpoint is unavailable.")
        identity = _Identity.parse(
            {
                "name": get_runnable_for_task(call.func).name,
                "parent_ns": configurable.get("checkpoint_ns"),
                "path": ("__pregel_push", parent.path, index, True),
                "step": (parent.config.get("metadata") or {}).get("langgraph_step"),
                "version": 1,
            }
        )
        task_id = identity.task_id(checkpoint)
        with self._lock:
            _check_existing(identity, self._identities.get(task_id))
            self._identities[task_id] = identity

    def put(
        self, saver: BaseCheckpointSaver[Any], config: RunnableConfig, task_id: str
    ) -> None:
        """Persist a root call's identity before its result, using the same saver."""
        with self._lock:
            identity = self._identities.get(task_id)
        if (
            identity is None
            or (config.get("configurable") or {}).get("checkpoint_ns", "") != ""
        ):
            return
        config = _index_config(config, task_id)
        if not _check_existing(identity, _read_index(config, saver.get_tuple(config))):
            saver.put(
                {
                    **config,
                    "configurable": {
                        **config["configurable"],
                        "checkpoint_id": None,
                    },
                },
                _index_checkpoint(config, identity),
                {"source": "update", "step": identity.step, "parents": {}},
                {"identity": 1},
            )

    async def aput(
        self, saver: BaseCheckpointSaver[Any], config: RunnableConfig, task_id: str
    ) -> None:
        """Persist an asynchronous root call's identity before its result."""
        with self._lock:
            identity = self._identities.get(task_id)
        if (
            identity is not None
            and (config.get("configurable") or {}).get("checkpoint_ns", "") == ""
        ):
            await _save_index(saver, config, task_id, identity)


async def fork_functional_tasks(
    saver: BaseCheckpointSaver[Any],
    saved: CheckpointTuple,
    fork_config: RunnableConfig,
    checkpoint: Checkpoint,
    root_ids: Mapping[str, str],
) -> dict[str, TaskFork]:
    """Rebind persisted functional calls using verified upstream task identities."""
    pending = {
        task_id
        for task_id, channel, _ in saved.pending_writes or ()
        if task_id not in root_ids
        and task_id not in {"", _NULL_TASK_ID}
        and channel in {"__return__", "__error__", "__interrupt__", "__resume__"}
    }
    identities = {}
    while pending:
        task_id = pending.pop()
        if task_id in root_ids or task_id in identities:
            continue
        config = _index_config(saved.config, task_id)
        identity = _read_index(config, await saver.aget_tuple(config))
        if identity is None:
            raise FunctionalTaskError(
                "The paused checkpoint lacks a persisted functional task identity."
            )
        if identity.task_id(saved.checkpoint) != task_id:
            raise FunctionalTaskError(
                "The persisted functional task identity could not be verified."
            )
        identities[task_id] = identity
        for segment in identity.parent_ns.split("|"):
            parent_id = segment.rsplit(":", 1)[-1]
            if parent_id not in root_ids and parent_id not in identities:
                pending.add(parent_id)

    remapped = dict(root_ids)
    forks: dict[str, TaskFork] = {}
    while identities:
        ready = False
        for task_id, identity in list(identities.items()):
            segments = [
                segment.rsplit(":", 1) for segment in identity.parent_ns.split("|")
            ]
            if any(
                len(segment) != 2 or segment[1] not in remapped for segment in segments
            ):
                continue
            parent_ns = "|".join(
                f"{name}:{remapped[parent_id]}" for name, parent_id in segments
            )
            fork_identity = replace(identity, parent_ns=parent_ns)
            fork_id = fork_identity.task_id(checkpoint)
            if fork_id in remapped.values():
                raise FunctionalTaskError(
                    "The fork functional task identities collide."
                )
            await _save_index(saver, fork_config, fork_id, fork_identity)
            remapped[task_id] = fork_id
            forks[task_id] = TaskFork(
                fork_id,
                f"{identity.parent_ns}|{identity.name}:{task_id}",
                f"{parent_ns}|{identity.name}:{fork_id}",
            )
            del identities[task_id]
            ready = True
        if not ready:
            raise FunctionalTaskError("The functional task ancestry is invalid.")
    return forks
