# Copyright (c) Microsoft. All rights reserved.

"""Two-tier state persistence: Zero-Config `InMemorySaver` bundle handoff vs. external.

checkpointers.

`langchain_azure_compute.graph` supports two state persistence tiers transparently:

1. **Tier 1 - Persistent Checkpointer (Pointer-Only Mode):**
   When the caller supplies an external checkpointer (`SqliteSaver`, `AsyncSqliteSaver`,
   `PostgresSaver`, etc.), `inline_checkpoints` defaults to `False`. Graph state lives
   exclusively in that external store. DTS orchestration payloads carry only lightweight
   control pointers (`thread_id`, `checkpoint_id`, step counters, and tool/interrupt
   envelopes), keeping DTS history `O(steps)` regardless of state size.

2. **Tier 2 - Zero-Config `InMemorySaver` (Inline Checkpoint Bundle Mode):**
   When a user compiles a graph with `InMemorySaver()` (or omits a checkpointer),
   `inline_checkpoints` defaults to `True`. Because `InMemorySaver` lives in process
   RAM, a worker crash or scale-out across multiple worker processes would lose the
   in-memory checkpoints unless DTS carries them.
   - After each superstep, `_export_checkpoint_bundle()` extracts only the latest
     checkpoint (plus its immediate parent for static-breakpoint inspection), pending
     writes, and referenced channel blobs for that `thread_id` into a compact base64
     dict (`checkpoint_bundle`).
   - On the next activity or after a worker restart, `_hydrate_checkpoint_bundle()`
     restores that slice into the worker's `InMemorySaver` before running the step,
     and `Run.wait_result()` hydrates the client's `InMemorySaver` so multi-turn
     conversations and `graph.get_state()` work seamlessly out of the box.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast

import ormsgpack
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from langchain_azure_compute.graph._config import (
    _SERDE,
    CheckpointBundle,
    Checkpointer,
    CompiledGraph,
    _strict_msgpack_ext_hook,
)

MAX_INLINE_THREAD_BUNDLES = 256
MAX_BUNDLE_BYTES = 900_000
_INLINE_SAVER_LOCK = threading.RLock()

# Retain saver identity in the bounded cache: thread IDs are not globally unique,
# and retaining the saver prevents object-id reuse from leaking a previous graph.
_INLINE_THREAD_BUNDLES: dict[tuple[Checkpointer, str], CheckpointBundle] = {}


def _cache_inline_bundle(
    saver: Checkpointer, thread: str, bundle: CheckpointBundle
) -> None:
    key = (saver, str(thread))
    _INLINE_THREAD_BUNDLES.pop(key, None)
    while len(_INLINE_THREAD_BUNDLES) >= MAX_INLINE_THREAD_BUNDLES:
        _INLINE_THREAD_BUNDLES.pop(next(iter(_INLINE_THREAD_BUNDLES)), None)
    _INLINE_THREAD_BUNDLES[key] = bundle


def _get_cached_inline_bundle(
    graph: CompiledGraph, thread: str
) -> CheckpointBundle | None:
    """Return the cached bundle belonging to this graph's checkpointer and thread."""
    saver = _ensure_checkpointer(graph)
    with _INLINE_SAVER_LOCK:
        return _INLINE_THREAD_BUNDLES.get((saver, str(thread)))


def _encode_typed(pair: tuple[str, bytes]) -> list[str]:
    return [pair[0], base64.b64encode(pair[1]).decode("ascii")]


def _decode_typed(item: Sequence[str]) -> tuple[str, bytes]:
    if not isinstance(item, (list, tuple)) or len(item) != 2:  # pragma: no cover
        raise ValueError("invalid checkpoint bundle typed entry")  # pragma: no cover
    kind, b64 = item[0], item[1]
    if not isinstance(kind, str) or kind not in (
        "json",
        "msgpack",
        "null",
        "bytes",
        "empty",
    ):  # pragma: no cover
        raise ValueError(
            f"unsupported checkpoint bundle entry type {kind!r}"
        )  # pragma: no cover
    if not isinstance(b64, str):  # pragma: no cover
        raise ValueError(
            "checkpoint bundle entry payload must be a base64 string"
        )  # pragma: no cover
    raw = base64.b64decode(b64)
    if kind == "msgpack":
        ormsgpack.unpackb(
            raw, ext_hook=_strict_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
        )
    elif kind != "empty":
        _SERDE.loads_typed((kind, raw))
    return (kind, raw)


def _ensure_checkpointer(graph: CompiledGraph) -> Checkpointer:
    """Return `graph.checkpointer`, attaching a fresh `InMemorySaver` if none is.

    configured.
    """
    cp = getattr(graph, "checkpointer", None)
    if cp is None:  # pragma: no cover
        cp = InMemorySaver(serde=_SERDE)  # pragma: no cover
        graph.checkpointer = cp  # pragma: no cover
    elif isinstance(cp, InMemorySaver):
        cp.serde = _SERDE
    return cp  # type: ignore[return-value]


def _is_inline_checkpointer(graph: CompiledGraph) -> bool:
    """Return True if the graph is configured to ship `InMemorySaver` checkpoint.

    bundles via DTS.
    """
    flag = getattr(graph, "_durable_inline_checkpoints", None)
    if flag is not None:
        return bool(flag)
    cp = _ensure_checkpointer(graph)
    return isinstance(cp, InMemorySaver)


def _export_checkpoint_bundle(
    graph: CompiledGraph, thread: str
) -> CheckpointBundle | None:
    """Export the minimal in-memory checkpoint slice needed to resume a thread.

    Includes latest and parent checkpoints, writes, and live blobs for crash recovery.
    """
    if not _is_inline_checkpointer(graph):  # pragma: no cover
        return None  # pragma: no cover
    saver = getattr(graph, "checkpointer", None)
    if not isinstance(saver, InMemorySaver):  # pragma: no cover
        return None  # pragma: no cover
    with _INLINE_SAVER_LOCK:
        ns_map = saver.storage.get(thread)
        if not ns_map:  # pragma: no cover
            return None  # pragma: no cover

        out_storage: dict[str, dict[str, list[Any]]] = {}
        out_writes: list[list[Any]] = []
        out_blobs: list[list[Any]] = []
        needed_blobs: set[tuple[str, str, Any]] = set()

        has_ua = bool(getattr(graph, "_durable_user_interrupt_after", None))
        for ns, ckpts in list(ns_map.items()):
            if not ckpts:
                continue
            latest_cid = max(ckpts.keys())
            keep_cids: list[str] = [latest_cid]
            if has_ua:
                _, _, parent_cid = ckpts[latest_cid]
                if (
                    parent_cid and parent_cid in ckpts and parent_cid not in keep_cids
                ):  # pragma: no cover
                    keep_cids.append(parent_cid)
            ns_out: dict[str, list[Any]] = {}
            for cid in keep_cids:
                ckpt_typed, meta_typed, par_id = ckpts[cid]
                ns_out[cid] = [
                    _encode_typed(ckpt_typed),
                    _encode_typed(meta_typed),
                    par_id,
                ]
                ckpt_obj = saver.serde.loads_typed(ckpt_typed)
                for ch, ver in (ckpt_obj.get("channel_versions") or {}).items():
                    needed_blobs.add((ns, ch, ver))
                w_dict = saver.writes.get((thread, ns, cid))
                if w_dict:
                    w_entries = [
                        [task_id, idx, ch, _encode_typed(val_typed), task_path]
                        for (task_id, idx), (
                            _,
                            ch,
                            val_typed,
                            task_path,
                        ) in w_dict.items()
                    ]
                    out_writes.append([ns, cid, w_entries])
            out_storage[ns] = ns_out

        for ns, ch, ver in needed_blobs:
            blob_key = (thread, ns, ch, ver)
            if blob_key in saver.blobs:
                out_blobs.append([ns, ch, ver, _encode_typed(saver.blobs[blob_key])])

        bundle: CheckpointBundle = {
            "v": 1,
            "storage": out_storage,
            "writes": out_writes,
            "blobs": out_blobs,
        }
        from langchain_azure_compute.graph._config import RunError, _size

        bundle_bytes = _size(bundle)
        if bundle_bytes > MAX_BUNDLE_BYTES:
            raise RunError(
                f"checkpoint_bundle size ({bundle_bytes} bytes) exceeds the "
                f"{MAX_BUNDLE_BYTES}-byte DTS payload limit; configure a persistent "
                "checkpointer (SqliteSaver, PostgresSaver, CosmosDBSaver)"
            )
        _cache_inline_bundle(saver, thread, bundle)
        return bundle


def _hydrate_checkpoint_bundle(
    graph: CompiledGraph,
    thread: str,
    bundle: CheckpointBundle | Mapping[str, Any] | None,
    *,
    only_if_missing_cfg: RunnableConfig | None = None,
    preserve_newer: bool = False,
) -> None:
    """Restore an exported `checkpoint_bundle` into `graph.checkpointer`.

    (`InMemorySaver`).
    """
    if not bundle:  # pragma: no cover
        return  # pragma: no cover
    with _INLINE_SAVER_LOCK:
        saver = _ensure_checkpointer(graph)
        if not isinstance(saver, InMemorySaver):  # pragma: no cover
            return  # pragma: no cover
        if (
            only_if_missing_cfg is not None
            and saver.get_tuple(only_if_missing_cfg) is not None
        ):
            return
        if preserve_newer:
            incoming = bundle.get("storage") or {}
            # Bundles contain only the latest slice, not every older checkpoint.
            # Compare IDs rather than treating all omitted historical IDs as newer.
            for ns, existing_ckpts in saver.storage.get(thread, {}).items():
                incoming_ckpts = incoming.get(ns, {})
                if existing_ckpts and (
                    not incoming_ckpts or max(existing_ckpts) > max(incoming_ckpts)
                ):
                    return
        saver.delete_thread(thread)
        for ns, ckpts in (bundle.get("storage") or {}).items():
            ns_dict = saver.storage[thread][ns]
            for cid, entry in ckpts.items():
                ns_dict[cid] = (
                    _decode_typed(entry[0]),
                    _decode_typed(entry[1]),
                    entry[2],
                )
        for ns, cid, w_entries in bundle.get("writes") or ():
            outer = saver.writes[(thread, ns, cid)]
            for task_id, idx, ch, val_enc, task_path in w_entries:
                outer[(task_id, int(idx))] = (
                    task_id,
                    ch,
                    _decode_typed(val_enc),
                    task_path,
                )
        for ns, ch, ver, blob_enc in bundle.get("blobs") or ():
            saver.blobs[(thread, ns, ch, ver)] = _decode_typed(blob_enc)
        _cache_inline_bundle(saver, thread, cast(CheckpointBundle, dict(bundle)))
