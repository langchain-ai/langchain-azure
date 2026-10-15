# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for Responses resilience storage."""

import asyncio
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from azure.ai.agentserver.core.tasks import TaskContext
from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import ResponseObject
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import InvocationsHostServer, ResponsesHostServer
from langchain_azure_ai.agents.hosting._responses import (
    CONVERSATION_CHAIN_STORE_PREFIX,
    CONVERSATION_CHECKPOINT_KEY,
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    CheckpointRef,
    FoundryConversationChainStore,
    HostingRunnableConfig,
    TaskStorageManager,
    consume_pending_interrupt_ids,
    conversation_chain_store,
    load_pending_interrupt_index,
    store_pending_interrupt_ids,
)

from .hitl.conftest import assistant_text, resume_item, sentinels
from .hitl.graphs import build_simple_interrupt_graph


def test_task_configuration_override_reports_state_read_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    base_store = FoundryConversationChainStore()

    class FailingStateStore:
        fail = False

        async def get(self, chain: str, key: str) -> dict[str, str] | None:
            if self.fail and key.startswith("langgraph_pending_interrupts:"):
                raise RuntimeError("Pause state is unavailable.")
            return await base_store.get(chain, key)

        async def set(self, chain: str, key: str, value: dict[str, str]) -> None:
            await base_store.set(chain, key, value)

    class CustomConfigHost(InvocationsHostServer):
        def build_task_runnable_config(
            self, session_id: str, context: TaskContext[dict[str, Any]]
        ) -> RunnableConfig:
            config = super().build_task_runnable_config(session_id, context)
            config["configurable"]["thread_id"] = f"custom-{context.task_id}"
            return config

    store = FailingStateStore()
    host = CustomConfigHost(
        build_simple_interrupt_graph(),
        options=ResponsesServerOptions(resilient_background=True),
        conversation_chain_store=store,
    )
    endpoint = "/invocations?agent_session_id=custom-config-state-error"
    with TestClient(host.app) as client:
        first = client.post(endpoint, json={"message": "start"})
        assert first.status_code == 200, first.text
        call_id = sentinels(first.json())[0]["call_id"]
        store.fail = True
        failed = client.post(
            endpoint, json={"message": [resume_item(call_id, "discarded")]}
        )
        assert failed.status_code == 500, failed.text
        invocation = client.get(
            f"/invocations/{failed.headers['x-agent-invocation-id']}"
        )
        assert invocation.status_code == 200, invocation.text
        assert invocation.json()["status"] == "failed", invocation.text
        assert invocation.json()["error"]["code"] == "internal_error"

        store.fail = False
        resumed = client.post(
            endpoint, json={"message": [resume_item(call_id, "accepted")]}
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json() == {"response": "ok:accepted"}


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
@pytest.mark.parametrize("task_backed", [False, True])
def test_failed_pause_consumption_does_not_consume_resume_answers(
    protocol: str,
    task_backed: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    base_store = FoundryConversationChainStore()

    class FailingConsumptionStore:
        fail = False

        async def get(self, chain: str, key: str) -> dict[str, str] | None:
            return await base_store.get(chain, key)

        async def set(self, chain: str, key: str, value: dict[str, str]) -> None:
            if self.fail and value.get("state") == "consumed":
                raise RuntimeError("Pause consumption is unavailable.")
            await base_store.set(chain, key, value)

    store = FailingConsumptionStore()
    graph = build_simple_interrupt_graph()
    options = ResponsesServerOptions(resilient_background=True) if task_backed else None
    host: ResponsesHostServer | InvocationsHostServer
    if protocol == "responses":
        host = ResponsesHostServer(
            graph, options=options, conversation_chain_store=store
        )
        endpoint = "/responses"
        linkage = {"conversation": "consumption-failure"}
        input_key = "input"
    else:
        host = InvocationsHostServer(
            graph, options=options, conversation_chain_store=store
        )
        endpoint = "/invocations?agent_session_id=consumption-failure"
        linkage = {}
        input_key = "message"
    with TestClient(host.app) as client:
        first = client.post(endpoint, json={**linkage, input_key: "start"})
        assert first.status_code == 200, first.text
        call_id = sentinels(first.json())[0]["call_id"]
        store.fail = True
        failed = client.post(
            endpoint,
            json={**linkage, input_key: [resume_item(call_id, "discarded")]},
        )
        if protocol == "responses":
            assert failed.status_code == 200, failed.text
            assert failed.json()["status"] == "failed", failed.text
            assert failed.json()["error"]["code"] == "internal_error"
        else:
            assert failed.status_code == 500, failed.text

        store.fail = False
        resumed = client.post(
            endpoint,
            json={**linkage, input_key: [resume_item(call_id, "accepted")]},
        )
        assert resumed.status_code == 200, resumed.text
        text = (
            assistant_text(resumed.json())
            if protocol == "responses"
            else resumed.json()["response"]
        )
        assert text == "ok:accepted"


@pytest.mark.parametrize("protocol", ["responses", "invocations"])
def test_serializer_supported_cyclic_state_can_pause_and_resume(protocol: str) -> None:
    class State(MessagesState):
        cache: Any

    def cache_value(state: State) -> dict[str, Any]:
        value: list[Any] = []
        value.append(value)
        return {"cache": value}

    def ask(state: State) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"answer:{interrupt('Approve?')}")]}

    builder = StateGraph(State)
    builder.add_node("write_cache", cache_value)
    builder.add_node("ask", ask)
    builder.add_edge(START, "write_cache")
    builder.add_edge(START, "ask")
    builder.add_edge("write_cache", END)
    builder.add_edge("ask", END)
    graph = builder.compile(
        checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
    )
    host: ResponsesHostServer | InvocationsHostServer
    if protocol == "responses":
        host = ResponsesHostServer(graph)
        endpoint = "/responses"
        linkage = {"conversation": "cyclic-checkpoint"}
        input_key = "input"
    else:
        host = InvocationsHostServer(graph)
        endpoint = "/invocations?agent_session_id=cyclic-checkpoint"
        linkage = {}
        input_key = "message"
    with TestClient(host.app) as client:
        first = client.post(endpoint, json={**linkage, input_key: "start"})
        assert first.status_code == 200, first.text
        calls = sentinels(first.json())
        assert len(calls) == 1, first.text
        resumed = client.post(
            endpoint,
            json={
                **linkage,
                input_key: [resume_item(calls[0]["call_id"], "yes")],
            },
        )
        assert resumed.status_code == 200, resumed.text
        text = (
            assistant_text(resumed.json())
            if protocol == "responses"
            else resumed.json()["response"]
        )
        assert text == "answer:yes"


def test_hosting_runnable_config_reads_checkpoint_ref() -> None:
    config = cast(
        RunnableConfig,
        {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_id": "checkpoint-1",
            }
        },
    )

    assert HostingRunnableConfig(config).checkpoint_ref == CheckpointRef(
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
    )


def test_task_storage_manager_reads_checkpoint_ref_from_seeded_stream() -> None:
    response = cast(
        ResponseObject,
        {
            "metadata": {
                "_internal_metadata": {
                    METADATA_LANGGRAPH_THREAD_ID: "thread-1",
                    METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-1",
                }
            }
        },
    )
    stream = ResponseEventStream(response_id="response-1", response=response)

    assert TaskStorageManager.from_stream(stream).checkpoint_ref == CheckpointRef(
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
    )


async def test_stored_checkpoint_requires_thread_id(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] = {
        CONVERSATION_CHECKPOINT_KEY: {"checkpoint_id": "checkpoint-1"}
    }

    store = FoundryConversationChainStore()
    assert (
        CheckpointRef.from_dict(await store.get("chain-1", CONVERSATION_CHECKPOINT_KEY))
        is None
    )


def test_checkpoint_ref_readers_ignore_invalid_values() -> None:
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "", "checkpoint_id": 1}},
    )
    response = cast(
        ResponseObject,
        {
            "metadata": {
                "_internal_metadata": {
                    METADATA_LANGGRAPH_THREAD_ID: "",
                    METADATA_LANGGRAPH_CHECKPOINT_ID: 1,
                }
            }
        },
    )
    stream = ResponseEventStream(response_id="response-1", response=response)

    assert HostingRunnableConfig(config).checkpoint_ref is None
    assert TaskStorageManager.from_stream(stream).checkpoint_ref is None


@pytest.mark.parametrize("count", [0, 2, 120])
async def test_pending_interrupt_index_has_explicit_consumed_generations(
    count: int,
) -> None:
    store = FoundryConversationChainStore()
    ref = CheckpointRef("thread-1", "checkpoint-1")
    interrupt_ids = frozenset(f"interrupt-{index:032}" for index in range(count))

    await store_pending_interrupt_ids(store, ref, interrupt_ids)
    ready = await load_pending_interrupt_index(store, ref)
    assert ready is not None
    assert ready.interrupt_ids == interrupt_ids

    await consume_pending_interrupt_ids(store, ref)
    consumed = await load_pending_interrupt_index(store, ref)
    assert consumed is not None
    assert consumed.interrupt_ids is None
    assert consumed.generation != ready.generation

    await store_pending_interrupt_ids(store, ref, {"next"})
    next_pause = await load_pending_interrupt_index(store, ref)
    assert next_pause is not None
    assert next_pause.interrupt_ids == {"next"}
    assert next_pause.generation != consumed.generation

    await store_pending_interrupt_ids(store, ref, ())
    empty = await load_pending_interrupt_index(store, ref)
    assert empty is not None
    assert empty.interrupt_ids == frozenset()
    assert empty.generation != next_pause.generation


async def test_legacy_pending_interrupt_index_is_untrusted() -> None:
    store = FoundryConversationChainStore()
    ref = CheckpointRef("thread-1", "checkpoint-1")
    await store_pending_interrupt_ids(store, ref, {"pending"})
    key = f"langgraph_pending_interrupts:{ref.checkpoint_id}"
    marker = await store.get(ref.thread_id, key)
    assert marker is not None
    del marker["version"]
    marker["state_version"] = "legacy-content-digest"
    await store.set(ref.thread_id, key, marker)

    assert await load_pending_interrupt_index(store, ref) is None


@pytest.mark.parametrize("corruption", ["marker", "chunk"])
async def test_pending_interrupt_index_propagates_corrupt_storage(
    corruption: str, foundry_state_stores: dict[str, dict[str, object]]
) -> None:
    store = FoundryConversationChainStore()
    ref = CheckpointRef("thread-1", "checkpoint-1")
    await store_pending_interrupt_ids(store, ref, {"pending"})
    key = f"langgraph_pending_interrupts:{ref.checkpoint_id}"
    marker = await store.get(ref.thread_id, key)
    assert marker is not None
    if corruption == "marker":
        marker["chunks"] = "invalid"
        await store.set(ref.thread_id, key, marker)
    else:
        records = foundry_state_stores[
            f"{CONVERSATION_CHAIN_STORE_PREFIX}/{ref.thread_id}"
        ]
        del records[f"{key}:{marker['generation']}:0"]

    with pytest.raises(ValueError):
        await load_pending_interrupt_index(store, ref)


async def test_foundry_conversation_chain_store_preserves_512_characters(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    store = FoundryConversationChainStore()
    data = {"key": "x" * 509}

    await store.set("chain-1", "custom-key", data)

    assert await store.get("chain-1", "custom-key") == data
    assert foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] == {
        "custom-key": data
    }


async def test_foundry_conversation_chain_store_rejects_non_string_dictionary(
    foundry_state_stores: dict[str, dict[str, object]],
) -> None:
    foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] = {
        "invalid-key": "not-a-dictionary"
    }

    with pytest.raises(TypeError, match="is not a string dictionary"):
        await FoundryConversationChainStore().get("chain-1", "invalid-key")

    foundry_state_stores[f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"] = {
        "invalid-key": {"value": 1}
    }
    with pytest.raises(TypeError, match="is not a string dictionary"):
        await FoundryConversationChainStore().get("chain-1", "invalid-key")


@pytest.mark.parametrize("on_mismatch", ["fail", "ignore"])
async def test_foundry_conversation_chain_store_forwards_all_options(
    monkeypatch: pytest.MonkeyPatch,
    on_mismatch: Literal["fail", "ignore"],
) -> None:
    calls: list[tuple[str, object, object, dict[str, object]]] = []
    get_or_create = conversation_chain_store.FoundryStateStore.get_or_create

    async def capture_options(
        name: str,
        credential: Any = None,
        endpoint: Any = None,
        **kwargs: Any,
    ) -> object:
        calls.append((name, credential, endpoint, kwargs))
        return await get_or_create(name, credential, endpoint, **kwargs)

    monkeypatch.setattr(
        conversation_chain_store.FoundryStateStore,
        "get_or_create",
        capture_options,
    )
    credential = cast(Any, object())
    store = FoundryConversationChainStore(
        credential,
        "https://example.services.ai.azure.com/api/projects/test",
        user_isolation=True,
        item_ttl_seconds=600,
        description="Custom description",
        tags={"environment": "test"},
        user_id="user-1",
        api_version="v2",
        on_mismatch=on_mismatch,
        retry_total=3,
    )

    await store.set("chain-1", "key", {"value": "1"})
    await store.get("chain-1", "key")

    expected = (
        f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1",
        credential,
        "https://example.services.ai.azure.com/api/projects/test",
        {
            "user_isolation": True,
            "item_ttl_seconds": 600,
            "description": "Custom description",
            "tags": {"environment": "test"},
            "user_id": "user-1",
            "api_version": "v2",
            "retry_total": 3,
        },
    )
    assert calls == [expected, expected]


@pytest.mark.parametrize("policy", [None, "fail", "ignore"])
@pytest.mark.parametrize("operation", ["get", "set"])
@pytest.mark.parametrize(
    "setting,existing,requested",
    [
        ("user_isolation", False, True),
        ("user_isolation", True, False),
        ("item_ttl_seconds", 600, 1200),
        ("item_ttl_seconds", -1, 600),
    ],
)
async def test_conversation_chain_store_mismatch_policy(
    monkeypatch: pytest.MonkeyPatch,
    foundry_state_stores: dict[str, dict[str, Any]],
    policy: str | None,
    operation: str,
    setting: str,
    existing: Any,
    requested: Any,
) -> None:
    store_name = f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1"
    original = FoundryConversationChainStore(**{setting: existing})
    await original.set("chain-1", "key", {"value": "original"})
    state_store = await conversation_chain_store.FoundryStateStore.get_or_create(
        store_name
    )
    original_properties = await state_store.get()
    read_properties = AsyncMock(wraps=state_store.get)
    read_item = AsyncMock(wraps=state_store.get_item)
    write_item = AsyncMock(wraps=state_store.set_item)
    exit_store = AsyncMock(return_value=None)
    monkeypatch.setattr(type(state_store), "get", read_properties)
    monkeypatch.setattr(type(state_store), "get_item", read_item)
    monkeypatch.setattr(type(state_store), "set_item", write_item)
    monkeypatch.setattr(type(state_store), "__aexit__", exit_store)
    options = {setting: requested}
    if policy is not None:
        options["on_mismatch"] = policy
    store = FoundryConversationChainStore(**options)

    async def perform_operation() -> None:
        if operation == "get":
            assert await store.get("chain-1", "key") == {"value": "original"}
        else:
            await store.set("chain-1", "key", {"value": "replacement"})

    if policy == "ignore":
        await perform_operation()
        read_properties.assert_not_awaited()
        assert read_item.await_count == (operation == "get")
        assert write_item.await_count == (operation == "set")
    else:
        with pytest.raises(
            ValueError, match=f"{setting}={existing}; expected {requested}"
        ) as raised:
            await perform_operation()
        assert store_name in str(raised.value)
        read_properties.assert_awaited_once()
        read_item.assert_not_awaited()
        write_item.assert_not_awaited()
    exit_store.assert_awaited_once()
    assert await state_store.get() == original_properties
    expected = (
        "replacement" if policy == "ignore" and operation == "set" else "original"
    )
    assert foundry_state_stores[store_name] == {"key": {"value": expected}}


def test_conversation_chain_store_rejects_invalid_mismatch_policy() -> None:
    with pytest.raises(ValueError, match="on_mismatch"):
        FoundryConversationChainStore(on_mismatch=cast(Any, "overwrite"))


@pytest.mark.parametrize("on_mismatch", ["fail", "ignore"])
@pytest.mark.parametrize("user_isolation,item_ttl_seconds", [(False, 600), (True, -1)])
async def test_conversation_chain_store_matching_properties(
    monkeypatch: pytest.MonkeyPatch,
    on_mismatch: Literal["fail", "ignore"],
    user_isolation: bool,
    item_ttl_seconds: int,
) -> None:
    state_store = await conversation_chain_store.FoundryStateStore.get_or_create(
        f"{CONVERSATION_CHAIN_STORE_PREFIX}/chain-1",
        user_isolation=user_isolation,
        item_ttl_seconds=item_ttl_seconds,
    )
    read_properties = AsyncMock(wraps=state_store.get)
    monkeypatch.setattr(type(state_store), "get", read_properties)
    store = FoundryConversationChainStore(
        user_isolation=user_isolation,
        item_ttl_seconds=item_ttl_seconds,
        on_mismatch=on_mismatch,
    )

    assert await store.get("chain-1", "missing") is None
    await store.set("chain-1", "key", {"value": "stored"})
    assert await store.get("chain-1", "key") == {"value": "stored"}
    assert read_properties.await_count == (3 if on_mismatch == "fail" else 0)


@pytest.mark.parametrize("on_mismatch", ["fail", "ignore"])
@pytest.mark.parametrize("operation", ["get", "set"])
async def test_conversation_chain_store_policy_propagates_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
    on_mismatch: Literal["fail", "ignore"],
    operation: str,
) -> None:
    store_type = conversation_chain_store.FoundryStateStore
    error = RuntimeError("Storage unavailable")
    read_item = AsyncMock(side_effect=error if on_mismatch == "ignore" else None)
    write_item = AsyncMock(side_effect=error if on_mismatch == "ignore" else None)
    read_properties = AsyncMock(side_effect=error)
    exit_store = AsyncMock(return_value=None)
    monkeypatch.setattr(store_type, "get", read_properties)
    monkeypatch.setattr(store_type, "get_item", read_item)
    monkeypatch.setattr(store_type, "set_item", write_item)
    monkeypatch.setattr(store_type, "__aexit__", exit_store)
    store = FoundryConversationChainStore(on_mismatch=on_mismatch)

    with pytest.raises(RuntimeError, match="Storage unavailable") as raised:
        if operation == "get":
            await store.get("chain-1", "key")
        else:
            await store.set("chain-1", "key", {"value": "stored"})

    assert raised.value is error
    exit_store.assert_awaited_once()
    if on_mismatch == "fail":
        read_properties.assert_awaited_once()
        read_item.assert_not_awaited()
        write_item.assert_not_awaited()
    else:
        read_properties.assert_not_awaited()
        assert read_item.await_count == (operation == "get")
        assert write_item.await_count == (operation == "set")


def test_hosting_runnable_config_returns_pinned_config_copy() -> None:
    config = cast(
        RunnableConfig,
        {
            "tags": ["existing"],
            "configurable": {
                "thread_id": "thread-old",
                "checkpoint_id": "checkpoint-old",
                "response_context": "context",
            },
        },
    )

    updated = (
        HostingRunnableConfig(config)
        .with_checkpoint_ref(CheckpointRef("thread-new", "checkpoint-new"))
        .runnable_config
    )

    assert updated == {
        "tags": ["existing"],
        "configurable": {
            "thread_id": "thread-new",
            "checkpoint_id": "checkpoint-new",
            "checkpoint_ns": "",
            "response_context": "context",
        },
    }
    assert config["configurable"]["checkpoint_id"] == "checkpoint-old"


def test_hosting_runnable_config_creates_thread_without_checkpoint_ref() -> None:
    response_context = cast(ResponseContext, object())

    hosting_config = HostingRunnableConfig.create(
        "thread-1",
        response_context,
    )

    assert hosting_config.checkpoint_ref is None
    assert hosting_config.runnable_config["configurable"] == {
        "thread_id": "thread-1",
        "response_context": response_context,
    }


def test_hosting_runnable_config_wraps_all_hosting_data() -> None:
    response_context = cast(ResponseContext, object())
    cancellation_signal = asyncio.Event()

    hosting_config = HostingRunnableConfig.create_from_checkpoint(
        CheckpointRef("thread-1", "checkpoint-1"),
        response_context,
    ).with_cancellation_signal(cancellation_signal)

    assert hosting_config.checkpoint_ref == CheckpointRef(
        "thread-1",
        "checkpoint-1",
    )
    assert hosting_config.runnable_config["configurable"] == {
        "thread_id": "thread-1",
        "checkpoint_id": "checkpoint-1",
        "checkpoint_ns": "",
        "response_context": response_context,
        "response_cancellation_signal": cancellation_signal,
    }
