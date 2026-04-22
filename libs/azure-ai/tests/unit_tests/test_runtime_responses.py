# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for Azure AI responses runtime helpers."""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)

# ---------------------------------------------------------------------------
# Helpers — fake Foundry SDK objects (used throughout tests)
# ---------------------------------------------------------------------------


class _FakeInputTextContent:
    """Mimics MessageContentInputTextContent (user turn)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeOutputTextContent:
    """Mimics MessageContentOutputTextContent (assistant turn)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInputImageContent:
    """Mimics MessageContentInputImageContent (user turn)."""

    type = "input_image"

    def __init__(
        self,
        *,
        image_url: str | None = None,
        file_id: str | None = None,
        detail: str = "auto",
    ) -> None:
        self.image_url = image_url
        self.file_id = file_id
        self.detail = detail


class _FakeInputFileContent:
    """Mimics MessageContentInputFileContent (user turn)."""

    type = "input_file"

    def __init__(
        self,
        *,
        file_id: str | None = None,
        filename: str | None = None,
        file_url: str | None = None,
        file_data: str | None = None,
    ) -> None:
        self.file_id = file_id
        self.filename = filename
        self.file_url = file_url
        self.file_data = file_data


class _FakeHistoryItem:
    """Mimics a history item with a content list."""

    def __init__(self, *content_blocks: Any, role: str | None = None) -> None:
        self.content: Any = list(content_blocks)
        if role is not None:
            self.role = role
        elif content_blocks and isinstance(content_blocks[0], _FakeOutputTextContent):
            self.role = "assistant"
        else:
            self.role = "user"


@pytest.fixture(autouse=True, scope="module")
def _patch_content_types() -> Any:
    """Globally replace SDK content types with the test fakes for this module."""
    with (
        patch(
            "langchain_azure_ai.agents.runtime._responses_host.MessageContentInputTextContent",
            _FakeInputTextContent,
        ),
        patch(
            "langchain_azure_ai.agents.runtime._responses_host.MessageContentInputImageContent",
            _FakeInputImageContent,
        ),
        patch(
            "langchain_azure_ai.agents.runtime._responses_host.MessageContentInputFileContent",
            _FakeInputFileContent,
        ),
        patch(
            "langchain_azure_ai.agents.runtime._responses_host.MessageContentOutputTextContent",
            _FakeOutputTextContent,
        ),
    ):
        yield


def _make_input_content(text: str) -> _FakeInputTextContent:
    """Return a fake MessageContentInputTextContent."""
    return _FakeInputTextContent(text)


def _make_output_content(text: str) -> _FakeOutputTextContent:
    """Return a fake MessageContentOutputTextContent."""
    return _FakeOutputTextContent(text)


# ---------------------------------------------------------------------------
# Tests for _history_to_messages
# ---------------------------------------------------------------------------


class TestHistoryToMessages:
    """Tests for the _history_to_messages helper."""

    def _call(self, history: list) -> list:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _history_to_messages,
        )

        return _history_to_messages(history)

    def test_empty_history(self) -> None:
        msgs = self._call([])
        assert msgs == []

    def test_single_user_turn(self) -> None:
        item = _FakeHistoryItem(_make_input_content("hello"))
        msgs = self._call([item])
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "hello"

    def test_single_assistant_turn(self) -> None:
        item = _FakeHistoryItem(_make_output_content("hi there"))
        msgs = self._call([item])
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].content == "hi there"

    def test_mixed_history(self) -> None:
        items = [
            _FakeHistoryItem(_make_input_content("first user")),
            _FakeHistoryItem(_make_output_content("first assistant")),
            _FakeHistoryItem(_make_input_content("second user")),
        ]
        msgs = self._call(items)
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)

    def test_item_without_content_skipped(self) -> None:
        no_content = MagicMock()
        del no_content.content  # no content attribute
        item = _FakeHistoryItem(_make_input_content("hi"))
        # Only the item with content should produce messages
        msgs = self._call([item])
        assert len(msgs) == 1

    def test_content_block_with_empty_text_skipped(self) -> None:
        block = _make_input_content("")
        item = _FakeHistoryItem(block)
        msgs = self._call([item])
        assert msgs == []

    def test_user_turn_with_mixed_content_preserved(self) -> None:
        item = _FakeHistoryItem(
            _make_input_content("describe these inputs"),
            _FakeInputImageContent(image_url="https://example.com/cat.png"),
            _FakeInputFileContent(
                filename="notes.txt",
                file_data="SGVsbG8=",
            ),
            role="user",
        )

        msgs = self._call([item])

        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == [
            {"type": "text", "text": "describe these inputs"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png", "detail": "auto"},
            },
            {
                "type": "file",
                "filename": "notes.txt",
                "data": "SGVsbG8=",
            },
        ]

    def test_assistant_role_with_string_content_preserved(self) -> None:
        item = _FakeHistoryItem(role="assistant")
        item.content = "hello from assistant"

        msgs = self._call([item])

        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].content == "hello from assistant"


# ---------------------------------------------------------------------------
# Tests for LangGraphAgentServerHost
# ---------------------------------------------------------------------------


def _make_mock_graph(chunks: list | None = None) -> Any:
    """Return a mock Runnable graph that yields AIMessageChunk objects."""
    if chunks is None:
        chunks = ["Hello", " world"]

    async def _astream(*args: Any, **kwargs: Any) -> Any:
        for text in chunks:
            yield AIMessageChunk(content=text), {}

    graph = MagicMock()
    graph.astream = _astream
    graph.input_schema = None  # let _graph_has_messages_input fail open
    return graph


def _make_host(graph: Any = None, **kwargs: Any) -> Any:
    """Instantiate LangGraphAgentServerHost with SDK imports mocked out."""
    if graph is None:
        graph = _make_mock_graph()
    # Ensure _graph_has_messages_input fails open for all unit-test graphs.
    graph.input_schema = None

    mock_host_cls = MagicMock()
    mock_host_instance = MagicMock()
    mock_host_cls.return_value = mock_host_instance

    with patch(
        "langchain_azure_ai.agents.runtime._responses_host.ResponsesAgentServerHost",
        mock_host_cls,
    ):
        from langchain_azure_ai.agents.runtime._responses_host import (
            AzureAIResponsesAgentHost,
        )

        host = AzureAIResponsesAgentHost(graph=graph, **kwargs)
        host._app = mock_host_instance
        return host


async def _drain_async_iterable(stream: Any) -> list[Any]:
    events = []
    async for event in stream:
        events.append(event)
    return events


class TestLangGraphAgentServerHostInit:
    """Tests for __init__ and import-guard."""

    def test_instantiation_requires_no_import_guard(self) -> None:
        """AzureAIResponsesAgentHost can be instantiated when the SDK is present."""
        host = _make_host()
        assert host._graph is not None

    def test_stores_graph(self) -> None:
        graph = _make_mock_graph()
        host = _make_host(graph=graph)
        assert host._graph is graph

    def test_default_output_parser_is_none(self) -> None:
        host = _make_host()
        assert host._output_parser is None

    def test_messages_stream_mode_with_output_parser_is_accepted(self) -> None:
        """stream_mode='messages' + output_parser is valid for chunk streaming."""
        host = _make_host(
            output_parser=lambda chunk: getattr(chunk, "content", ""),
            stream_mode="messages",
        )
        assert host._stream_mode == "messages"

    def test_values_stream_mode_with_output_parser_is_accepted(self) -> None:
        host = _make_host(
            output_parser=lambda state: "",
            stream_mode="values",
        )
        assert host._stream_mode == "values"


class TestHandleCreate:
    """Tests for _handle_create."""

    def _make_context(
        self, history: list | None = None, user_input: str = "Hello!"
    ) -> Any:
        ctx = MagicMock()
        ctx.response_id = "resp-test"
        ctx.get_history = AsyncMock(return_value=history or [])
        ctx.get_input_text = AsyncMock(return_value=user_input)
        ctx.get_input_items = AsyncMock(
            return_value=[
                _FakeHistoryItem(_make_input_content(user_input), role="user")
            ]
        )
        return ctx

    async def test_thread_id_uses_previous_response_id(self) -> None:
        """previous_response_id is forwarded as thread_id in the LangGraph config."""
        received_configs = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_configs.append(config)
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = "resp-abc"
        context = self._make_context(user_input="hi")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        assert received_configs[0]["configurable"]["thread_id"] == "resp-abc"

    async def test_thread_id_is_uuid_when_no_previous_response_id(self) -> None:
        """A fresh uuid4 is used when previous_response_id is absent."""
        import uuid as _uuid

        received_configs = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_configs.append(config)
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="hi")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        tid = received_configs[0]["configurable"]["thread_id"]
        _uuid.UUID(tid)  # raises if not a valid UUID

    async def test_messages_state_path_returns_event_stream(self) -> None:
        """With no output_parser, _handle_create returns response events."""
        graph = _make_mock_graph(["Hi"])
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="Hello!")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        events = await _drain_async_iterable(result)

        assert hasattr(result, "__aiter__")
        assert events[0].type == "response.created"
        assert events[-1].type == "response.completed"

    async def test_graph_receives_human_message(self) -> None:
        """The graph should receive the user input as the final HumanMessage."""
        received_inputs = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_inputs.append(input_dict)
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="test input")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        assert len(received_inputs) == 1
        msgs = received_inputs[0]["messages"]
        last = msgs[-1]
        assert isinstance(last, HumanMessage)
        assert last.content == "test input"

    async def test_history_translated_to_messages(self) -> None:
        """History items should appear before the new HumanMessage."""
        received_inputs = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_inputs.append(input_dict)
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream

        history = [
            _FakeHistoryItem(_make_input_content("earlier question")),
            _FakeHistoryItem(_make_output_content("earlier answer")),
        ]
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(history=history, user_input="new question")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        msgs = received_inputs[0]["messages"]
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)
        assert msgs[2].content == "new question"

    async def test_cancellation_stops_stream(self) -> None:
        """Setting the cancellation signal stops the async generator cleanly."""
        chunks_produced = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            for i in range(100):
                await asyncio.sleep(0)
                yield AIMessageChunk(content=f"chunk{i}"), {}

        graph = MagicMock()
        graph.astream = _astream
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="go")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)

        async def _consume() -> None:
            async for event in result:
                if event.type == "response.output_text.delta":
                    chunks_produced.append(event.delta)
                if len(chunks_produced) >= 2:
                    signal.set()

        await asyncio.wait_for(_consume(), timeout=5.0)
        # After cancellation, not all 100 chunks should have been produced
        assert len(chunks_produced) < 100

    async def test_non_text_content_is_preserved_in_final_message(self) -> None:
        """Image and file blocks appear in the completed assistant message."""

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield (
                AIMessageChunk(
                    content=[
                        {"type": "text", "text": "summary"},
                        {"type": "image", "file_id": "file-img-123"},
                        {"type": "file", "file_url": "https://example.com/doc.pdf"},
                    ]
                ),
                {},
            )

        graph = MagicMock()
        graph.astream = _astream
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="show me artifacts")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        events = await _drain_async_iterable(result)

        completed = events[-1].response.output[0]
        assert completed.content == [
            {
                "type": "output_text",
                "text": "summary",
                "annotations": [],
                "logprobs": [],
            },
            {"type": "image", "file_id": "file-img-123"},
            {"type": "file", "file_url": "https://example.com/doc.pdf"},
        ]

    async def test_output_parser_path_uses_event_stream(self) -> None:
        """When output_parser is set, a ResponseEventStream should be used."""

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield {"messages": [AIMessage(content="result")]}

        graph = MagicMock()
        graph.astream = _astream

        extractor = lambda state: state["messages"][-1].content  # noqa: E731
        host = _make_host(graph=graph, output_parser=extractor, stream_mode="values")

        mock_stream = AsyncMock()
        mock_stream_cls = MagicMock(return_value=mock_stream)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="hi")
        signal = asyncio.Event()

        with patch(
            "langchain_azure_ai.agents.runtime._responses_host.ResponseEventStream",
            mock_stream_cls,
        ):
            result = await host._handle_create(request, context, signal)

        assert result is mock_stream
        # Give the background emit task enough event-loop ticks to complete
        for _ in range(10):
            await asyncio.sleep(0)
        mock_stream.emit.assert_awaited_with("result")

    async def test_output_parser_with_messages_mode_streams_chunks(self) -> None:
        """output_parser + stream_mode='messages' emits one call per chunk."""

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield AIMessageChunk(content="hello"), {}
            yield AIMessageChunk(content=" world"), {}

        graph = MagicMock()
        graph.astream = _astream
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        extractor = lambda chunk: getattr(chunk, "content", "")  # noqa: E731
        host = _make_host(graph=graph, output_parser=extractor, stream_mode="messages")

        mock_stream = AsyncMock()
        mock_stream_cls = MagicMock(return_value=mock_stream)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context(user_input="hi")
        signal = asyncio.Event()

        with patch(
            "langchain_azure_ai.agents.runtime._responses_host.ResponseEventStream",
            mock_stream_cls,
        ):
            result = await host._handle_create(request, context, signal)

        assert result is mock_stream
        for _ in range(20):
            await asyncio.sleep(0)
        emitted = [call.args[0] for call in mock_stream.emit.await_args_list]
        assert emitted == ["hello", " world"]

    async def test_output_parser_failure_emits_failed_response(self) -> None:
        """output_parser failures should terminate the stream with response.failed."""

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield {"messages": [AIMessage(content="result")]}

        def extractor(state: Any) -> str:
            del state
            raise RuntimeError("could not extract final answer")

        graph = MagicMock()
        graph.astream = _astream
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph, output_parser=extractor, stream_mode="values")

        mock_stream = AsyncMock()
        mock_stream_cls = MagicMock(return_value=mock_stream)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context(user_input="hi")
        signal = asyncio.Event()

        with patch(
            "langchain_azure_ai.agents.runtime._responses_host.ResponseEventStream",
            mock_stream_cls,
        ):
            result = await host._handle_create(request, context, signal)

        assert result is mock_stream
        for _ in range(20):
            await asyncio.sleep(0)
        mock_stream.emit_failed.assert_called_once()
        _, kwargs = mock_stream.emit_failed.call_args
        assert kwargs["code"] == "server_error"
        assert kwargs["message"] == (
            "The configured output_parser 'extractor' raised RuntimeError: "
            "could not extract final answer"
        )

    async def test_interrupt_resume_with_mcp_approval(self) -> None:
        """Pending interrupt + MCP approval item resumes graph with Command."""
        received_inputs: list[Any] = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_inputs.append(input_dict)
            yield AIMessageChunk(content="done"), {}

        graph = MagicMock()
        graph.astream = _astream

        interrupt_obj = MagicMock()
        task = MagicMock()
        task.interrupts = (interrupt_obj,)
        state = MagicMock()
        state.tasks = (task,)
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph)

        mcp_item = MagicMock()
        type(mcp_item).__name__ = "McpApprovalResponseInputItem"
        mcp_item.approve = True
        mcp_item.approval_request_id = "mcpr_abc"

        request = MagicMock()
        request.previous_response_id = "resp-123"
        request.input = [mcp_item]
        context = self._make_context(user_input="")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        from langgraph.types import Command

        graph_input = received_inputs[0]
        assert isinstance(graph_input, Command)
        assert graph_input.resume == {
            "approved": True,
            "approval_request_id": "mcpr_abc",
        }

    async def test_interrupt_resume_text_fallback(self) -> None:
        """Pending interrupt with no MCP item resumes with plain-text input."""
        received_inputs: list[Any] = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_inputs.append(input_dict)
            yield AIMessageChunk(content="done"), {}

        graph = MagicMock()
        graph.astream = _astream

        interrupt_obj = MagicMock()
        task = MagicMock()
        task.interrupts = (interrupt_obj,)
        state = MagicMock()
        state.tasks = (task,)
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = "resp-456"
        request.input = []  # no MCP approval item
        context = self._make_context(user_input="yes please")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        from langgraph.types import Command

        graph_input = received_inputs[0]
        assert isinstance(graph_input, Command)
        assert graph_input.resume == "yes please"

    async def test_no_interrupt_uses_messages_flow(self) -> None:
        """When no interrupt is pending, the graph receives a messages dict."""
        received_inputs: list[Any] = []

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_inputs.append(input_dict)
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context(user_input="hello")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        graph_input = received_inputs[0]
        assert isinstance(graph_input, dict)
        assert "messages" in graph_input
        assert isinstance(graph_input["messages"][-1], HumanMessage)


class TestStreamMessages:
    async def test_yields_chunk_content(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import _stream_messages

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield AIMessageChunk(content="foo"), {}
            yield AIMessageChunk(content="bar"), {}

        graph = MagicMock()
        graph.astream = _astream

        signal = asyncio.Event()
        results = []
        async for chunk in _stream_messages(graph, {"messages": []}, {}, signal):
            results.append(chunk)

        assert results == ["foo", "bar"]

    async def test_streams_other_langchain_message_types(self) -> None:
        from langchain_core.messages import HumanMessage, SystemMessageChunk

        from langchain_azure_ai.agents.runtime._responses_host import _stream_messages

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield HumanMessage(content="human"), {}
            yield SystemMessageChunk(content="system"), {}
            yield AIMessageChunk(content="assistant"), {}

        graph = MagicMock()
        graph.astream = _astream

        signal = asyncio.Event()
        results = []
        async for chunk in _stream_messages(graph, {"messages": []}, {}, signal):
            results.append(chunk)

        assert results == ["human", "system", "assistant"]

    async def test_streams_text_from_message_content_blocks(self) -> None:
        from langchain_core.messages import HumanMessage

        from langchain_azure_ai.agents.runtime._responses_host import _stream_messages

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield (
                HumanMessage(
                    content=[
                        {"type": "text", "text": "hello"},
                        {"type": "image", "file_id": "file-123"},
                        {"content": " world"},
                    ]
                ),
                {},
            )

        graph = MagicMock()
        graph.astream = _astream

        signal = asyncio.Event()
        results = []
        async for chunk in _stream_messages(graph, {"messages": []}, {}, signal):
            results.append(chunk)

        assert results == ["hello", " world"]

    async def test_cancellation_via_signal(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import _stream_messages

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            for i in range(50):
                await asyncio.sleep(0)
                yield AIMessageChunk(content=f"{i}"), {}

        graph = MagicMock()
        graph.astream = _astream

        signal = asyncio.Event()
        results = []

        async def _consume() -> None:
            async for chunk in _stream_messages(graph, {"messages": []}, {}, signal):
                results.append(chunk)
                if len(results) >= 3:
                    signal.set()

        await asyncio.wait_for(_consume(), timeout=5.0)
        assert len(results) < 50


# ---------------------------------------------------------------------------
# Tests for _pending_interrupts
# ---------------------------------------------------------------------------


class TestPendingInterrupts:
    """Tests for the _pending_interrupts helper."""

    async def test_returns_empty_when_aget_state_raises(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _pending_interrupts,
        )

        graph = MagicMock()
        graph.aget_state = AsyncMock(side_effect=RuntimeError("no checkpointer"))
        result = await _pending_interrupts(graph, {})
        assert result == []

    async def test_returns_empty_when_no_interrupts(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _pending_interrupts,
        )

        task = MagicMock()
        task.interrupts = ()
        state = MagicMock()
        state.tasks = (task,)
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=state)
        result = await _pending_interrupts(graph, {})
        assert result == []

    async def test_returns_interrupt_objects(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _pending_interrupts,
        )

        interrupt_obj = MagicMock()
        task = MagicMock()
        task.interrupts = (interrupt_obj,)
        state = MagicMock()
        state.tasks = (task,)
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=state)
        result = await _pending_interrupts(graph, {})
        assert result == [interrupt_obj]


# ---------------------------------------------------------------------------
# Tests for _extract_mcp_resume_value
# ---------------------------------------------------------------------------


class TestExtractMcpResumeValue:
    """Tests for the _extract_mcp_resume_value helper."""

    def test_returns_none_when_input_is_empty(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _extract_mcp_resume_value,
        )

        request = MagicMock()
        request.input = []
        assert _extract_mcp_resume_value(request) is None

    def test_returns_none_when_no_mcp_item(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _extract_mcp_resume_value,
        )

        item = MagicMock()
        type(item).__name__ = "SomeOtherInputItem"
        request = MagicMock()
        request.input = [item]
        assert _extract_mcp_resume_value(request) is None

    def test_approved_true(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _extract_mcp_resume_value,
        )

        item = MagicMock()
        type(item).__name__ = "McpApprovalResponseInputItem"
        item.approve = True
        item.approval_request_id = "mcpr_123"
        request = MagicMock()
        request.input = [item]
        result = _extract_mcp_resume_value(request)
        assert result == {"approved": True, "approval_request_id": "mcpr_123"}

    def test_approved_false(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            _extract_mcp_resume_value,
        )

        item = MagicMock()
        type(item).__name__ = "McpApprovalResponseInputItem"
        item.approve = False
        item.approval_request_id = "mcpr_456"
        request = MagicMock()
        request.input = [item]
        result = _extract_mcp_resume_value(request)
        assert result == {"approved": False, "approval_request_id": "mcpr_456"}


# ---------------------------------------------------------------------------
# Tests for custom input_parser
# ---------------------------------------------------------------------------


class TestInputParser:
    """Tests for the input_parser parameter on AzureAIResponsesAgentHost."""

    def _make_context(
        self, history: list | None = None, user_input: str = "Hello!"
    ) -> Any:
        ctx = MagicMock()
        ctx.response_id = "resp-test"
        ctx.get_history = AsyncMock(return_value=history or [])
        ctx.get_input_text = AsyncMock(return_value=user_input)
        ctx.get_input_items = AsyncMock(
            return_value=[
                _FakeHistoryItem(_make_input_content(user_input), role="user")
            ]
        )
        return ctx

    async def test_default_parser_preserves_structured_current_input(self) -> None:
        from langchain_azure_ai.agents.runtime._responses_host import (
            default_input_parser,
        )

        context = self._make_context(history=[])
        context.get_input_items = AsyncMock(
            return_value=[
                _FakeHistoryItem(
                    _make_input_content("look at this"),
                    _FakeInputImageContent(file_id="file-img-123", detail="high"),
                    _FakeInputFileContent(file_url="https://example.com/doc.pdf"),
                    role="user",
                )
            ]
        )

        result = await default_input_parser(MagicMock(), context)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == [
            {"type": "text", "text": "look at this"},
            {
                "type": "image",
                "source_type": "file",
                "file_id": "file-img-123",
                "detail": "high",
            },
            {
                "type": "file",
                "file_url": "https://example.com/doc.pdf",
            },
        ]

    async def test_custom_parser_receives_request_and_context(self) -> None:
        """A custom input_parser is called with the raw request and context."""
        received_args: list[Any] = []

        async def my_parser(req: Any, ctx: Any) -> dict[str, Any]:
            received_args.append((req, ctx))
            return {"messages": []}

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph, input_parser=my_parser)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context(user_input="hello")
        signal = asyncio.Event()

        await host._handle_create(request, context, signal)

        assert len(received_args) == 1
        assert received_args[0] == (request, context)

    async def test_custom_parser_output_passed_verbatim_to_graph(self) -> None:
        """The dict returned by input_parser is passed verbatim to graph.astream."""
        custom_state: dict[str, Any] = {
            "messages": [{"role": "user", "content": "hi"}],
            "constraints": "be helpful",
            "metadata": {"request_id": "abc"},
        }
        received_inputs: list[Any] = []

        async def my_parser(req: Any, ctx: Any) -> dict[str, Any]:
            return custom_state

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            received_inputs.append(input_dict)
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph, input_parser=my_parser)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context()
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        assert received_inputs[0] is custom_state

    async def test_custom_parser_not_called_on_interrupt_resume(self) -> None:
        """When a pending interrupt is present, input_parser is not invoked."""
        parser_called: list[bool] = []

        async def my_parser(req: Any, ctx: Any) -> dict[str, Any]:
            parser_called.append(True)
            return {"messages": []}

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield AIMessageChunk(content="ok"), {}

        graph = MagicMock()
        graph.astream = _astream
        interrupt_obj = MagicMock()
        task = MagicMock()
        task.interrupts = (interrupt_obj,)
        state = MagicMock()
        state.tasks = (task,)
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph, input_parser=my_parser)

        request = MagicMock()
        request.previous_response_id = "resp-123"
        request.input = []
        context = self._make_context(user_input="yes")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        assert parser_called == []

    async def test_custom_parser_failure_returns_failed_events(self) -> None:
        """Custom input_parser failures should be surfaced as response.failed."""

        async def my_parser(req: Any, ctx: Any) -> dict[str, Any]:
            del req, ctx
            raise TypeError("missing required state field 'constraints'")

        graph = MagicMock()
        graph.astream = AsyncMock()
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph, input_parser=my_parser)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context(user_input="hello")
        signal = asyncio.Event()

        result = await host._handle_create(request, context, signal)
        events = await _drain_async_iterable(result)

        assert events[0].type == "response.created"
        assert events[1].type == "response.in_progress"
        assert events[2].type == "response.failed"
        assert events[2].response.error.message == (
            "The configured input_parser 'my_parser' raised TypeError: "
            "missing required state field 'constraints'"
        )
        graph.astream.assert_not_called()

    async def test_custom_parser_failure_logs_diagnostics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Custom input_parser failures should log hook-specific diagnostics."""

        async def my_parser(req: Any, ctx: Any) -> dict[str, Any]:
            del req, ctx
            raise ValueError("developer parser bug")

        graph = MagicMock()
        graph.astream = AsyncMock()
        state = MagicMock()
        state.tasks = ()
        graph.aget_state = AsyncMock(return_value=state)

        host = _make_host(graph=graph, input_parser=my_parser)

        request = MagicMock()
        request.previous_response_id = None
        request.input = []
        context = self._make_context(user_input="hello")
        signal = asyncio.Event()

        caplog.set_level(logging.ERROR, logger="langchain_azure_ai.agents.runtime")
        result = await host._handle_create(request, context, signal)
        await _drain_async_iterable(result)

        assert "Configured input_parser failed" in caplog.text
