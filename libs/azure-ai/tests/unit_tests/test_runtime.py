# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for LangGraphAgentServerHost and helpers."""

import asyncio
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


class _FakeHistoryItem:
    """Mimics a history item with a content list."""

    def __init__(self, *content_blocks: Any) -> None:
        self.content = list(content_blocks)


@pytest.fixture(autouse=True, scope="module")
def _patch_content_types() -> Any:
    """Globally replace SDK content types with the test fakes for this module."""
    with (
        patch(
            "langchain_azure_ai.agents.runtime._host.MessageContentInputTextContent",
            _FakeInputTextContent,
        ),
        patch(
            "langchain_azure_ai.agents.runtime._host.MessageContentOutputTextContent",
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
        from langchain_azure_ai.agents.runtime._host import _history_to_messages

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
    return graph


def _make_host(graph: Any = None, **kwargs: Any) -> Any:
    """Instantiate LangGraphAgentServerHost with SDK imports mocked out."""
    if graph is None:
        graph = _make_mock_graph()

    mock_host_cls = MagicMock()
    mock_host_instance = MagicMock()
    mock_host_cls.return_value = mock_host_instance

    mock_text_response_cls = MagicMock(
        side_effect=lambda ctx, req, text: ("TextResponse", text)
    )

    with (
        patch(
            "langchain_azure_ai.agents.runtime._host.ResponsesAgentServerHost",
            mock_host_cls,
        ),
        patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ),
        patch(
            "langchain_azure_ai.agents.runtime._host._AGENTSERVER_IMPORT_ERROR",
            None,
        ),
    ):
        from langchain_azure_ai.agents.runtime._host import AzureAIResponsesAgentHost

        host = AzureAIResponsesAgentHost(graph=graph, **kwargs)
        host._app = mock_host_instance
        return host


class TestLangGraphAgentServerHostInit:
    """Tests for __init__ and import-guard."""

    def test_raises_import_error_when_sdk_missing(self) -> None:
        with patch(
            "langchain_azure_ai.agents.runtime._host._AGENTSERVER_IMPORT_ERROR",
            ImportError("missing"),
        ):
            import langchain_azure_ai.agents.runtime._host as mod

            original = mod._AGENTSERVER_IMPORT_ERROR
            mod._AGENTSERVER_IMPORT_ERROR = ImportError("missing")
            try:
                with pytest.raises(ImportError, match="missing"):
                    mod.AzureAIResponsesAgentHost(graph=MagicMock())
            finally:
                mod._AGENTSERVER_IMPORT_ERROR = original

    def test_stores_graph(self) -> None:
        graph = _make_mock_graph()
        host = _make_host(graph=graph)
        assert host._graph is graph

    def test_default_output_extractor_is_none(self) -> None:
        host = _make_host()
        assert host._output_extractor is None

    def test_messages_stream_mode_with_output_extractor_raises(self) -> None:
        with pytest.raises(ValueError, match="stream_mode='messages' is incompatible"):
            _make_host(
                output_extractor=lambda state: "",
                stream_mode="messages",
            )

    def test_values_stream_mode_with_output_extractor_is_accepted(self) -> None:
        host = _make_host(
            output_extractor=lambda state: "",
            stream_mode="values",
        )
        assert host._stream_mode == "values"


class TestHandleCreate:
    """Tests for _handle_create."""

    def _make_context(
        self, history: list | None = None, user_input: str = "Hello!"
    ) -> Any:
        ctx = MagicMock()
        ctx.get_history = AsyncMock(return_value=history or [])
        ctx.get_input_text = AsyncMock(return_value=user_input)
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

        mock_text_response_cls = MagicMock(return_value="resp")
        with patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ):
            await host._handle_create(request, context, signal)

        _, call_kwargs = mock_text_response_cls.call_args
        async for _ in call_kwargs["text"]:
            pass

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

        mock_text_response_cls = MagicMock(return_value="resp")
        with patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ):
            await host._handle_create(request, context, signal)

        _, call_kwargs = mock_text_response_cls.call_args
        async for _ in call_kwargs["text"]:
            pass

        tid = received_configs[0]["configurable"]["thread_id"]
        _uuid.UUID(tid)  # raises if not a valid UUID

    async def test_messages_state_path_returns_text_response(self) -> None:
        """With no output_extractor, _handle_create returns a TextResponse."""
        graph = _make_mock_graph(["Hi"])
        host = _make_host(graph=graph)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="Hello!")
        signal = asyncio.Event()

        mock_text_response_cls = MagicMock(return_value="mocked-text-response")
        with patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ):
            await host._handle_create(request, context, signal)

        mock_text_response_cls.assert_called_once()
        # Third kwarg should be 'text=' (the async generator)
        _, call_kwargs = mock_text_response_cls.call_args
        assert "text" in call_kwargs

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

        mock_text_response_cls = MagicMock(return_value="resp")
        with patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ):
            await host._handle_create(request, context, signal)

        # Drain the async generator so the producer task runs
        _, call_kwargs = mock_text_response_cls.call_args
        gen = call_kwargs["text"]
        async for _ in gen:
            pass

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

        mock_text_response_cls = MagicMock(return_value="resp")
        with patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ):
            await host._handle_create(request, context, signal)

        _, call_kwargs = mock_text_response_cls.call_args
        gen = call_kwargs["text"]
        async for _ in gen:
            pass

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

        mock_text_response_cls = MagicMock(return_value="resp")
        with patch(
            "langchain_azure_ai.agents.runtime._host.TextResponse",
            mock_text_response_cls,
        ):
            await host._handle_create(request, context, signal)

        _, call_kwargs = mock_text_response_cls.call_args
        gen = call_kwargs["text"]

        async def _consume() -> None:
            async for chunk in gen:
                chunks_produced.append(chunk)
                if len(chunks_produced) >= 2:
                    signal.set()

        await asyncio.wait_for(_consume(), timeout=5.0)
        # After cancellation, not all 100 chunks should have been produced
        assert len(chunks_produced) < 100

    async def test_output_extractor_path_uses_event_stream(self) -> None:
        """When output_extractor is set, a ResponseEventStream should be used."""

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield {"messages": [AIMessage(content="result")]}

        graph = MagicMock()
        graph.astream = _astream

        extractor = lambda state: state["messages"][-1].content  # noqa: E731
        host = _make_host(graph=graph, output_extractor=extractor, stream_mode="values")

        mock_stream = AsyncMock()
        mock_stream_cls = MagicMock(return_value=mock_stream)

        request = MagicMock()
        request.previous_response_id = None
        context = self._make_context(user_input="hi")
        signal = asyncio.Event()

        with patch(
            "langchain_azure_ai.agents.runtime._host.ResponseEventStream",
            mock_stream_cls,
        ):
            result = await host._handle_create(request, context, signal)

        assert result is mock_stream
        # Give the background emit task enough event-loop ticks to complete
        for _ in range(10):
            await asyncio.sleep(0)
        mock_stream.emit.assert_awaited_with("result")


# ---------------------------------------------------------------------------
# Tests for _stream_messages (streaming helper)
# ---------------------------------------------------------------------------


class TestStreamMessages:
    async def test_yields_chunk_content(self) -> None:
        from langchain_azure_ai.agents.runtime._host import _stream_messages

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield AIMessageChunk(content="foo"), {}
            yield AIMessageChunk(content="bar"), {}

        graph = MagicMock()
        graph.astream = _astream

        signal = asyncio.Event()
        results = []
        async for chunk in _stream_messages(graph, [], {}, signal):
            results.append(chunk)

        assert results == ["foo", "bar"]

    async def test_skips_non_ai_message_chunks(self) -> None:
        from langchain_core.messages import HumanMessage

        from langchain_azure_ai.agents.runtime._host import _stream_messages

        async def _astream(
            input_dict: Any, config: Any = None, *, stream_mode: Any = None
        ) -> Any:
            yield HumanMessage(content="ignored"), {}
            yield AIMessageChunk(content="kept"), {}

        graph = MagicMock()
        graph.astream = _astream

        signal = asyncio.Event()
        results = []
        async for chunk in _stream_messages(graph, [], {}, signal):
            results.append(chunk)

        assert results == ["kept"]

    async def test_cancellation_via_signal(self) -> None:
        from langchain_azure_ai.agents.runtime._host import _stream_messages

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
            async for chunk in _stream_messages(graph, [], {}, signal):
                results.append(chunk)
                if len(results) >= 3:
                    signal.set()

        await asyncio.wait_for(_consume(), timeout=5.0)
        assert len(results) < 50
