"""Unit tests for the guardrails module."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_middleware_cls(*, has_before: bool = False, has_after: bool = False):
    """Return an AgentMiddleware subclass that optionally overrides hooks."""
    try:
        from langchain.agents.middleware.types import AgentMiddleware
    except ImportError:
        pytest.skip("langchain not available", allow_module_level=True)

    class _Middleware(AgentMiddleware):
        call_log: List[str] = []

        @property
        def name(self) -> str:
            return "TestMiddleware"

        if has_before:

            def before_agent(  # type: ignore[override]
                self, state: Dict[str, Any]
            ) -> None:
                self.call_log.append("before")
                return None

        if has_after:

            def after_agent(  # type: ignore[override]
                self, state: Dict[str, Any]
            ) -> None:
                self.call_log.append("after")
                return None

    return _Middleware


# ---------------------------------------------------------------------------
# Tests for apply_middleware
# ---------------------------------------------------------------------------


class TestApplyMiddleware:
    """Tests for :func:`langchain_azure_ai.guardrails.apply_middleware`."""

    def _import(self):  # type: ignore[return]
        try:
            from langchain_azure_ai.guardrails._middleware import apply_middleware

            return apply_middleware
        except ImportError:
            pytest.skip("langgraph not available")

    def test_no_middleware_returns_agent_node_and_end(self) -> None:
        """With empty middleware the entry is agent_node and exit is END."""
        apply_middleware = self._import()
        from langgraph.graph import END, MessagesState, StateGraph

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [], agent_node="agent")

        assert entry == "agent"
        assert after is END

    def test_before_agent_only_entry_is_middleware_node(self) -> None:
        """entry_node should be the before_agent node when middleware has one."""
        apply_middleware = self._import()
        from langgraph.graph import END, MessagesState, StateGraph

        cls = _make_middleware_cls(has_before=True)
        m = cls()

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [m], agent_node="agent")

        assert entry == "TestMiddleware.before_agent"
        assert after is END
        assert "TestMiddleware.before_agent" in builder.nodes

    def test_after_agent_only_exit_is_middleware_node(self) -> None:
        """after_agent_entry should be the after_agent node when middleware has one."""
        apply_middleware = self._import()
        from langgraph.graph import MessagesState, StateGraph

        cls = _make_middleware_cls(has_after=True)
        m = cls()

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [m], agent_node="agent")

        assert entry == "agent"
        assert after == "TestMiddleware.after_agent"
        assert "TestMiddleware.after_agent" in builder.nodes

    def test_both_hooks_entry_and_exit_set(self) -> None:
        """Both entry and exit should be middleware nodes when both hooks exist."""
        apply_middleware = self._import()
        from langgraph.graph import MessagesState, StateGraph

        cls = _make_middleware_cls(has_before=True, has_after=True)
        m = cls()

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [m], agent_node="agent")

        assert entry == "TestMiddleware.before_agent"
        assert after == "TestMiddleware.after_agent"

    def test_multiple_before_agent_chained(self) -> None:
        """Multiple before_agent middleware should be chained correctly."""
        apply_middleware = self._import()
        from langgraph.graph import END, MessagesState, StateGraph

        try:
            from langchain.agents.middleware.types import AgentMiddleware
        except ImportError:
            pytest.skip("langchain not available")

        class Ma(AgentMiddleware):
            @property
            def name(self) -> str:
                return "Ma"

            def before_agent(  # type: ignore[override]
                self, state: Dict[str, Any]
            ) -> None:
                return None

        class Mb(AgentMiddleware):
            @property
            def name(self) -> str:
                return "Mb"

            def before_agent(  # type: ignore[override]
                self, state: Dict[str, Any]
            ) -> None:
                return None

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [Ma(), Mb()], agent_node="agent")

        # Entry is the FIRST before_agent node
        assert entry == "Ma.before_agent"
        # No after_agent hooks → after is END
        assert after is END
        # Both nodes present
        assert "Ma.before_agent" in builder.nodes
        assert "Mb.before_agent" in builder.nodes

    def test_multiple_after_agent_exit_is_last(self) -> None:
        """The after-chain entry should be the LAST middleware's after_agent node."""
        apply_middleware = self._import()
        from langgraph.graph import MessagesState, StateGraph

        try:
            from langchain.agents.middleware.types import AgentMiddleware
        except ImportError:
            pytest.skip("langchain not available")

        class Ma(AgentMiddleware):
            @property
            def name(self) -> str:
                return "Ma"

            def after_agent(  # type: ignore[override]
                self, state: Dict[str, Any]
            ) -> None:
                return None

        class Mb(AgentMiddleware):
            @property
            def name(self) -> str:
                return "Mb"

            def after_agent(  # type: ignore[override]
                self, state: Dict[str, Any]
            ) -> None:
                return None

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [Ma(), Mb()], agent_node="agent")

        # The after-chain entry exposed to the agent is the LAST middleware
        assert after == "Mb.after_agent"
        assert "Ma.after_agent" in builder.nodes
        assert "Mb.after_agent" in builder.nodes

    def test_input_schema_forwarded_to_nodes(self) -> None:
        """input_schema kwarg should be passed when adding nodes."""
        apply_middleware = self._import()
        from langgraph.graph import MessagesState, StateGraph

        cls = _make_middleware_cls(has_before=True)
        m = cls()

        added_kwargs: Dict[str, Any] = {}
        original_add_node = StateGraph.add_node

        def spy_add_node(self_inner, node_name, node_val, **kwargs):  # type: ignore[no-untyped-def]
            added_kwargs[node_name] = kwargs
            return original_add_node(self_inner, node_name, node_val, **kwargs)

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        with patch.object(StateGraph, "add_node", spy_add_node):
            apply_middleware(
                builder, [m], agent_node="agent", input_schema=MessagesState
            )

        assert "TestMiddleware.before_agent" in added_kwargs
        assert added_kwargs["TestMiddleware.before_agent"]["input_schema"] is MessagesState


# ---------------------------------------------------------------------------
# Tests for AzureContentSafetyMiddleware
# ---------------------------------------------------------------------------


class TestAzureContentSafetyMiddlewareInit:
    """Tests for AzureContentSafetyMiddleware instantiation."""

    def _make(self, **kwargs: Any) -> Any:
        with patch.dict("sys.modules", {"azure.ai.contentsafety": MagicMock()}):
            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                **kwargs,
            )

    def test_default_name(self) -> None:
        """Default name should be 'azure_content_safety'."""
        m = self._make()
        assert m.name == "azure_content_safety"

    def test_custom_name(self) -> None:
        """Custom name should be respected."""
        m = self._make(name="my_safety")
        assert m.name == "my_safety"

    def test_default_categories(self) -> None:
        """Default categories cover all four harm types."""
        m = self._make()
        assert set(m._categories) == {"Hate", "SelfHarm", "Sexual", "Violence"}

    def test_custom_categories(self) -> None:
        """Custom categories list should be used."""
        m = self._make(categories=["Hate", "Violence"])
        assert m._categories == ["Hate", "Violence"]

    def test_missing_endpoint_raises(self) -> None:
        """ValueError raised when no endpoint is provided and env var absent."""
        with patch.dict("sys.modules", {"azure.ai.contentsafety": MagicMock()}):
            import os

            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            env_backup = os.environ.pop("AZURE_CONTENT_SAFETY_ENDPOINT", None)
            try:
                with pytest.raises(ValueError, match="endpoint"):
                    AzureContentSafetyMiddleware(credential="fake-key")
            finally:
                if env_backup is not None:
                    os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"] = env_backup

    def test_endpoint_from_env(self) -> None:
        """Endpoint falls back to AZURE_CONTENT_SAFETY_ENDPOINT env var."""
        import os

        with patch.dict("sys.modules", {"azure.ai.contentsafety": MagicMock()}):
            with patch.dict(
                os.environ,
                {"AZURE_CONTENT_SAFETY_ENDPOINT": "https://env.cognitiveservices.azure.com/"},
            ):
                from langchain_azure_ai.guardrails._content_safety import (
                    AzureContentSafetyMiddleware,
                )

                m = AzureContentSafetyMiddleware(credential="fake-key")
                assert m._endpoint == "https://env.cognitiveservices.azure.com/"

    def test_missing_sdk_raises_import_error(self) -> None:
        """ImportError raised when azure-ai-contentsafety is not installed."""
        import sys

        saved = sys.modules.pop("azure.ai.contentsafety", None)
        try:
            # Remove any cached module to simulate missing package
            import importlib

            import langchain_azure_ai.guardrails._content_safety as cs_mod

            importlib.reload(cs_mod)

            with patch.dict(sys.modules, {"azure.ai.contentsafety": None}):  # type: ignore[dict-item]
                with pytest.raises(ImportError, match="azure-ai-contentsafety"):
                    cs_mod.AzureContentSafetyMiddleware(
                        endpoint="https://test.cognitiveservices.azure.com/",
                        credential="fake-key",
                    )
        finally:
            if saved is not None:
                sys.modules["azure.ai.contentsafety"] = saved

    def test_state_schema_is_content_safety_state(self) -> None:
        """state_schema should be _ContentSafetyState (includes violation field)."""
        from langchain_azure_ai.guardrails._content_safety import (
            _ContentSafetyState,
        )

        m = self._make()
        assert m.state_schema is _ContentSafetyState

    def test_tools_is_empty_list(self) -> None:
        """tools attribute should default to an empty list."""
        m = self._make()
        assert m.tools == []


# ---------------------------------------------------------------------------
# Tests for message text extraction helpers
# ---------------------------------------------------------------------------


class TestMessageTextExtraction:
    """Tests for _extract_human_text and _extract_ai_text helpers."""

    def _cls(self) -> Any:
        with patch.dict("sys.modules", {"azure.ai.contentsafety": MagicMock()}):
            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware

    def test_extract_human_text_string_content(self) -> None:
        """String HumanMessage content is returned directly."""
        cls = self._cls()
        msgs = [HumanMessage(content="hello world")]
        assert cls._extract_human_text(msgs) == "hello world"

    def test_extract_human_text_list_content(self) -> None:
        """Text blocks in list content are joined."""
        cls = self._cls()
        msgs = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "foo"},
                    {"type": "image_url", "image_url": "http://x.com/img.png"},
                    {"type": "text", "text": "bar"},
                ]
            )
        ]
        assert cls._extract_human_text(msgs) == "foo bar"

    def test_extract_human_text_most_recent(self) -> None:
        """Only the most recent HumanMessage is used."""
        cls = self._cls()
        msgs = [
            HumanMessage(content="first"),
            AIMessage(content="reply"),
            HumanMessage(content="second"),
        ]
        assert cls._extract_human_text(msgs) == "second"

    def test_extract_human_text_no_human_message(self) -> None:
        """Returns None when no HumanMessage present."""
        cls = self._cls()
        msgs = [AIMessage(content="hi")]
        assert cls._extract_human_text(msgs) is None

    def test_extract_ai_text_string_content(self) -> None:
        """String AIMessage content is returned directly."""
        cls = self._cls()
        msgs = [AIMessage(content="answer")]
        assert cls._extract_ai_text(msgs) == "answer"

    def test_extract_ai_text_most_recent(self) -> None:
        """Only the most recent AIMessage is used."""
        cls = self._cls()
        msgs = [
            AIMessage(content="first answer"),
            HumanMessage(content="follow-up"),
            AIMessage(content="second answer"),
        ]
        assert cls._extract_ai_text(msgs) == "second answer"

    def test_extract_ai_text_no_ai_message(self) -> None:
        """Returns None when no AIMessage present."""
        cls = self._cls()
        msgs = [HumanMessage(content="question")]
        assert cls._extract_ai_text(msgs) is None


# ---------------------------------------------------------------------------
# Tests for _handle_violations
# ---------------------------------------------------------------------------


class TestHandleViolations:
    """Tests for AzureContentSafetyMiddleware._handle_violations."""

    def _instance(self, action: str = "block") -> Any:
        with patch.dict("sys.modules", {"azure.ai.contentsafety": MagicMock()}):
            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                action=action,  # type: ignore[arg-type]
            )

    def test_no_violations_returns_none(self) -> None:
        """No violations should always return None regardless of action."""
        for action in ("block", "warn", "flag"):
            m = self._instance(action=action)
            result = m._handle_violations([], "test")
            assert result is None

    def test_block_raises_violation_error(self) -> None:
        """action='block' should raise ContentSafetyViolationError."""
        from langchain_azure_ai.guardrails._content_safety import (
            ContentSafetyViolationError,
        )

        m = self._instance(action="block")
        violations = [{"category": "Hate", "severity": 6}]
        with pytest.raises(ContentSafetyViolationError) as exc_info:
            m._handle_violations(violations, "input")
        assert exc_info.value.violations == violations
        assert "Hate" in str(exc_info.value)

    def test_warn_returns_none_and_logs(self) -> None:
        """action='warn' should log a warning and return None."""
        import logging

        m = self._instance(action="warn")
        violations = [{"category": "Violence", "severity": 4}]
        with patch.object(
            logging.getLogger("langchain_azure_ai.guardrails._content_safety"),
            "warning",
        ) as mock_warn:
            result = m._handle_violations(violations, "output")
        assert result is None
        mock_warn.assert_called_once()

    def test_flag_returns_state_dict(self) -> None:
        """action='flag' should return a dict with content_safety_violations."""
        m = self._instance(action="flag")
        violations = [{"category": "Sexual", "severity": 6}]
        result = m._handle_violations(violations, "output")
        assert result is not None
        assert result["content_safety_violations"] == violations


# ---------------------------------------------------------------------------
# Tests for before_agent / after_agent (sync) with mocked SDK
# ---------------------------------------------------------------------------


class TestBeforeAfterAgentSync:
    """Tests for synchronous before_agent and after_agent hooks."""

    def _make_middleware(
        self,
        action: str = "block",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
    ) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                action=action,  # type: ignore[arg-type]
                apply_to_input=apply_to_input,
                apply_to_output=apply_to_output,
            )

    @staticmethod
    def _mock_sdk() -> Any:
        """Return a context manager that mocks the contentsafety SDK modules."""
        mock_models = MagicMock()
        mock_models.AnalyzeTextOptions = MagicMock(return_value=MagicMock())
        mock_models.TextCategory = MagicMock(side_effect=lambda x: x)
        mock_sdk = MagicMock()
        mock_sdk.models = mock_models
        return patch.dict(
            "sys.modules",
            {
                "azure.ai.contentsafety": mock_sdk,
                "azure.ai.contentsafety.models": mock_models,
            },
        )

    def _mock_response(self, severity: int) -> MagicMock:
        cat = MagicMock()
        cat.category = "Hate"
        cat.severity = severity
        response = MagicMock()
        response.categories_analysis = [cat]
        response.blocklists_match = []
        return response

    def test_before_agent_block_raises(self) -> None:
        """before_agent with 'block' raises on high-severity input."""
        from langchain_azure_ai.guardrails._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(action="block")
            mock_client = MagicMock()
            mock_client.analyze_text.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [HumanMessage(content="bad content")]}
                with pytest.raises(ContentSafetyViolationError):
                    m.before_agent(state)

    def test_before_agent_warn_returns_none(self) -> None:
        """before_agent with 'warn' returns None even on violation."""
        with self._mock_sdk():
            m = self._make_middleware(action="warn")
            mock_client = MagicMock()
            mock_client.analyze_text.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [HumanMessage(content="bad content")]}
                result = m.before_agent(state)
        assert result is None

    def test_before_agent_flag_returns_dict(self) -> None:
        """before_agent with 'flag' returns state dict on violation."""
        with self._mock_sdk():
            m = self._make_middleware(action="flag")
            mock_client = MagicMock()
            mock_client.analyze_text.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [HumanMessage(content="bad content")]}
                result = m.before_agent(state)
        assert result is not None
        assert "content_safety_violations" in result

    def test_before_agent_no_violation_returns_none(self) -> None:
        """before_agent returns None when severity is below threshold."""
        with self._mock_sdk():
            m = self._make_middleware(action="block")
            mock_client = MagicMock()
            # Severity 0 – no violation
            mock_client.analyze_text.return_value = self._mock_response(severity=0)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [HumanMessage(content="hello")]}
                result = m.before_agent(state)
        assert result is None

    def test_before_agent_skipped_when_apply_to_input_false(self) -> None:
        """before_agent is a no-op when apply_to_input=False."""
        m = self._make_middleware(apply_to_input=False)
        state = {"messages": [HumanMessage(content="bad")]}
        # No client needed – should short-circuit
        result = m.before_agent(state)
        assert result is None

    def test_after_agent_block_raises(self) -> None:
        """after_agent with 'block' raises on high-severity AI output."""
        from langchain_azure_ai.guardrails._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(action="block")
            mock_client = MagicMock()
            mock_client.analyze_text.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [AIMessage(content="harmful reply")]}
                with pytest.raises(ContentSafetyViolationError):
                    m.after_agent(state)

    def test_after_agent_skipped_when_apply_to_output_false(self) -> None:
        """after_agent is a no-op when apply_to_output=False."""
        m = self._make_middleware(apply_to_output=False)
        state = {"messages": [AIMessage(content="bad")]}
        result = m.after_agent(state)
        assert result is None

    def test_before_agent_empty_messages_returns_none(self) -> None:
        """before_agent returns None gracefully on empty message list."""
        m = self._make_middleware()
        result = m.before_agent({"messages": []})
        assert result is None

    def test_after_agent_no_ai_message_returns_none(self) -> None:
        """after_agent returns None gracefully when no AIMessage present."""
        m = self._make_middleware()
        state = {"messages": [HumanMessage(content="question")]}
        result = m.after_agent(state)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for async hooks
# ---------------------------------------------------------------------------


class TestBeforeAfterAgentAsync:
    """Tests for asynchronous abefore_agent and aafter_agent hooks."""

    @staticmethod
    def _mock_sdk() -> Any:
        mock_models = MagicMock()
        mock_models.AnalyzeTextOptions = MagicMock(return_value=MagicMock())
        mock_models.TextCategory = MagicMock(side_effect=lambda x: x)
        mock_sdk = MagicMock()
        mock_sdk.models = mock_models
        return patch.dict(
            "sys.modules",
            {
                "azure.ai.contentsafety": mock_sdk,
                "azure.ai.contentsafety.models": mock_models,
            },
        )

    def _make_middleware(self, action: str = "block") -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                action=action,  # type: ignore[arg-type]
            )

    def _mock_async_response(self, severity: int) -> MagicMock:
        cat = MagicMock()
        cat.category = "Violence"
        cat.severity = severity
        response = MagicMock()
        response.categories_analysis = [cat]
        response.blocklists_match = []
        return response

    async def test_abefore_agent_block_raises(self) -> None:
        """abefore_agent with 'block' raises on violation."""
        from langchain_azure_ai.guardrails._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(action="block")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_text = AsyncMock(
                return_value=self._mock_async_response(severity=6)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                state = {"messages": [HumanMessage(content="violent content")]}
                with pytest.raises(ContentSafetyViolationError):
                    await m.abefore_agent(state)

    async def test_abefore_agent_no_violation_returns_none(self) -> None:
        """abefore_agent returns None when no violations found."""
        with self._mock_sdk():
            m = self._make_middleware(action="block")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_text = AsyncMock(
                return_value=self._mock_async_response(severity=0)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                state = {"messages": [HumanMessage(content="safe content")]}
                result = await m.abefore_agent(state)
        assert result is None

    async def test_aafter_agent_flag_returns_dict(self) -> None:
        """aafter_agent with 'flag' returns violation dict."""
        with self._mock_sdk():
            m = self._make_middleware(action="flag")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_text = AsyncMock(
                return_value=self._mock_async_response(severity=6)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                state = {"messages": [AIMessage(content="flagged output")]}
                result = await m.aafter_agent(state)
        assert result is not None
        assert "content_safety_violations" in result


# ---------------------------------------------------------------------------
# Tests for public imports from guardrails package
# ---------------------------------------------------------------------------


class TestGuardrailsPublicAPI:
    """Tests for public imports from langchain_azure_ai.guardrails."""

    def test_apply_middleware_importable(self) -> None:
        """apply_middleware should be importable from the guardrails package."""
        from langchain_azure_ai.guardrails import apply_middleware

        assert callable(apply_middleware)

    def test_content_safety_violation_error_importable(self) -> None:
        """ContentSafetyViolationError should be importable."""
        from langchain_azure_ai.guardrails import ContentSafetyViolationError

        assert issubclass(ContentSafetyViolationError, ValueError)

    def test_azure_content_safety_middleware_importable(self) -> None:
        """AzureContentSafetyMiddleware should be importable."""
        with patch.dict(
            "sys.modules", {"azure.ai.contentsafety": MagicMock()}
        ):
            from langchain_azure_ai.guardrails import AzureContentSafetyMiddleware

            assert AzureContentSafetyMiddleware is not None

    def test_unknown_attr_raises_attribute_error(self) -> None:
        """Accessing an unknown attribute on the package raises AttributeError."""
        import langchain_azure_ai.guardrails as g

        with pytest.raises(AttributeError):
            _ = g.NonExistentClass  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Integration-style test: apply_middleware with AzureContentSafetyMiddleware
# ---------------------------------------------------------------------------


class TestApplyMiddlewareWithContentSafety:
    """Verify apply_middleware wires AzureContentSafetyMiddleware into a graph."""

    def test_graph_has_before_and_after_nodes(self) -> None:
        """Compiled graph should contain the middleware nodes."""
        try:
            from langchain.agents.middleware.types import AgentMiddleware  # noqa: F401
            from langgraph.graph import END, START, MessagesState, StateGraph
        except ImportError:
            pytest.skip("langgraph / langchain not available")

        from langchain_azure_ai.guardrails._middleware import apply_middleware

        with patch.dict("sys.modules", {"azure.ai.contentsafety": MagicMock()}):
            from langchain_azure_ai.guardrails._content_safety import (
                AzureContentSafetyMiddleware,
            )

            safety = AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
            )

        builder = StateGraph(MessagesState)
        builder.add_node("agent", lambda s: s)

        entry, after = apply_middleware(builder, [safety], agent_node="agent")

        builder.add_edge(START, entry)
        builder.add_edge("agent", after)

        graph = builder.compile()
        node_names = set(graph.nodes.keys())

        assert "azure_content_safety.before_agent" in node_names
        assert "azure_content_safety.after_agent" in node_names
        assert "agent" in node_names
