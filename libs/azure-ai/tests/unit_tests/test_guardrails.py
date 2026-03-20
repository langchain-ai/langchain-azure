"""Unit tests for the guardrails module."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_middleware_cls(*, has_before: bool = False, has_after: bool = False) -> type:
    """Return an AgentMiddleware subclass that optionally overrides hooks."""
    from langchain.agents.middleware.types import AgentMiddleware

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
# Tests for AzureContentSafetyMiddleware
# ---------------------------------------------------------------------------


class TestAzureContentSafetyMiddlewareInit:
    """Tests for AzureContentSafetyMiddleware instantiation."""

    def _make(self, **kwargs: Any) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
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
        """ValueError raised when no endpoint is provided and env vars absent."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            import os

            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyMiddleware,
            )

            env_backup_cs = os.environ.pop("AZURE_CONTENT_SAFETY_ENDPOINT", None)
            env_backup_proj = os.environ.pop("AZURE_AI_PROJECT_ENDPOINT", None)
            try:
                with pytest.raises(ValueError, match="endpoint"):
                    AzureContentSafetyMiddleware(credential="fake-key")
            finally:
                if env_backup_cs is not None:
                    os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"] = env_backup_cs
                if env_backup_proj is not None:
                    os.environ["AZURE_AI_PROJECT_ENDPOINT"] = env_backup_proj

    def test_endpoint_from_env(self) -> None:
        """Endpoint falls back to AZURE_CONTENT_SAFETY_ENDPOINT env var."""
        import os

        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            with patch.dict(
                os.environ,
                {
                    "AZURE_CONTENT_SAFETY_ENDPOINT": (
                        "https://env.cognitiveservices.azure.com/"
                    )
                },
            ):
                from langchain_azure_ai.agents.middleware._content_safety import (
                    AzureContentSafetyMiddleware,
                )

                m = AzureContentSafetyMiddleware(credential="fake-key")
                assert m._endpoint == "https://env.cognitiveservices.azure.com/"

    def test_project_endpoint_extracts_base_url(self) -> None:
        """project_endpoint extracts the base resource URL."""
        m = self._make(
            endpoint=None,
            project_endpoint=(
                "https://myres.services.ai.azure.com/api/projects/myproj"
            ),
        )
        assert m._endpoint == "https://myres.services.ai.azure.com"

    def test_project_endpoint_from_env(self) -> None:
        """Endpoint falls back to AZURE_AI_PROJECT_ENDPOINT env var."""
        import os

        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            with patch.dict(
                os.environ,
                {
                    "AZURE_AI_PROJECT_ENDPOINT": (
                        "https://myres.services.ai.azure.com/api/projects/myproj"
                    )
                },
            ):
                from langchain_azure_ai.agents.middleware._content_safety import (
                    AzureContentSafetyMiddleware,
                )

                m = AzureContentSafetyMiddleware(credential="fake-key")
                assert m._endpoint == "https://myres.services.ai.azure.com"

    def test_both_endpoint_and_project_endpoint_raises(self) -> None:
        """ValueError raised when both endpoint and project_endpoint are given."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyMiddleware,
            )

            with pytest.raises(ValueError, match="mutually exclusive"):
                AzureContentSafetyMiddleware(
                    endpoint="https://test.cognitiveservices.azure.com/",
                    credential="fake-key",
                    project_endpoint=(
                        "https://res.services.ai.azure.com/api/projects/proj"
                    ),
                )

    def test_invalid_project_endpoint_raises(self) -> None:
        """ValueError raised when project_endpoint has no /api/projects/ path."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyMiddleware,
            )

            with pytest.raises(ValueError, match="does not look like"):
                AzureContentSafetyMiddleware(
                    credential="fake-key",
                    project_endpoint="https://bad-endpoint.azure.com/",
                )

    def test_missing_sdk_raises_import_error(self) -> None:
        """ImportError raised when azure-ai-contentsafety is not installed."""
        import sys

        saved = sys.modules.pop("azure.ai.contentsafety", None)
        try:
            # Remove any cached module to simulate missing package
            import importlib

            import langchain_azure_ai.agents.middleware._content_safety as cs_mod

            importlib.reload(cs_mod)

            with patch.dict(  # type: ignore[dict-item]
                sys.modules, {"azure.ai.contentsafety": None}
            ):
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
        from langchain_azure_ai.agents.middleware._content_safety import (
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
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
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

    def _instance(self, exit_behavior: str = "error", **kwargs: Any) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
                **kwargs,
            )

    def test_no_violations_returns_none(self) -> None:
        """No violations should always return None regardless of exit_behavior."""
        for eb in ("error", "continue"):
            m = self._instance(exit_behavior=eb)
            result = m._handle_violations([], "test")
            assert result is None

    def test_error_raises_violation_error(self) -> None:
        """exit_behavior='error' should raise ContentSafetyViolationError."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        m = self._instance(exit_behavior="error")
        violations = [{"category": "Hate", "severity": 6}]
        with pytest.raises(ContentSafetyViolationError) as exc_info:
            m._handle_violations(violations, "input")
        assert exc_info.value.violations == violations
        assert "Hate" in str(exc_info.value)

    def test_continue_replaces_message_and_logs(self) -> None:
        """exit_behavior='continue' should log, replace message, and return state."""
        import logging

        m = self._instance(exit_behavior="continue")
        violations = [{"category": "Violence", "severity": 4}]
        offending = HumanMessage(content="bad content", id="msg-1")
        with patch.object(
            logging.getLogger("langchain_azure_ai.agents.middleware._content_safety"),
            "warning",
        ) as mock_warn:
            result = m._handle_violations(violations, "output", offending)
        mock_warn.assert_called_once()
        assert result is not None
        assert "messages" in result
        replacement = result["messages"][0]
        assert isinstance(replacement, HumanMessage)
        assert replacement.id == "msg-1"
        assert "Violence" in replacement.content

    def test_continue_uses_custom_violation_message(self) -> None:
        """exit_behavior='continue' with violation_message uses custom text."""
        m = self._instance(
            exit_behavior="continue",
            violation_message="This content was blocked.",
        )
        violations = [{"category": "Sexual", "severity": 6}]
        offending = AIMessage(content="bad output", id="msg-2")
        result = m._handle_violations(violations, "output", offending)
        assert result is not None
        replacement = result["messages"][0]
        assert isinstance(replacement, AIMessage)
        assert replacement.content == "This content was blocked."

    def test_continue_without_offending_message_returns_none(self) -> None:
        """exit_behavior='continue' without offending_message returns None."""
        m = self._instance(exit_behavior="continue")
        violations = [{"category": "Hate", "severity": 6}]
        result = m._handle_violations(violations, "output")
        assert result is None


# ---------------------------------------------------------------------------
# Tests for before_agent / after_agent (sync) with mocked SDK
# ---------------------------------------------------------------------------


class TestBeforeAfterAgentSync:
    """Tests for synchronous before_agent and after_agent hooks."""

    def _make_middleware(
        self,
        exit_behavior: str = "error",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
    ) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
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
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": mock_sdk,
                "azure.ai.contentsafety.models": mock_models,
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
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
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
            mock_client = MagicMock()
            mock_client.analyze_text.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [HumanMessage(content="bad content")]}
                with pytest.raises(ContentSafetyViolationError):
                    m.before_agent(state)

    def test_before_agent_continue_replaces_message(self) -> None:
        """before_agent with 'continue' replaces offending HumanMessage."""
        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="continue")
            mock_client = MagicMock()
            mock_client.analyze_text.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                state = {"messages": [HumanMessage(content="bad content")]}
                result = m.before_agent(state)
        assert result is not None
        assert "messages" in result
        assert "Hate" in result["messages"][0].content

    def test_before_agent_no_violation_returns_none(self) -> None:
        """before_agent returns None when severity is below threshold."""
        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
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
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
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
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": mock_sdk,
                "azure.ai.contentsafety.models": mock_models,
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make_middleware(self, exit_behavior: str = "error") -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyMiddleware,
            )

            return AzureContentSafetyMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
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
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
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
            m = self._make_middleware(exit_behavior="error")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_text = AsyncMock(
                return_value=self._mock_async_response(severity=0)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                state = {"messages": [HumanMessage(content="safe content")]}
                result = await m.abefore_agent(state)
        assert result is None

    async def test_aafter_agent_continue_replaces_message(self) -> None:
        """aafter_agent with 'continue' replaces offending AIMessage."""
        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="continue")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_text = AsyncMock(
                return_value=self._mock_async_response(severity=6)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                state = {"messages": [AIMessage(content="flagged output")]}
                result = await m.aafter_agent(state)
        assert result is not None
        assert "messages" in result
        assert "Violence" in result["messages"][0].content


# ---------------------------------------------------------------------------
# Tests for canonical agents.middleware public API
# ---------------------------------------------------------------------------


class TestAgentMiddlewarePublicAPI:
    """Tests for public imports from langchain_azure_ai.agents.middleware."""

    def test_apply_middleware_not_in_public_api(self) -> None:
        """apply_middleware must NOT be part of the public agents.middleware API."""
        import langchain_azure_ai.agents.middleware as m

        with pytest.raises(AttributeError):
            _ = m.apply_middleware  # type: ignore[attr-defined]

    def test_content_safety_violation_error_importable(self) -> None:
        """ContentSafetyViolationError should be importable."""
        from langchain_azure_ai.agents.middleware import ContentSafetyViolationError

        assert issubclass(ContentSafetyViolationError, ValueError)

    def test_azure_content_safety_middleware_importable(self) -> None:
        """AzureContentSafetyMiddleware should be importable."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware import (
                AzureContentSafetyMiddleware,
            )

            assert AzureContentSafetyMiddleware is not None

    def test_azure_content_safety_image_middleware_importable(self) -> None:
        """AzureContentSafetyImageMiddleware should be importable."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware import (
                AzureContentSafetyImageMiddleware,
            )

            assert AzureContentSafetyImageMiddleware is not None

    def test_azure_protected_material_middleware_importable(self) -> None:
        """AzureProtectedMaterialMiddleware should be importable."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware import (
                AzureProtectedMaterialMiddleware,
            )

            assert AzureProtectedMaterialMiddleware is not None

    def test_azure_prompt_shield_middleware_importable(self) -> None:
        """AzurePromptShieldMiddleware should be importable."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware import AzurePromptShieldMiddleware

            assert AzurePromptShieldMiddleware is not None

    def test_unknown_attr_raises_attribute_error(self) -> None:
        """Accessing an unknown attribute on the package raises AttributeError."""
        import langchain_azure_ai.agents.middleware as m

        with pytest.raises(AttributeError):
            _ = m.NonExistentClass  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests for AzureContentSafetyImageMiddleware
# ---------------------------------------------------------------------------


class TestAzureContentSafetyImageMiddlewareInit:
    """Tests for AzureContentSafetyImageMiddleware instantiation."""

    def _make(self, **kwargs: Any) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyImageMiddleware,
            )

            return AzureContentSafetyImageMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                **kwargs,
            )

    def test_default_name(self) -> None:
        """Default name should be 'azure_content_safety_image'."""
        m = self._make()
        assert m.name == "azure_content_safety_image"

    def test_custom_name(self) -> None:
        """Custom name should be respected."""
        m = self._make(name="my_image_safety")
        assert m.name == "my_image_safety"

    def test_default_categories(self) -> None:
        """Default categories cover all four harm types."""
        m = self._make()
        assert set(m._categories) == {"Hate", "SelfHarm", "Sexual", "Violence"}

    def test_apply_to_output_false_by_default(self) -> None:
        """apply_to_output defaults to False for image middleware."""
        m = self._make()
        assert m.apply_to_output is False

    def test_apply_to_input_true_by_default(self) -> None:
        """apply_to_input defaults to True for image middleware."""
        m = self._make()
        assert m.apply_to_input is True

    def test_state_schema_matches_text_middleware(self) -> None:
        """Both middleware classes share the same _ContentSafetyState schema."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            _ContentSafetyState,
        )

        m = self._make()
        assert m.state_schema is _ContentSafetyState

    def test_tools_is_empty_list(self) -> None:
        """tools attribute should default to an empty list."""
        m = self._make()
        assert m.tools == []


# ---------------------------------------------------------------------------
# Tests for image extraction helpers
# ---------------------------------------------------------------------------


class TestImageExtraction:
    """Tests for _images_from_message and extract_images helpers."""

    def _cls(self) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyImageMiddleware,
            )

            return AzureContentSafetyImageMiddleware

    def test_base64_data_url_decoded_to_bytes(self) -> None:
        """Base64 data URLs should be decoded and returned as bytes."""
        import base64 as b64_mod

        cls = self._cls()
        raw = b64_mod.b64encode(b"fake-image-bytes").decode()
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": f"data:image/png;base64,{raw}"},
            ]
        )
        images = cls._images_from_message(msg)
        assert len(images) == 1
        assert "content" in images[0]
        assert images[0]["content"] == b"fake-image-bytes"

    def test_https_url_returned_as_url(self) -> None:
        """HTTP(S) URLs should be returned as-is."""
        cls = self._cls()
        url = "https://example.com/photo.jpg"
        msg = HumanMessage(content=[{"type": "image_url", "image_url": url}])
        images = cls._images_from_message(msg)
        assert len(images) == 1
        assert images[0] == {"url": url}

    def test_dict_url_format_supported(self) -> None:
        """OpenAI-style dict image_url is also extracted."""
        cls = self._cls()
        url = "https://example.com/photo.jpg"
        msg = HumanMessage(content=[{"type": "image_url", "image_url": {"url": url}}])
        images = cls._images_from_message(msg)
        assert len(images) == 1
        assert images[0] == {"url": url}

    def test_non_image_blocks_skipped(self) -> None:
        """Text blocks should not appear in the image list."""
        cls = self._cls()
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": "https://example.com/img.png"},
            ]
        )
        images = cls._images_from_message(msg)
        assert len(images) == 1
        assert images[0] == {"url": "https://example.com/img.png"}

    def test_multiple_images_extracted(self) -> None:
        """Multiple image blocks from one message are all returned."""
        cls = self._cls()
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": "https://example.com/a.jpg"},
                {"type": "image_url", "image_url": "https://example.com/b.jpg"},
            ]
        )
        images = cls._images_from_message(msg)
        assert len(images) == 2

    def test_string_content_message_returns_empty(self) -> None:
        """String-content messages contain no images."""
        cls = self._cls()
        msg = HumanMessage(content="just text")
        images = cls._images_from_message(msg)
        assert images == []

    def test_extract_from_most_recent_human_message(self) -> None:
        """Only the most recent HumanMessage is inspected."""
        cls = self._cls()
        msgs = [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": "https://old.example.com/img.jpg",
                    }
                ]
            ),
            AIMessage(content="reply"),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": "https://new.example.com/img.jpg",
                    }
                ]
            ),
        ]
        images = cls._extract_images_from_last_human(msgs)
        assert len(images) == 1
        assert images[0]["url"] == "https://new.example.com/img.jpg"

    def test_no_human_message_returns_empty(self) -> None:
        """Returns empty list when no HumanMessage present."""
        cls = self._cls()
        msgs = [AIMessage(content="hi")]
        images = cls._extract_images_from_last_human(msgs)
        assert images == []


# ---------------------------------------------------------------------------
# Tests for AzureContentSafetyImageMiddleware sync hooks
# ---------------------------------------------------------------------------


class TestImageMiddlewareSync:
    """Tests for synchronous before_agent / after_agent on image middleware."""

    @staticmethod
    def _mock_sdk() -> Any:
        mock_models = MagicMock()
        mock_models.AnalyzeImageOptions = MagicMock(return_value=MagicMock())
        mock_models.ImageCategory = MagicMock(side_effect=lambda x: x)
        mock_models.ImageData = MagicMock(side_effect=lambda **kw: kw)
        mock_sdk = MagicMock()
        mock_sdk.models = mock_models
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": mock_sdk,
                "azure.ai.contentsafety.models": mock_models,
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make_middleware(self, exit_behavior: str = "error") -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyImageMiddleware,
            )

            return AzureContentSafetyImageMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
            )

    def _mock_response(self, severity: int) -> MagicMock:
        cat = MagicMock()
        cat.category = "Sexual"
        cat.severity = severity
        response = MagicMock()
        response.categories_analysis = [cat]
        return response

    def test_before_agent_block_raises_on_image_violation(self) -> None:
        """before_agent blocks a high-severity image."""
        import base64 as b64_mod

        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
            mock_client = MagicMock()
            mock_client.analyze_image.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                raw = b64_mod.b64encode(b"img").decode()
                msg = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{raw}",
                        }
                    ]
                )
                with pytest.raises(ContentSafetyViolationError):
                    m.before_agent({"messages": [msg]})

    def test_before_agent_no_images_returns_none(self) -> None:
        """before_agent is a no-op when the message contains no images."""
        m = self._make_middleware()
        msg = HumanMessage(content="text only")
        result = m.before_agent({"messages": [msg]})
        assert result is None

    def test_before_agent_skipped_when_apply_to_input_false(self) -> None:
        """before_agent is a no-op when apply_to_input=False."""
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyImageMiddleware,
            )

            m = AzureContentSafetyImageMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                apply_to_input=False,
            )
        result = m.before_agent({"messages": [HumanMessage(content="x")]})
        assert result is None

    def test_after_agent_skipped_by_default(self) -> None:
        """after_agent is a no-op by default (apply_to_output=False)."""
        m = self._make_middleware()
        result = m.after_agent({"messages": [AIMessage(content="y")]})
        assert result is None

    def test_before_agent_continue_replaces_message(self) -> None:
        """before_agent with 'continue' replaces offending message on image violation."""
        import base64 as b64_mod

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="continue")
            mock_client = MagicMock()
            mock_client.analyze_image.return_value = self._mock_response(severity=6)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                raw = b64_mod.b64encode(b"img").decode()
                msg = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{raw}",
                        }
                    ]
                )
                result = m.before_agent({"messages": [msg]})
        assert result is not None
        assert "messages" in result
        assert "Sexual" in result["messages"][0].content

    def test_before_agent_safe_image_returns_none(self) -> None:
        """before_agent returns None when image severity is below threshold."""
        import base64 as b64_mod

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
            mock_client = MagicMock()
            mock_client.analyze_image.return_value = self._mock_response(severity=0)
            with patch.object(m, "_get_sync_client", return_value=mock_client):
                raw = b64_mod.b64encode(b"img").decode()
                msg = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{raw}",
                        }
                    ]
                )
                result = m.before_agent({"messages": [msg]})
        assert result is None


# ---------------------------------------------------------------------------
# Tests for AzureContentSafetyImageMiddleware async hooks
# ---------------------------------------------------------------------------


class TestImageMiddlewareAsync:
    """Tests for async abefore_agent / aafter_agent on image middleware."""

    @staticmethod
    def _mock_sdk() -> Any:
        mock_models = MagicMock()
        mock_models.AnalyzeImageOptions = MagicMock(return_value=MagicMock())
        mock_models.ImageCategory = MagicMock(side_effect=lambda x: x)
        mock_models.ImageData = MagicMock(side_effect=lambda **kw: kw)
        mock_sdk = MagicMock()
        mock_sdk.models = mock_models
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": mock_sdk,
                "azure.ai.contentsafety.models": mock_models,
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make_middleware(self, exit_behavior: str = "error") -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureContentSafetyImageMiddleware,
            )

            return AzureContentSafetyImageMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
            )

    def _mock_async_response(self, severity: int) -> MagicMock:
        cat = MagicMock()
        cat.category = "Violence"
        cat.severity = severity
        response = MagicMock()
        response.categories_analysis = [cat]
        return response

    async def test_abefore_agent_block_raises(self) -> None:
        """abefore_agent raises on high-severity image."""
        import base64 as b64_mod

        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_image = AsyncMock(
                return_value=self._mock_async_response(severity=6)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                raw = b64_mod.b64encode(b"img").decode()
                msg = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{raw}",
                        }
                    ]
                )
                with pytest.raises(ContentSafetyViolationError):
                    await m.abefore_agent({"messages": [msg]})

    async def test_abefore_agent_safe_returns_none(self) -> None:
        """abefore_agent returns None when image is safe."""
        import base64 as b64_mod

        with self._mock_sdk():
            m = self._make_middleware(exit_behavior="error")
            mock_async_client = AsyncMock()
            mock_async_client.analyze_image = AsyncMock(
                return_value=self._mock_async_response(severity=0)
            )
            with patch.object(m, "_get_async_client", return_value=mock_async_client):
                raw = b64_mod.b64encode(b"img").decode()
                msg = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{raw}",
                        }
                    ]
                )
                result = await m.abefore_agent({"messages": [msg]})
        assert result is None


# ---------------------------------------------------------------------------
# Tests for public imports of new middleware classes
# ---------------------------------------------------------------------------


class TestNewMiddlewarePublicAPI:
    """Verify new middleware classes are importable."""

    def test_azure_protected_material_middleware_importable(self) -> None:
        """AzureProtectedMaterialMiddleware is importable."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware import (
                AzureProtectedMaterialMiddleware,
            )

            assert AzureProtectedMaterialMiddleware is not None

    def test_azure_prompt_shield_middleware_importable(self) -> None:
        """AzurePromptShieldMiddleware should be importable from agents.middleware."""
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware import AzurePromptShieldMiddleware

            assert AzurePromptShieldMiddleware is not None


# ---------------------------------------------------------------------------
# Tests for AzureProtectedMaterialMiddleware
# ---------------------------------------------------------------------------


class TestProtectedMaterialMiddlewareInit:
    """Tests for AzureProtectedMaterialMiddleware instantiation."""

    @staticmethod
    def _mock_sdk() -> Any:
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make(self, **kwargs: Any) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureProtectedMaterialMiddleware,
            )

            return AzureProtectedMaterialMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                **kwargs,
            )

    def test_default_name(self) -> None:
        """Default name should be 'azure_protected_material'."""
        m = self._make()
        assert m.name == "azure_protected_material"

    def test_custom_name(self) -> None:
        """Custom name is respected."""
        m = self._make(name="my_pm")
        assert m.name == "my_pm"

    def test_apply_to_input_true_by_default(self) -> None:
        """apply_to_input defaults to True."""
        m = self._make()
        assert m.apply_to_input is True

    def test_apply_to_output_true_by_default(self) -> None:
        """apply_to_output defaults to True."""
        m = self._make()
        assert m.apply_to_output is True

    def test_tools_is_empty(self) -> None:
        """tools attribute should be an empty list."""
        m = self._make()
        assert m.tools == []


class TestProtectedMaterialCollectViolations:
    """Unit tests for AzureProtectedMaterialMiddleware._collect_protected_violations."""

    def _cls(self) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureProtectedMaterialMiddleware,
            )

            return AzureProtectedMaterialMiddleware

    def test_not_detected_returns_empty(self) -> None:
        """_collect_protected_violations returns empty list when not detected."""
        cls = self._cls()
        response = {"protectedMaterialAnalysis": {"detected": False}}
        assert cls._collect_protected_violations(response) == []

    def test_detected_returns_violation(self) -> None:
        """_collect_protected_violations returns a violation dict when detected."""
        cls = self._cls()
        response = {"protectedMaterialAnalysis": {"detected": True}}
        violations = cls._collect_protected_violations(response)
        assert len(violations) == 1
        assert violations[0]["category"] == "ProtectedMaterial"
        assert violations[0]["detected"] is True

    def test_missing_analysis_attr_returns_empty(self) -> None:
        """Missing protectedMaterialAnalysis key is handled gracefully."""
        cls = self._cls()
        response: Dict[str, Any] = {}
        assert cls._collect_protected_violations(response) == []


class TestProtectedMaterialMiddlewareSync:
    """Sync hook tests for AzureProtectedMaterialMiddleware."""

    @staticmethod
    def _mock_sdk() -> Any:
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make(self, exit_behavior: str = "error", **kwargs: Any) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureProtectedMaterialMiddleware,
            )

            return AzureProtectedMaterialMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
                **kwargs,
            )

    def _response(self, detected: bool) -> Dict[str, Any]:
        return {"protectedMaterialAnalysis": {"detected": detected}}

    def test_before_agent_block_raises_when_detected(self) -> None:
        """before_agent raises when protected material is found."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(detected=True)
            ):
                with pytest.raises(ContentSafetyViolationError):
                    m.before_agent(
                        {"messages": [HumanMessage(content="song lyrics here")]}
                    )

    def test_before_agent_no_detection_returns_none(self) -> None:
        """before_agent returns None when nothing is detected."""
        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(detected=False)
            ):
                result = m.before_agent(
                    {"messages": [HumanMessage(content="safe input")]}
                )
        assert result is None

    def test_before_agent_continue_replaces_message(self) -> None:
        """before_agent with 'continue' replaces offending HumanMessage."""
        with self._mock_sdk():
            m = self._make(exit_behavior="continue")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(detected=True)
            ):
                result = m.before_agent(
                    {"messages": [HumanMessage(content="some lyrics")]}
                )
        assert result is not None
        assert "messages" in result
        assert "ProtectedMaterial" in result["messages"][0].content

    def test_after_agent_block_raises_when_detected(self) -> None:
        """after_agent raises when protected material is found in AI output."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(detected=True)
            ):
                with pytest.raises(ContentSafetyViolationError):
                    m.after_agent({"messages": [AIMessage(content="quote from book")]})

    def test_before_agent_skipped_when_apply_to_input_false(self) -> None:
        """before_agent is a no-op when apply_to_input=False."""
        m = self._make(apply_to_input=False)
        result = m.before_agent({"messages": [HumanMessage(content="x")]})
        assert result is None

    def test_after_agent_skipped_when_apply_to_output_false(self) -> None:
        """after_agent is a no-op when apply_to_output=False."""
        m = self._make(apply_to_output=False)
        result = m.after_agent({"messages": [AIMessage(content="y")]})
        assert result is None

    def test_before_agent_no_human_message_returns_none(self) -> None:
        """before_agent is a no-op when there is no HumanMessage."""
        m = self._make()
        result = m.before_agent({"messages": [AIMessage(content="hello")]})
        assert result is None


class TestProtectedMaterialMiddlewareAsync:
    """Async hook tests for AzureProtectedMaterialMiddleware."""

    @staticmethod
    def _mock_sdk() -> Any:
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make(self, exit_behavior: str = "error", **kwargs: Any) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzureProtectedMaterialMiddleware,
            )

            return AzureProtectedMaterialMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
                **kwargs,
            )

    def _response(self, detected: bool) -> Dict[str, Any]:
        return {"protectedMaterialAnalysis": {"detected": detected}}

    async def test_abefore_agent_block_raises(self) -> None:
        """abefore_agent raises when protected material is detected."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m,
                "_send_rest_async",
                new_callable=AsyncMock,
                return_value=self._response(detected=True),
            ):
                with pytest.raises(ContentSafetyViolationError):
                    await m.abefore_agent(
                        {"messages": [HumanMessage(content="lyrics")]}
                    )

    async def test_abefore_agent_safe_returns_none(self) -> None:
        """abefore_agent returns None when no protected material found."""
        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m,
                "_send_rest_async",
                new_callable=AsyncMock,
                return_value=self._response(detected=False),
            ):
                result = await m.abefore_agent(
                    {"messages": [HumanMessage(content="safe text")]}
                )
        assert result is None

    async def test_aafter_agent_continue_replaces_message(self) -> None:
        """aafter_agent with 'continue' replaces offending AIMessage."""
        with self._mock_sdk():
            m = self._make(exit_behavior="continue")
            with patch.object(
                m,
                "_send_rest_async",
                new_callable=AsyncMock,
                return_value=self._response(detected=True),
            ):
                result = await m.aafter_agent(
                    {"messages": [AIMessage(content="book excerpt")]}
                )
        assert result is not None
        assert "messages" in result
        assert "ProtectedMaterial" in result["messages"][0].content


# ---------------------------------------------------------------------------
# Tests for AzurePromptShieldMiddleware
# ---------------------------------------------------------------------------


class TestPromptShieldMiddlewareInit:
    """Tests for AzurePromptShieldMiddleware instantiation."""

    @staticmethod
    def _mock_sdk() -> Any:
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make(self, **kwargs: Any) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzurePromptShieldMiddleware,
            )

            return AzurePromptShieldMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                **kwargs,
            )

    def test_default_name(self) -> None:
        """Default name should be 'azure_prompt_shield'."""
        m = self._make()
        assert m.name == "azure_prompt_shield"

    def test_custom_name(self) -> None:
        """Custom name is respected."""
        m = self._make(name="my_shield")
        assert m.name == "my_shield"

    def test_apply_to_input_true_by_default(self) -> None:
        """apply_to_input defaults to True."""
        m = self._make()
        assert m.apply_to_input is True

    def test_apply_to_output_false_by_default(self) -> None:
        """apply_to_output defaults to False (injection is an input-side attack)."""
        m = self._make()
        assert m.apply_to_output is False

    def test_tools_is_empty(self) -> None:
        """tools attribute should be an empty list."""
        m = self._make()
        assert m.tools == []


class TestPromptShieldCollectViolations:
    """Unit tests for AzurePromptShieldMiddleware._collect_injection_violations."""

    def _cls(self) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzurePromptShieldMiddleware,
            )

            return AzurePromptShieldMiddleware

    def test_no_attack_returns_empty(self) -> None:
        """Returns empty list when no attack is detected."""
        cls = self._cls()
        response = {
            "userPromptAnalysis": {"attackDetected": False},
            "documentsAnalysis": [{"attackDetected": False}],
        }
        assert cls._collect_injection_violations(response) == []

    def test_user_prompt_attack_detected(self) -> None:
        """Returns a user_prompt violation when attackDetected is True."""
        cls = self._cls()
        response = {
            "userPromptAnalysis": {"attackDetected": True},
            "documentsAnalysis": [],
        }
        violations = cls._collect_injection_violations(response)
        assert len(violations) == 1
        assert violations[0]["category"] == "PromptInjection"
        assert violations[0]["source"] == "user_prompt"
        assert violations[0]["detected"] is True

    def test_document_attack_detected(self) -> None:
        """Returns a document violation when a document has attackDetected=True."""
        cls = self._cls()
        response = {
            "userPromptAnalysis": {"attackDetected": False},
            "documentsAnalysis": [
                {"attackDetected": False},
                {"attackDetected": True},
            ],
        }
        violations = cls._collect_injection_violations(response)
        assert len(violations) == 1
        assert violations[0]["source"] == "document[1]"

    def test_both_prompt_and_document_attacked(self) -> None:
        """Both user_prompt and document violations are returned."""
        cls = self._cls()
        response = {
            "userPromptAnalysis": {"attackDetected": True},
            "documentsAnalysis": [{"attackDetected": True}],
        }
        violations = cls._collect_injection_violations(response)
        assert len(violations) == 2
        sources = {v["source"] for v in violations}
        assert "user_prompt" in sources
        assert "document[0]" in sources

    def test_no_user_prompt_analysis_key(self) -> None:
        """Missing userPromptAnalysis key is handled gracefully."""
        cls = self._cls()
        response: Dict[str, Any] = {}
        assert cls._collect_injection_violations(response) == []

    def test_no_documents_analysis_returns_only_prompt_violation(self) -> None:
        """Missing documentsAnalysis does not cause an error."""
        cls = self._cls()
        response = {"userPromptAnalysis": {"attackDetected": False}}
        assert cls._collect_injection_violations(response) == []


class TestPromptShieldExtractToolTexts:
    """Unit tests for AzurePromptShieldMiddleware._extract_tool_texts."""

    def _cls(self) -> Any:
        with patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        ):
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzurePromptShieldMiddleware,
            )

            return AzurePromptShieldMiddleware

    def test_no_tool_messages_returns_empty(self) -> None:
        """Returns empty list when no ToolMessage items exist."""
        cls = self._cls()
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert cls._extract_tool_texts(msgs) == []

    def test_tool_message_text_extracted(self) -> None:
        """Text from ToolMessage items is returned."""
        from langchain_core.messages import ToolMessage

        cls = self._cls()
        msgs = [
            HumanMessage(content="search for X"),
            ToolMessage(content="result content", tool_call_id="1"),
        ]
        texts = cls._extract_tool_texts(msgs)
        assert texts == ["result content"]

    def test_multiple_tool_messages_all_extracted(self) -> None:
        """All ToolMessage texts are returned."""
        from langchain_core.messages import ToolMessage

        cls = self._cls()
        msgs = [
            ToolMessage(content="first result", tool_call_id="1"),
            ToolMessage(content="second result", tool_call_id="2"),
        ]
        texts = cls._extract_tool_texts(msgs)
        assert texts == ["first result", "second result"]

    def test_empty_tool_message_skipped(self) -> None:
        """Empty ToolMessage content is not included."""
        from langchain_core.messages import ToolMessage

        cls = self._cls()
        msgs = [ToolMessage(content="", tool_call_id="1")]
        texts = cls._extract_tool_texts(msgs)
        assert texts == []


class TestPromptShieldMiddlewareSync:
    """Sync hook tests for AzurePromptShieldMiddleware."""

    @staticmethod
    def _mock_sdk() -> Any:
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make(self, exit_behavior: str = "error", **kwargs: Any) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzurePromptShieldMiddleware,
            )

            return AzurePromptShieldMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
                **kwargs,
            )

    def _response(self, user_attacked: bool) -> Dict[str, Any]:
        return {
            "userPromptAnalysis": {"attackDetected": user_attacked},
            "documentsAnalysis": [],
        }

    def test_before_agent_block_raises_on_injection(self) -> None:
        """before_agent raises when a direct injection is detected."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(user_attacked=True)
            ):
                with pytest.raises(ContentSafetyViolationError):
                    m.before_agent(
                        {"messages": [HumanMessage(content="ignore all instructions")]}
                    )

    def test_before_agent_no_injection_returns_none(self) -> None:
        """before_agent returns None when no injection is found."""
        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(user_attacked=False)
            ):
                result = m.before_agent(
                    {"messages": [HumanMessage(content="safe prompt")]}
                )
        assert result is None

    def test_before_agent_continue_replaces_message(self) -> None:
        """before_agent with 'continue' replaces offending HumanMessage."""
        with self._mock_sdk():
            m = self._make(exit_behavior="continue")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(user_attacked=True)
            ):
                result = m.before_agent(
                    {"messages": [HumanMessage(content="inject here")]}
                )
        assert result is not None
        assert "messages" in result
        assert "PromptInjection" in result["messages"][0].content

    def test_after_agent_skipped_by_default(self) -> None:
        """after_agent is a no-op by default (apply_to_output=False)."""
        m = self._make()
        result = m.after_agent({"messages": [AIMessage(content="output")]})
        assert result is None

    def test_before_agent_skipped_when_apply_to_input_false(self) -> None:
        """before_agent is a no-op when apply_to_input=False."""
        m = self._make(apply_to_input=False)
        result = m.before_agent({"messages": [HumanMessage(content="x")]})
        assert result is None

    def test_before_agent_no_human_message_returns_none(self) -> None:
        """before_agent is a no-op when there is no HumanMessage."""
        m = self._make()
        result = m.before_agent({"messages": [AIMessage(content="hello")]})
        assert result is None

    def test_before_agent_passes_tool_messages_as_documents(self) -> None:
        """Tool message content is passed as documents to shieldPrompt API."""
        from langchain_core.messages import ToolMessage

        with self._mock_sdk():
            m = self._make(exit_behavior="continue")
            with patch.object(
                m, "_send_rest_sync", return_value=self._response(user_attacked=False)
            ) as mock_rest:
                msgs = [
                    HumanMessage(content="search for X"),
                    ToolMessage(content="malicious tool result", tool_call_id="1"),
                ]
                m.before_agent({"messages": msgs})
            assert mock_rest.called
            call_args = mock_rest.call_args
            body = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["body"]
            assert "documents" in body
            assert "malicious tool result" in body["documents"]


class TestPromptShieldMiddlewareAsync:
    """Async hook tests for AzurePromptShieldMiddleware."""

    @staticmethod
    def _mock_sdk() -> Any:
        return patch.dict(
            "sys.modules",
            {
                "azure": MagicMock(),
                "azure.ai": MagicMock(),
                "azure.ai.contentsafety": MagicMock(),
                "azure.core": MagicMock(),
                "azure.core.credentials": MagicMock(),
                "azure.identity": MagicMock(),
            },
        )

    def _make(self, exit_behavior: str = "error", **kwargs: Any) -> Any:
        with self._mock_sdk():
            from langchain_azure_ai.agents.middleware._content_safety import (
                AzurePromptShieldMiddleware,
            )

            return AzurePromptShieldMiddleware(
                endpoint="https://test.cognitiveservices.azure.com/",
                credential="fake-key",
                exit_behavior=exit_behavior,  # type: ignore[arg-type]
                **kwargs,
            )

    def _response(self, user_attacked: bool) -> Dict[str, Any]:
        return {
            "userPromptAnalysis": {"attackDetected": user_attacked},
            "documentsAnalysis": [],
        }

    async def test_abefore_agent_block_raises(self) -> None:
        """abefore_agent raises when injection is detected."""
        from langchain_azure_ai.agents.middleware._content_safety import (
            ContentSafetyViolationError,
        )

        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m,
                "_send_rest_async",
                new_callable=AsyncMock,
                return_value=self._response(user_attacked=True),
            ):
                with pytest.raises(ContentSafetyViolationError):
                    await m.abefore_agent(
                        {"messages": [HumanMessage(content="ignore instructions")]}
                    )

    async def test_abefore_agent_safe_returns_none(self) -> None:
        """abefore_agent returns None when no injection is detected."""
        with self._mock_sdk():
            m = self._make(exit_behavior="error")
            with patch.object(
                m,
                "_send_rest_async",
                new_callable=AsyncMock,
                return_value=self._response(user_attacked=False),
            ):
                result = await m.abefore_agent(
                    {"messages": [HumanMessage(content="safe prompt")]}
                )
        assert result is None

    async def test_aafter_agent_continue_replaces_message(self) -> None:
        """aafter_agent with 'continue' replaces offending AIMessage."""
        with self._mock_sdk():
            m = self._make(exit_behavior="continue", apply_to_output=True)
            with patch.object(
                m,
                "_send_rest_async",
                new_callable=AsyncMock,
                return_value=self._response(user_attacked=True),
            ):
                result = await m.aafter_agent(
                    {"messages": [AIMessage(content="injected output")]}
                )
        assert result is not None
        assert "messages" in result
        assert "PromptInjection" in result["messages"][0].content
