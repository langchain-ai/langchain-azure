"""Unit tests for the evaluation module and tracer evaluation events."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

pytest.importorskip("azure.monitor.opentelemetry")
pytest.importorskip("opentelemetry")
pytest.importorskip("langgraph")


# ============================================================
# Tests for converter.py
# ============================================================


class TestMessagesToFoundryFormat:
    """Test LangChain message conversion to Foundry format."""

    def test_empty_messages(self) -> None:
        from langchain_azure_ai.evaluation.converter import messages_to_foundry_format

        result = messages_to_foundry_format([])
        assert result["query"] == ""
        assert result["response"] == ""
        assert result["tool_definitions"] is None

    def test_simple_conversation(self) -> None:
        from langchain_azure_ai.evaluation.converter import messages_to_foundry_format

        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        result = messages_to_foundry_format(msgs)
        assert isinstance(result["query"], list) or isinstance(result["query"], str)
        assert isinstance(result["response"], list) or isinstance(
            result["response"], str
        )

    def test_system_and_user_in_query(self) -> None:
        from langchain_azure_ai.evaluation.converter import messages_to_foundry_format

        msgs = [
            SystemMessage(content="You are an analyst."),
            HumanMessage(content="Analyze X"),
            AIMessage(content="Analysis complete."),
        ]
        result = messages_to_foundry_format(msgs)
        if isinstance(result["query"], list):
            roles = [m["role"] for m in result["query"]]
            assert "system" in roles
            assert "user" in roles

    def test_tool_calls_in_response(self) -> None:
        from langchain_azure_ai.evaluation.converter import messages_to_foundry_format

        msgs = [
            HumanMessage(content="Get data"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_data", "args": {"key": "value"}, "id": "call_1"}
                ],
            ),
            ToolMessage(content='{"result": "data"}', tool_call_id="call_1"),
            AIMessage(content="Here's the data."),
        ]
        result = messages_to_foundry_format(msgs)
        assert isinstance(result["response"], list)
        assert len(result["response"]) >= 2

    def test_with_tool_definitions(self) -> None:
        from langchain_azure_ai.evaluation.converter import messages_to_foundry_format

        tool_defs = [{"name": "search", "description": "Search for info"}]
        msgs = [HumanMessage(content="Search"), AIMessage(content="Found it")]
        result = messages_to_foundry_format(msgs, tool_definitions=tool_defs)
        assert result["tool_definitions"] == tool_defs

    def test_multipart_content(self) -> None:
        from langchain_azure_ai.evaluation.converter import _text_content

        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        )
        text = _text_content(msg)
        assert "Hello" in text
        assert "World" in text


class TestToolSchemasToFoundryFormat:
    """Test tool schema conversion."""

    def test_basic_tool(self) -> None:
        from langchain_azure_ai.evaluation.converter import (
            tool_schemas_to_foundry_format,
        )

        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search for info"
        tool.args_schema = None
        tool.args = {"query": {"type": "string"}}

        result = tool_schemas_to_foundry_format([tool])
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search for info"

    def test_empty_tools(self) -> None:
        from langchain_azure_ai.evaluation.converter import (
            tool_schemas_to_foundry_format,
        )

        result = tool_schemas_to_foundry_format([])
        assert result == []


class TestConvertSingleMessage:
    """Test individual message conversion."""

    def test_system_message(self) -> None:
        from langchain_azure_ai.evaluation.converter import _convert_single_message

        msg = SystemMessage(content="Be helpful")
        result = _convert_single_message(msg, timestamp="2025-01-01T00:00:00Z")
        assert result is not None
        assert result["role"] == "system"
        assert result["content"] == "Be helpful"

    def test_human_message(self) -> None:
        from langchain_azure_ai.evaluation.converter import _convert_single_message

        msg = HumanMessage(content="Hello")
        result = _convert_single_message(msg, timestamp="2025-01-01T00:00:00Z")
        assert result is not None
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello"

    def test_ai_message_with_text(self) -> None:
        from langchain_azure_ai.evaluation.converter import _convert_single_message

        msg = AIMessage(content="Response text")
        result = _convert_single_message(msg, timestamp="2025-01-01T00:00:00Z")
        assert result is not None
        assert result["role"] == "assistant"
        text_parts = [p for p in result["content"] if p["type"] == "text"]
        assert len(text_parts) >= 1

    def test_ai_message_with_tool_calls(self) -> None:
        from langchain_azure_ai.evaluation.converter import _convert_single_message

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "func", "args": {"a": 1}, "id": "c1"}],
        )
        result = _convert_single_message(msg, timestamp="2025-01-01T00:00:00Z")
        assert result is not None
        assert result["role"] == "assistant"
        tool_parts = [p for p in result["content"] if p["type"] == "tool_call"]
        assert len(tool_parts) == 1
        assert tool_parts[0]["name"] == "func"

    def test_tool_message(self) -> None:
        from langchain_azure_ai.evaluation.converter import _convert_single_message

        msg = ToolMessage(content='{"key": "val"}', tool_call_id="c1")
        result = _convert_single_message(msg, timestamp="2025-01-01T00:00:00Z")
        assert result is not None
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "c1"

    def test_tool_message_non_json(self) -> None:
        from langchain_azure_ai.evaluation.converter import _convert_single_message

        msg = ToolMessage(content="plain text result", tool_call_id="c2")
        result = _convert_single_message(msg, timestamp="2025-01-01T00:00:00Z")
        assert result is not None
        assert result["role"] == "tool"
        assert result["content"][0]["tool_result"] == "plain text result"


# ============================================================
# Tests for foundry.py
# ============================================================


class TestFoundryEvalResult:
    """Test the FoundryEvalResult dataclass."""

    def test_creation(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvalResult

        result = FoundryEvalResult(
            evaluator_name="task_completion",
            passed=True,
            score=4.5,
            label="pass",
            explanation="Good job",
        )
        assert result.evaluator_name == "task_completion"
        assert result.passed is True
        assert result.score == 4.5
        assert result.label == "pass"

    def test_defaults(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvalResult

        result = FoundryEvalResult(evaluator_name="test", passed=False)
        assert result.score is None
        assert result.label is None
        assert result.explanation is None
        assert result.raw_output == {}


class TestFoundryEvaluator:
    """Test FoundryEvaluator (mocked — no network)."""

    def test_init(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluator

        with patch(
            "langchain_azure_ai.evaluation.foundry.FoundryEvaluator.__init__",
            return_value=None,
        ):
            evaluator = FoundryEvaluator.__new__(FoundryEvaluator)
            evaluator._project_endpoint = "https://test.endpoint"
            evaluator._evaluator_name = "builtin.task_completion"
            evaluator._deployment_name = "gpt-4o"
            evaluator._display_name = "task_completion"
            assert evaluator._display_name == "task_completion"

    def test_parse_result_failed_status(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluator

        evaluator = FoundryEvaluator.__new__(FoundryEvaluator)
        evaluator._display_name = "test_eval"
        result = evaluator._parse_result(MagicMock(), "eval_1", "run_1", "failed")
        assert result.passed is False
        assert result.label == "error"
        assert "failed" in (result.explanation or "").lower()

    def test_get_client_reuses_cached_clients(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluator

        mock_project_client = MagicMock()
        mock_openai_client = MagicMock()
        mock_project_client.get_openai_client.return_value = mock_openai_client

        evaluator = FoundryEvaluator.__new__(FoundryEvaluator)
        evaluator._project_endpoint = "https://test.endpoint"
        evaluator._credential = object()
        evaluator._project_client = None
        evaluator._openai_client = None
        evaluator._client_lock = threading.Lock()
        evaluator._eval_definition_lock = threading.Lock()

        with patch.dict(
            "sys.modules",
            {
                "azure.ai.projects": SimpleNamespace(
                    AIProjectClient=MagicMock(return_value=mock_project_client)
                ),
                "azure.core.pipeline.policies": SimpleNamespace(
                    UserAgentPolicy=MagicMock()
                ),
            },
        ):
            project_client_1, client_1 = evaluator._get_client()
            project_client_2, client_2 = evaluator._get_client()

        assert project_client_1 is project_client_2 is mock_project_client
        assert client_1 is client_2 is mock_openai_client
        mock_project_client.get_openai_client.assert_called_once()

    def test_get_client_locks_when_returning_cached_clients(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluator

        class CountingLock:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.enter_count = 0

            def __enter__(self) -> CountingLock:
                self.enter_count += 1
                self._lock.acquire()
                return self

            def __exit__(self, *args: object) -> Literal[False]:
                self._lock.release()
                return False

        evaluator = FoundryEvaluator.__new__(FoundryEvaluator)
        evaluator._project_client = object()
        evaluator._openai_client = object()
        counting_lock = CountingLock()
        evaluator._client_lock = cast(Any, counting_lock)

        project_client, openai_client = evaluator._get_client()

        assert counting_lock.enter_count == 1
        assert project_client is evaluator._project_client
        assert openai_client is evaluator._openai_client

    def test_evaluate_times_out_with_failed_result(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluator

        mock_client = MagicMock()
        mock_run = MagicMock(id="run-1", status="in_progress")
        mock_client.evals.runs.create.return_value = mock_run
        mock_client.evals.runs.retrieve.return_value = mock_run

        evaluator = FoundryEvaluator.__new__(FoundryEvaluator)
        evaluator._display_name = "test_eval"
        evaluator._poll_interval = 5.0
        evaluator._max_wait = 10.0

        with (
            patch.object(
                evaluator, "_get_client", return_value=(MagicMock(), mock_client)
            ),
            patch.object(evaluator, "_ensure_eval_definition", return_value="eval-1"),
            patch(
                "langchain_azure_ai.evaluation.foundry.time.monotonic",
                side_effect=[0.0, 1.0, 11.0],
            ),
            patch("langchain_azure_ai.evaluation.foundry.time.sleep") as sleep_mock,
        ):
            result = evaluator.evaluate(query="q", response="r")

        assert result.passed is False
        assert result.label == "error"
        assert "timed out" in (result.explanation or "").lower()
        sleep_mock.assert_called_once_with(5.0)

    def test_parse_result_applies_label_override_for_sample_results(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluator

        sample_result = SimpleNamespace(
            passed=True,
            label="fail",
            score=0.1,
            reason="Label should win",
        )
        output_item = SimpleNamespace(
            sample=SimpleNamespace(results={"grader": sample_result})
        )

        client = MagicMock()
        client.evals.runs.output_items.list.return_value = [output_item]

        evaluator = FoundryEvaluator.__new__(FoundryEvaluator)
        evaluator._display_name = "test_eval"

        result = evaluator._parse_result(client, "eval-1", "run-1", "completed")

        assert result.passed is False
        assert result.label == "fail"
        assert result.score == 0.1


class TestFoundryEvaluatorSuite:
    """Test FoundryEvaluatorSuite."""

    def test_from_config(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluatorSuite

        with patch(
            "langchain_azure_ai.evaluation.foundry.FoundryEvaluator"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            suite = FoundryEvaluatorSuite.from_config(
                project_endpoint="https://test",
                deployment_name="gpt-4o",
                evaluator_configs=[
                    {"name": "eval1", "evaluator_name": "builtin.task_completion"},
                    {"name": "eval2", "evaluator_name": "builtin.task_adherence"},
                ],
            )
            assert len(suite._evaluators) == 2

    def test_all_passed_reflects_last_results(self) -> None:
        from langchain_azure_ai.evaluation.foundry import (
            FoundryEvalResult,
            FoundryEvaluatorSuite,
        )

        evaluator_1 = MagicMock()
        evaluator_2 = MagicMock()
        evaluator_1.evaluate.return_value = FoundryEvalResult("eval1", passed=True)
        evaluator_2.evaluate.return_value = FoundryEvalResult("eval2", passed=False)

        suite = FoundryEvaluatorSuite([evaluator_1, evaluator_2])
        assert suite.all_passed is False

        results = suite.evaluate_all(query="q", response="r")

        assert len(results) == 2
        assert suite.all_passed is False

        evaluator_2.evaluate.return_value = FoundryEvalResult("eval2", passed=True)
        suite.evaluate_all(query="q", response="r")
        assert suite.all_passed is True

    def test_close_closes_all_evaluators(self) -> None:
        from langchain_azure_ai.evaluation.foundry import FoundryEvaluatorSuite

        evaluator_1 = MagicMock()
        evaluator_2 = MagicMock()

        suite = FoundryEvaluatorSuite([evaluator_1, evaluator_2])

        suite.close()

        evaluator_1.close.assert_called_once_with()
        evaluator_2.close.assert_called_once_with()


# ============================================================
# Tests for tracer emit_evaluation_event
# ============================================================


class TestTracerEvalEvent:
    """Test AzureAIOpenTelemetryTracer.emit_evaluation_event.

    The method prefers the OTel Logs/Events API (for Azure Monitor
    customEvents table) and falls back to span.add_event() when
    the Logs API is not available.
    """

    def _make_tracer(self) -> Any:
        with patch(
            "langchain_azure_ai.callbacks.tracers.inference_tracing.configure_azure_monitor"
        ):
            from langchain_azure_ai.callbacks.tracers.inference_tracing import (
                AzureAIOpenTelemetryTracer,
            )

            return AzureAIOpenTelemetryTracer(
                connection_string=None,
                enable_content_recording=True,
            )

    def test_emit_with_no_spans_logs_warning(self) -> None:
        tracer = self._make_tracer()
        tracer.emit_evaluation_event(
            evaluation_name="test_eval",
            score_value=3.0,
            score_label="pass",
        )

    def test_emit_on_invoke_agent_span(self) -> None:
        tracer = self._make_tracer()

        mock_span = MagicMock()
        from langchain_azure_ai.callbacks.tracers.inference_tracing import _SpanRecord

        tracer._spans["test-run-1"] = _SpanRecord(
            run_id="test-run-1",
            span=mock_span,
            operation="invoke_agent",
            parent_run_id=None,
        )

        # Patch OTel Logs API to be unavailable → forces span.add_event fallback
        with patch.dict("sys.modules", {"opentelemetry._logs": None}):
            tracer.emit_evaluation_event(
                evaluation_name="task_completion",
                score_value=4.5,
                score_label="pass",
                explanation="Analysis is thorough",
            )

        mock_span.add_event.assert_called_once()
        event_name = mock_span.add_event.call_args[0][0]
        assert "evaluation" in event_name.lower()

    def test_emit_via_logs_api_when_available(self) -> None:
        """When OTel Logs API is available with a real LoggerProvider,
        emit_evaluation_event should use it instead of span.add_event()."""
        tracer = self._make_tracer()

        from opentelemetry.sdk._logs import LoggerProvider as SdkLP
        from opentelemetry.sdk.trace import TracerProvider

        tracer_provider = TracerProvider()
        test_span = tracer_provider.get_tracer(__name__).start_span("emit-evaluation")
        from langchain_azure_ai.callbacks.tracers.inference_tracing import (
            Attrs,
            _SpanRecord,
        )

        tracer._spans["r1"] = _SpanRecord(
            run_id="r1",
            span=test_span,
            operation="invoke_agent",
            parent_run_id=None,
        )

        mock_otel_logger = MagicMock()
        logger_provider = SdkLP()
        try:
            with (
                patch(
                    "opentelemetry._logs.get_logger_provider",
                    return_value=logger_provider,
                ),
                patch.object(
                    logger_provider,
                    "get_logger",
                    return_value=mock_otel_logger,
                ),
                patch.object(test_span, "add_event") as mock_add_event,
            ):
                tracer.emit_evaluation_event(
                    evaluation_name="quality",
                    score_value=4.0,
                    score_label="pass",
                )
                mock_add_event.assert_not_called()
        finally:
            test_span.end()

        mock_otel_logger.emit.assert_called_once()
        call_kwargs = mock_otel_logger.emit.call_args[1]
        assert call_kwargs["body"] == Attrs.EVALUATION_RESULT_EVENT
        assert call_kwargs["event_name"] == Attrs.EVALUATION_RESULT_EVENT
        assert (
            call_kwargs["attributes"][Attrs.AZURE_MONITOR_CUSTOM_EVENT_NAME]
            == Attrs.EVALUATION_RESULT_EVENT
        )
        assert call_kwargs["attributes"][Attrs.EVALUATION_NAME] == "quality"
        assert call_kwargs["attributes"][Attrs.EVALUATION_SCORE_VALUE] == 4.0

    def test_emit_with_explicit_run_id(self) -> None:
        tracer = self._make_tracer()

        mock_span_1 = MagicMock()
        mock_span_2 = MagicMock()

        from langchain_azure_ai.callbacks.tracers.inference_tracing import _SpanRecord

        tracer._spans["run-1"] = _SpanRecord(
            run_id="run-1",
            span=mock_span_1,
            operation="invoke_agent",
            parent_run_id=None,
        )
        tracer._spans["run-2"] = _SpanRecord(
            run_id="run-2", span=mock_span_2, operation="chat", parent_run_id=None
        )

        # Force span.add_event fallback
        with patch.dict("sys.modules", {"opentelemetry._logs": None}):
            tracer.emit_evaluation_event(
                evaluation_name="test",
                score_label="fail",
                run_id="run-2",
            )

        mock_span_2.add_event.assert_called_once()
        mock_span_1.add_event.assert_not_called()

    def test_emit_attributes_content(self) -> None:
        tracer = self._make_tracer()

        mock_span = MagicMock()
        from langchain_azure_ai.callbacks.tracers.inference_tracing import (
            Attrs,
            _SpanRecord,
        )

        tracer._spans["r1"] = _SpanRecord(
            run_id="r1", span=mock_span, operation="invoke_agent", parent_run_id=None
        )

        # Force span.add_event fallback to check attribute content
        with patch.dict("sys.modules", {"opentelemetry._logs": None}):
            tracer.emit_evaluation_event(
                evaluation_name="task_adherence",
                score_value=3.0,
                score_label="fail",
                explanation="Missing risk section",
                response_id="resp-123",
            )

        mock_span.add_event.assert_called_once()
        attrs = mock_span.add_event.call_args[1]["attributes"]
        assert attrs[Attrs.EVALUATION_NAME] == "task_adherence"
        assert attrs[Attrs.EVALUATION_SCORE_VALUE] == 3.0
        assert attrs[Attrs.EVALUATION_SCORE_LABEL] == "fail"
        assert attrs[Attrs.EVALUATION_EXPLANATION] == "Missing risk section"
        assert attrs[Attrs.RESPONSE_ID] == "resp-123"

    def test_logs_api_activates_span_context(self) -> None:
        """The Logs API path should activate the target span's context
        so the log record is parented to the correct trace/span."""
        tracer = self._make_tracer()

        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk._logs import LoggerProvider as SdkLP
        from opentelemetry.sdk.trace import TracerProvider

        tracer_provider = TracerProvider()
        test_span = tracer_provider.get_tracer(__name__).start_span(
            "emit-evaluation-context"
        )
        from langchain_azure_ai.callbacks.tracers.inference_tracing import (
            Attrs,
            _SpanRecord,
        )

        tracer._spans["r1"] = _SpanRecord(
            run_id="r1",
            span=test_span,
            operation="invoke_agent",
            parent_run_id=None,
        )

        mock_otel_logger = MagicMock()
        logger_provider = SdkLP()
        observed_spans: list[Any] = []

        def _capture_current_span(*_: Any, **__: Any) -> None:
            observed_spans.append(otel_trace.get_current_span())

        mock_otel_logger.emit.side_effect = _capture_current_span
        try:
            with (
                patch(
                    "opentelemetry._logs.get_logger_provider",
                    return_value=logger_provider,
                ),
                patch.object(
                    logger_provider,
                    "get_logger",
                    return_value=mock_otel_logger,
                ),
                patch.object(test_span, "add_event") as mock_add_event,
            ):
                tracer.emit_evaluation_event(
                    evaluation_name="test",
                    score_value=5.0,
                )
                mock_add_event.assert_not_called()
        finally:
            test_span.end()

        mock_otel_logger.emit.assert_called_once()
        call_kwargs = mock_otel_logger.emit.call_args[1]
        assert call_kwargs["event_name"] == Attrs.EVALUATION_RESULT_EVENT
        assert (
            call_kwargs["attributes"][Attrs.AZURE_MONITOR_CUSTOM_EVENT_NAME]
            == Attrs.EVALUATION_RESULT_EVENT
        )
        assert observed_spans == [test_span]


# ============================================================
# Tests for helpers.py
# ============================================================


class TestCreateEvalOptimizeSubgraph:
    """Test the eval-optimize subgraph builder."""

    def test_builds_and_compiles(self) -> None:
        from typing_extensions import TypedDict

        from langchain_azure_ai.evaluation.helpers import create_eval_optimize_subgraph

        class TestState(TypedDict):
            value: str
            accepted: bool
            iteration: int

        def evaluate(state: dict[str, Any]) -> dict[str, Any]:
            return {"accepted": True, "iteration": state["iteration"] + 1}

        def refine(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": state["value"] + " refined"}

        def should_refine(state: dict[str, Any]) -> str:
            return "accepted" if state["accepted"] else "refine"

        graph = create_eval_optimize_subgraph(
            evaluate_fn=evaluate,
            refine_fn=refine,
            should_refine_fn=should_refine,
            state_schema=TestState,
        )
        assert graph is not None

        result = graph.invoke(
            {
                "value": "draft",
                "accepted": False,
                "iteration": 0,
            }
        )
        assert result["accepted"] is True

    def test_refine_loop(self) -> None:
        from typing_extensions import TypedDict

        from langchain_azure_ai.evaluation.helpers import create_eval_optimize_subgraph

        class TestState(TypedDict):
            value: str
            accepted: bool
            iteration: int
            max_iterations: int

        call_count = {"evaluate": 0}

        def evaluate(state: dict[str, Any]) -> dict[str, Any]:
            call_count["evaluate"] += 1
            accepted = call_count["evaluate"] >= 2
            return {
                "accepted": accepted,
                "iteration": state["iteration"] + 1,
            }

        def refine(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": state["value"] + "+"}

        def should_refine(state: dict[str, Any]) -> str:
            if state["accepted"]:
                return "accepted"
            if state["iteration"] >= state.get("max_iterations", 3):
                return "accepted"
            return "refine"

        graph = create_eval_optimize_subgraph(
            evaluate_fn=evaluate,
            refine_fn=refine,
            should_refine_fn=should_refine,
            state_schema=TestState,
        )
        result = graph.invoke(
            {
                "value": "draft",
                "accepted": False,
                "iteration": 0,
                "max_iterations": 3,
            }
        )
        assert call_count["evaluate"] == 2
        assert "+" in result["value"]

    def test_max_iterations_forces_accepted_route(self) -> None:
        from typing_extensions import TypedDict

        from langchain_azure_ai.evaluation.helpers import create_eval_optimize_subgraph

        class TestState(TypedDict):
            value: str
            accepted: bool

        call_count = {"evaluate": 0, "refine": 0}

        def evaluate(state: dict[str, Any]) -> dict[str, Any]:
            call_count["evaluate"] += 1
            return {"accepted": False, "value": state["value"]}

        def refine(state: dict[str, Any]) -> dict[str, Any]:
            call_count["refine"] += 1
            return {"value": state["value"] + "+"}

        def should_refine(_: dict[str, Any]) -> str:
            return "refine"

        graph = create_eval_optimize_subgraph(
            evaluate_fn=evaluate,
            refine_fn=refine,
            should_refine_fn=should_refine,
            state_schema=TestState,
            max_iterations=2,
        )

        result = graph.invoke({"value": "draft", "accepted": False})

        assert result["accepted"] is False
        assert result["value"] == "draft+"
        assert call_count == {"evaluate": 2, "refine": 1}

    def test_max_iterations_are_tracked_per_concurrent_invocation(self) -> None:
        from typing_extensions import TypedDict

        from langchain_azure_ai.evaluation.helpers import create_eval_optimize_subgraph

        class TestState(TypedDict):
            value: str
            accepted: bool

        evaluate_barrier = threading.Barrier(2)
        call_count = {"evaluate": 0, "refine": 0}

        def evaluate(state: dict[str, Any]) -> dict[str, Any]:
            call_count["evaluate"] += 1
            if call_count["evaluate"] <= 2:
                evaluate_barrier.wait(timeout=2)
            return {"accepted": False, "value": state["value"]}

        def refine(state: dict[str, Any]) -> dict[str, Any]:
            call_count["refine"] += 1
            return {"value": state["value"] + "+"}

        graph = create_eval_optimize_subgraph(
            evaluate_fn=evaluate,
            refine_fn=refine,
            should_refine_fn=lambda _: "refine",
            state_schema=TestState,
            max_iterations=2,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(graph.invoke, {"value": "draft", "accepted": False})
                for _ in range(2)
            ]
            results = [future.result() for future in futures]

        assert [result["value"] for result in results] == ["draft+", "draft+"]
        assert call_count == {"evaluate": 4, "refine": 2}


class TestCreateAnalystSubgraph:
    """Test the analyst subgraph builder."""

    def test_custom_state_mapping_supports_reusable_state_shape(self) -> None:
        from typing_extensions import TypedDict

        from langchain_azure_ai.evaluation.helpers import create_analyst_subgraph

        class AnalystState(TypedDict):
            topic: str
            draft: str
            final_output: str

        eval_graph = MagicMock()
        eval_graph.invoke.return_value = {
            "optimized_draft": "draft content optimized",
            "score": "pass",
        }

        def research(state: dict[str, Any]) -> dict[str, Any]:
            return {"topic": state["topic"]}

        def write(_: dict[str, Any]) -> dict[str, Any]:
            return {"draft": "draft content"}

        def build_completed(state: dict[str, Any]) -> dict[str, Any]:
            return {"final_output": state["draft"]}

        graph = create_analyst_subgraph(
            research_fn=research,
            write_fn=write,
            eval_optimize_graph=eval_graph,
            build_completed_fn=build_completed,
            state_schema=AnalystState,
            state_to_eval_input=lambda state, iteration_limit: {
                "content": state["draft"],
                "limit": iteration_limit,
            },
            eval_output_to_state=lambda _state, eval_output: {
                "draft": eval_output["optimized_draft"]
            },
        )

        result = graph.invoke({"topic": "Azure", "draft": "", "final_output": ""})

        eval_graph.invoke.assert_called_once_with(
            {"content": "draft content", "limit": 3}
        )
        assert result["final_output"] == "draft content optimized"

    def test_default_state_mapping_supports_mapping_sections(self) -> None:
        from typing_extensions import TypedDict

        from langchain_azure_ai.evaluation.helpers import create_analyst_subgraph

        class AnalystState(TypedDict):
            section: dict[str, str]
            draft_content: str
            evaluation_result: str | None
            iteration_count: int

        eval_graph = MagicMock()
        eval_graph.invoke.return_value = {
            "draft_content": "refined draft",
            "evaluation_result": "pass",
            "iteration": 1,
        }

        graph = create_analyst_subgraph(
            research_fn=lambda state: {"section": state["section"]},
            write_fn=lambda state: {"draft_content": state["draft_content"]},
            eval_optimize_graph=eval_graph,
            build_completed_fn=lambda state: {"draft_content": state["draft_content"]},
            state_schema=AnalystState,
        )

        result = graph.invoke(
            {
                "section": {"area": "finance", "title": "Summary"},
                "draft_content": "draft",
                "evaluation_result": None,
                "iteration_count": 0,
            }
        )

        eval_graph.invoke.assert_called_once_with(
            {
                "section_area": "finance",
                "section_title": "Summary",
                "draft_content": "draft",
                "evaluation_feedback": "",
                "evaluation_result": None,
                "accepted": False,
                "iteration": 0,
                "max_iterations": 3,
            }
        )
        assert result["draft_content"] == "refined draft"
