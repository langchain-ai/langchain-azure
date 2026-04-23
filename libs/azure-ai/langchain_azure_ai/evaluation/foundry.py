"""Azure Foundry agent evaluator wrappers for LangGraph.

Provides wrappers around Foundry's built-in agent evaluators
(TaskCompletion, TaskAdherence, ToolCallAccuracy, etc.) that can
be used as LangGraph node functions or standalone evaluators.

These wrappers handle:
- Creating evaluation definitions via ``client.evals.create()``
- Running evaluations with inline data via ``client.evals.runs.create()``
- Polling for completion and extracting results
- Emitting OTel evaluation events via the tracer

Usage requires ``azure-ai-projects >= 2.0.0`` and a Foundry project endpoint.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_azure_ai._api.base import experimental

LOGGER = logging.getLogger(__name__)

_PACKAGE_NAME = "langchain-azure-ai"
try:
    _PACKAGE_VERSION = importlib.metadata.version(_PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    _PACKAGE_VERSION = "0.0.0"
_USER_AGENT = f"{_PACKAGE_NAME}/{_PACKAGE_VERSION}"


def _apply_label_override(passed: bool, label: str | None) -> bool:
    """Normalize pass/fail labels into the returned pass state."""
    if label is None:
        return passed

    normalized = label.strip().lower()
    if normalized in {"pass", "passed"}:
        return True
    if normalized in {"fail", "failed"}:
        return False
    return passed


@dataclass
class FoundryEvalResult:
    """Result from a Foundry evaluation run."""

    evaluator_name: str
    passed: bool
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    raw_output: dict[str, Any] = field(default_factory=dict)


@experimental(message="Foundry evaluation integration is in preview and may change.")
class FoundryEvaluator:
    """Wrapper around a single Foundry agent evaluator.

    Manages the lifecycle of creating an eval definition and running
    evaluations against it. Each instance corresponds to one evaluator
    type (e.g., TaskCompletion).

    Args:
        project_endpoint: Azure AI Foundry project endpoint URL.
        evaluator_name: Builtin evaluator name
            (e.g., ``"builtin.task_completion"``).
        deployment_name: Model deployment for the LLM judge.
        display_name: Human-readable name for the evaluator.
        credential: Azure credential. Defaults to ``DefaultAzureCredential``.
        poll_interval: Seconds between status checks.
        max_wait: Maximum seconds to wait for eval completion.
    """

    def __init__(
        self,
        *,
        project_endpoint: str,
        evaluator_name: str,
        deployment_name: str,
        display_name: str | None = None,
        credential: Any | None = None,
        poll_interval: float = 5.0,
        max_wait: float = 300.0,
    ) -> None:
        """Initialize a Foundry evaluator wrapper."""
        self._project_endpoint = project_endpoint
        self._evaluator_name = evaluator_name
        self._deployment_name = deployment_name
        self._display_name = display_name or evaluator_name.split(".")[-1]
        self._poll_interval = poll_interval
        self._max_wait = max_wait

        if credential is None:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
        self._credential = credential

        self._eval_id: str | None = None
        self._project_client: Any | None = None
        self._openai_client: Any | None = None
        self._client_lock = threading.Lock()
        self._eval_definition_lock = threading.Lock()

    def _get_client(self) -> tuple[Any, Any]:
        """Create or reuse the AIProjectClient and OpenAI client.

        Configures the project client with a custom user-agent:
        - ``langchain-azure-ai/<version>``
        """
        with self._client_lock:
            if self._project_client is not None and self._openai_client is not None:
                return self._project_client, self._openai_client

            from azure.ai.projects import AIProjectClient
            from azure.core.pipeline.policies import UserAgentPolicy

            user_agent_policy = UserAgentPolicy(
                user_agent=_USER_AGENT,
            )
            project_client = AIProjectClient(
                endpoint=self._project_endpoint,
                credential=self._credential,
                per_call_policies=[user_agent_policy],
            )
            self._project_client = project_client
            # The OpenAI client inherits the project client's transport
            # and will carry the user-agent through Azure pipeline policies.
            self._openai_client = project_client.get_openai_client()

            return self._project_client, self._openai_client

    def close(self) -> None:
        """Close cached project resources."""
        with self._client_lock:
            if self._project_client is None:
                return
            try:
                self._project_client.close()
            except Exception:
                LOGGER.debug("Failed to close Foundry project client", exc_info=True)
            finally:
                self._project_client = None
                self._openai_client = None

    def _ensure_eval_definition(self, client: Any) -> str:
        """Create the eval definition if not already created."""
        if self._eval_id is not None:
            return self._eval_id

        from openai.types.eval_create_params import DataSourceConfigCustom

        with self._eval_definition_lock:
            if self._eval_id is not None:
                return self._eval_id

            data_source_config = DataSourceConfigCustom(
                {
                    "type": "custom",
                    "item_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "object"}},
                                ]
                            },
                            "response": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "object"}},
                                ]
                            },
                            "tool_definitions": {
                                "anyOf": [
                                    {"type": "object"},
                                    {"type": "array", "items": {"type": "object"}},
                                ]
                            },
                        },
                        "required": ["query", "response"],
                    },
                    "include_sample_schema": True,
                }
            )

            testing_criteria = [
                {
                    "type": "azure_ai_evaluator",
                    "name": self._display_name,
                    "evaluator_name": self._evaluator_name,
                    "initialization_parameters": {
                        "deployment_name": self._deployment_name,
                    },
                    "data_mapping": {
                        "query": "{{item.query}}",
                        "response": "{{item.response}}",
                    },
                }
            ]

            eval_obj = client.evals.create(
                name=f"LangGraph {self._display_name} Eval",
                data_source_config=data_source_config,
                testing_criteria=testing_criteria,
            )
            self._eval_id = eval_obj.id
        return self._eval_id

    def evaluate(
        self,
        *,
        query: Any,
        response: Any,
        tool_definitions: Any | None = None,
        tracer: Any | None = None,
        run_id: Any | None = None,
    ) -> FoundryEvalResult:
        """Run a single evaluation against Foundry.

        Propagates the current OpenTelemetry trace context (W3C
        ``traceparent``) to the Foundry API so evaluation spans are
        correlated with the calling agent's trace.

        Args:
            query: The query in Foundry format (string or message array).
            response: The response in Foundry format.
            tool_definitions: Optional tool schemas.
            tracer: Optional ``AzureAIOpenTelemetryTracer`` to emit
                evaluation events.
            run_id: Optional run ID for the tracer event.

        Returns:
            Evaluation result with pass/fail and scores.
        """
        from openai.types.evals.create_eval_jsonl_run_data_source_param import (
            CreateEvalJSONLRunDataSourceParam,
            SourceFileContent,
            SourceFileContentContent,
        )

        # Inject W3C traceparent from the current OTel context
        trace_headers: dict[str, str] = {}
        try:
            from opentelemetry.propagate import inject

            inject(trace_headers)
        except Exception:
            pass  # OTel not available or no active context

        _, client = self._get_client()

        # Apply traceparent as extra headers on the OpenAI client
        if trace_headers:
            try:
                client = client.with_options(
                    default_headers=trace_headers,
                )
            except Exception:
                LOGGER.debug("Could not set traceparent headers on OpenAI client")

        eval_id = self._ensure_eval_definition(client)

        eval_run = client.evals.runs.create(
            eval_id=eval_id,
            name=f"{self._display_name}_run",
            data_source=CreateEvalJSONLRunDataSourceParam(
                type="jsonl",
                source=SourceFileContent(
                    type="file_content",
                    content=[
                        SourceFileContentContent(
                            item={
                                "query": query,
                                "response": response,
                                "tool_definitions": tool_definitions,
                            }
                        ),
                    ],
                ),
            ),
        )

        deadline = time.monotonic() + self._max_wait
        run = eval_run
        while True:
            run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_id)
            if run.status in ("completed", "failed"):
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            time.sleep(min(self._poll_interval, remaining))

        final_status = (
            run.status if run.status in ("completed", "failed") else "timeout"
        )
        result = self._parse_result(client, eval_id, eval_run.id, final_status)

        if tracer is not None:
            try:
                tracer.emit_evaluation_event(
                    evaluation_name=self._display_name,
                    score_value=result.score,
                    score_label=result.label,
                    explanation=result.explanation,
                    run_id=run_id,
                )
            except Exception:
                LOGGER.debug("Failed to emit evaluation event", exc_info=True)

        return result

    def _parse_result(
        self,
        client: Any,
        eval_id: str,
        run_id: str,
        status: str,
    ) -> FoundryEvalResult:
        """Parse the evaluation run output into a result."""
        if status != "completed":
            terminal_explanation = (
                f"Evaluation run did not complete successfully. Final status: {status}"
            )
            if status == "timeout":
                terminal_explanation = (
                    "Evaluation run timed out before reaching a terminal status. "
                    f"Final status: {status}"
                )
            return FoundryEvalResult(
                evaluator_name=self._display_name,
                passed=False,
                label="error",
                explanation=terminal_explanation,
            )

        try:
            output_items = list(
                client.evals.runs.output_items.list(run_id=run_id, eval_id=eval_id)
            )
        except Exception as e:
            return FoundryEvalResult(
                evaluator_name=self._display_name,
                passed=False,
                label="error",
                explanation=f"Failed to retrieve results: {e}",
            )

        if not output_items:
            return FoundryEvalResult(
                evaluator_name=self._display_name,
                passed=False,
                label="error",
                explanation="No output items returned",
            )

        item = output_items[0]
        raw: dict[str, Any] = {}
        passed = False
        score: float | None = None
        label: str | None = None
        explanation: str | None = None

        try:
            if hasattr(item, "results") and item.results:
                for result_item in item.results:
                    raw = (
                        dict(result_item)
                        if hasattr(result_item, "__iter__")
                        else {"raw": str(result_item)}
                    )
                    if hasattr(result_item, "passed"):
                        passed = bool(result_item.passed)
                    if hasattr(result_item, "label"):
                        label = str(result_item.label)
                    if hasattr(result_item, "score"):
                        score = (
                            float(result_item.score)
                            if result_item.score is not None
                            else None
                        )
                    if hasattr(result_item, "reason"):
                        explanation = str(result_item.reason)
                    break
            elif hasattr(item, "sample") and item.sample:
                sample = item.sample
                if hasattr(sample, "results") and sample.results:
                    for _, value in sample.results.items():
                        if hasattr(value, "passed"):
                            passed = bool(value.passed)
                        if hasattr(value, "label"):
                            label = str(value.label)
                        if hasattr(value, "score"):
                            score = (
                                float(value.score) if value.score is not None else None
                            )
                        if hasattr(value, "reason"):
                            explanation = str(value.reason)
                        break
        except Exception as e:
            LOGGER.debug("Error parsing eval result: %s", e)
            explanation = f"Parse error: {e}"

        passed = _apply_label_override(passed, label)
        if label is None:
            label = "pass" if passed else "fail"

        return FoundryEvalResult(
            evaluator_name=self._display_name,
            passed=passed,
            score=score,
            label=label,
            explanation=explanation,
            raw_output=raw,
        )


@experimental(message="Foundry evaluation integration is in preview and may change.")
class FoundryEvaluatorSuite:
    """Run multiple Foundry evaluators in sequence.

    Convenience class that holds multiple ``FoundryEvaluator`` instances
    and runs them all against the same input.
    """

    def __init__(self, evaluators: Sequence[FoundryEvaluator]) -> None:
        """Initialize the evaluator suite."""
        self._evaluators = list(evaluators)
        self._last_results: list[FoundryEvalResult] = []

    def close(self) -> None:
        """Close cached resources for all evaluators in the suite."""
        for evaluator in self._evaluators:
            evaluator.close()

    def evaluate_all(
        self,
        *,
        query: Any,
        response: Any,
        tool_definitions: Any | None = None,
        tracer: Any | None = None,
        run_id: Any | None = None,
    ) -> list[FoundryEvalResult]:
        """Run all evaluators and return results."""
        results = []
        for evaluator in self._evaluators:
            result = evaluator.evaluate(
                query=query,
                response=response,
                tool_definitions=tool_definitions,
                tracer=tracer,
                run_id=run_id,
            )
            results.append(result)
        self._last_results = results
        return results

    @property
    def all_passed(self) -> bool:
        """Check if all evaluators passed on the last run."""
        return bool(self._last_results) and all(
            result.passed for result in self._last_results
        )

    @classmethod
    def from_config(
        cls,
        *,
        project_endpoint: str,
        deployment_name: str,
        evaluator_configs: Sequence[dict[str, Any]],
        credential: Any | None = None,
    ) -> "FoundryEvaluatorSuite":
        """Create a suite from a list of evaluator config dicts.

        Each config dict should have:
        - ``name``: Display name
        - ``evaluator_name``: Builtin evaluator name
        """
        evaluators = [
            FoundryEvaluator(
                project_endpoint=project_endpoint,
                evaluator_name=cfg["evaluator_name"],
                deployment_name=deployment_name,
                display_name=cfg.get("name", cfg["evaluator_name"]),
                credential=credential,
            )
            for cfg in evaluator_configs
        ]
        return cls(evaluators)
