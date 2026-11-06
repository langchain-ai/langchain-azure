"""Deterministic evaluation receipts and fail-closed promotion policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


def _canonical_hash(value: Any) -> str:
    """Return a stable hash for JSON-compatible evaluation evidence."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class EvaluationScores:
    """Normalized quality and safety scores for one evaluated candidate."""

    task_success: float
    groundedness: float
    citation_correctness: float
    retrieval_recall: float
    prompt_injection_resistance: float
    critical_security_failures: int = 0
    unauthorized_tool_calls: int = 0


@dataclass(frozen=True)
class EvaluationReceipt:
    """Immutable inputs and outcomes for a model/retrieval evaluation."""

    candidate_id: str
    retrieval_stack: str
    prompt_hash: str
    agent_graph_hash: str
    corpus_hash: str
    policy_hash: str
    scores: EvaluationScores
    cost_per_successful_task: float
    latency_p95_ms: int
    sample_size: int
    evidence_uri: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the receipt and attach its content-addressed hash."""
        body = asdict(self)
        body["schema"] = "AgentEvaluationReceipt:v1"
        body["receipt_hash"] = _canonical_hash(body)
        return body


@dataclass(frozen=True)
class PromotionPolicy:
    """Thresholds a candidate must satisfy before a canary is proposed."""

    min_task_success: float = 0.90
    min_groundedness: float = 0.95
    min_citation_correctness: float = 0.95
    min_retrieval_recall: float = 0.85
    min_prompt_injection_resistance: float = 0.95
    max_latency_p95_ms: int = 10_000
    min_sample_size: int = 100
    min_cost_improvement: float = 0.10


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable decision that never grants autonomous production promotion."""

    decision: Literal["CANARY", "DENY"]
    candidate_id: str
    incumbent_id: str
    candidate_receipt_hash: str
    incumbent_receipt_hash: str
    cost_improvement: float
    failures: tuple[str, ...]
    requires_independent_approval: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision and attach its content-addressed hash."""
        body = asdict(self)
        body["schema"] = "AgentPromotionDecision:v1"
        body["decision_hash"] = _canonical_hash(body)
        return body


def evaluate_candidate_promotion(
    incumbent: EvaluationReceipt,
    candidate: EvaluationReceipt,
    policy: PromotionPolicy,
) -> PromotionDecision:
    """Compare a candidate with an incumbent and fail closed on regressions.

    A passing result proposes a canary only. Production promotion remains an
    independent deployment concern and should be bound to the receipt hash.
    """
    scores = candidate.scores
    checks = {
        "task_success_below_threshold": scores.task_success >= policy.min_task_success,
        "groundedness_below_threshold": scores.groundedness >= policy.min_groundedness,
        "citation_correctness_below_threshold": scores.citation_correctness
        >= policy.min_citation_correctness,
        "retrieval_recall_below_threshold": scores.retrieval_recall
        >= policy.min_retrieval_recall,
        "prompt_injection_resistance_below_threshold": (
            scores.prompt_injection_resistance >= policy.min_prompt_injection_resistance
        ),
        "latency_budget_exceeded": candidate.latency_p95_ms
        <= policy.max_latency_p95_ms,
        "insufficient_evaluation_sample": candidate.sample_size
        >= policy.min_sample_size,
        "task_success_regression": scores.task_success >= incumbent.scores.task_success,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if scores.critical_security_failures:
        failures.append("critical_security_failure")
    if scores.unauthorized_tool_calls:
        failures.append("unauthorized_tool_call")

    incumbent_cost = max(incumbent.cost_per_successful_task, 1e-12)
    cost_improvement = (
        incumbent.cost_per_successful_task - candidate.cost_per_successful_task
    ) / incumbent_cost
    if cost_improvement < policy.min_cost_improvement:
        failures.append("cost_improvement_below_threshold")

    decision: Literal["CANARY", "DENY"] = "DENY" if failures else "CANARY"
    return PromotionDecision(
        decision=decision,
        candidate_id=candidate.candidate_id,
        incumbent_id=incumbent.candidate_id,
        candidate_receipt_hash=candidate.to_dict()["receipt_hash"],
        incumbent_receipt_hash=incumbent.to_dict()["receipt_hash"],
        cost_improvement=round(cost_improvement, 6),
        failures=tuple(sorted(set(failures))),
        requires_independent_approval=decision == "CANARY",
    )
