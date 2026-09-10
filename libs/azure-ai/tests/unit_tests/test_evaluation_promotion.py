from __future__ import annotations

from dataclasses import replace

from langchain_azure_ai.evaluation import (
    EvaluationReceipt,
    EvaluationScores,
    PromotionPolicy,
    evaluate_candidate_promotion,
)


def _receipt(
    candidate_id: str,
    *,
    cost: float = 1.0,
    success: float = 0.95,
    security_failures: int = 0,
) -> EvaluationReceipt:
    return EvaluationReceipt(
        candidate_id=candidate_id,
        retrieval_stack="azure-ai-search+graph",
        prompt_hash="sha256:prompt",
        agent_graph_hash="sha256:graph",
        corpus_hash="sha256:corpus",
        policy_hash="sha256:policy",
        scores=EvaluationScores(
            task_success=success,
            groundedness=0.98,
            citation_correctness=0.97,
            retrieval_recall=0.92,
            prompt_injection_resistance=0.99,
            critical_security_failures=security_failures,
        ),
        cost_per_successful_task=cost,
        latency_p95_ms=2_000,
        sample_size=500,
        evidence_uri="https://example.invalid/evidence/1",
    )


def test_passing_candidate_only_proposes_canary() -> None:
    decision = evaluate_candidate_promotion(
        _receipt("azure/incumbent"),
        _receipt("azure/candidate", cost=0.8),
        PromotionPolicy(),
    )

    assert decision.decision == "CANARY"
    assert decision.requires_independent_approval is True
    assert decision.failures == ()


def test_security_failure_denies_cheaper_candidate() -> None:
    decision = evaluate_candidate_promotion(
        _receipt("azure/incumbent"),
        _receipt("azure/candidate", cost=0.1, security_failures=1),
        PromotionPolicy(),
    )

    assert decision.decision == "DENY"
    assert "critical_security_failure" in decision.failures


def test_quality_regression_denies_cheaper_candidate() -> None:
    decision = evaluate_candidate_promotion(
        _receipt("azure/incumbent", success=0.96),
        _receipt("azure/candidate", cost=0.1, success=0.94),
        PromotionPolicy(),
    )

    assert "task_success_regression" in decision.failures


def test_insufficient_sample_fails_closed() -> None:
    candidate = replace(_receipt("azure/candidate", cost=0.8), sample_size=10)
    decision = evaluate_candidate_promotion(
        _receipt("azure/incumbent"), candidate, PromotionPolicy()
    )

    assert "insufficient_evaluation_sample" in decision.failures


def test_receipt_hash_covers_retrieval_stack() -> None:
    receipt = _receipt("azure/candidate")
    other_stack = replace(receipt, retrieval_stack="pinecone+neo4j")

    assert receipt.to_dict()["receipt_hash"] != other_stack.to_dict()["receipt_hash"]


def test_decision_hash_covers_failures() -> None:
    incumbent = _receipt("azure/incumbent")
    passing = evaluate_candidate_promotion(
        incumbent, _receipt("azure/candidate", cost=0.8), PromotionPolicy()
    )
    failing = evaluate_candidate_promotion(
        incumbent,
        _receipt("azure/candidate", cost=0.8, security_failures=1),
        PromotionPolicy(),
    )

    assert passing.to_dict()["decision_hash"] != failing.to_dict()["decision_hash"]
