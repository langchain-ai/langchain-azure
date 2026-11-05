"""Unit tests for Cosmos DB request charge reporting."""

import logging

import pytest

from langchain_azure_cosmosdb._request_charge import (
    CosmosDBRequestCharge,
    RequestChargeAccumulator,
)


def test_accumulator_ignores_missing_malformed_and_non_finite_charges() -> None:
    events: list[CosmosDBRequestCharge] = []
    accumulator = RequestChargeAccumulator()

    accumulator.response_hook({}, {})
    accumulator.response_hook({"x-ms-request-charge": "invalid"}, {})
    accumulator.response_hook({"x-ms-request-charge": "nan"}, {})
    accumulator.response_hook({"x-ms-request-charge": "-1"}, {})
    accumulator.emit(events.append, operation="query")

    assert events == []


def test_accumulator_counts_a_valid_zero_charge() -> None:
    events: list[CosmosDBRequestCharge] = []
    accumulator = RequestChargeAccumulator()

    accumulator.response_hook({"x-ms-request-charge": "0.0"}, {})
    accumulator.emit(events.append, operation="query")

    assert events == [
        CosmosDBRequestCharge(
            operation="query",
            request_charge=0.0,
            request_count=1,
        )
    ]


def test_callback_failure_does_not_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    accumulator = RequestChargeAccumulator()
    accumulator.response_hook({"x-ms-request-charge": "1.5"}, {})

    def fail(_event: CosmosDBRequestCharge) -> None:
        raise RuntimeError("callback failed")

    with caplog.at_level(logging.WARNING):
        accumulator.emit(fail, operation="query")

    assert "Cosmos DB request charge callback failed" in caplog.text
