"""Request charge reporting for Azure Cosmos DB operations."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CosmosDBRequestCharge:
    """Aggregated request charge for one logical Cosmos DB operation."""

    operation: str
    request_charge: float
    request_count: int


CosmosDBRequestChargeCallback = Callable[[CosmosDBRequestCharge], None]


class RequestChargeAccumulator:
    """Accumulate request charges supplied by Cosmos SDK response hooks."""

    def __init__(self) -> None:
        self.request_charge = 0.0
        self.request_count = 0

    def response_hook(self, headers: Mapping[str, str], _result: Any) -> None:
        """Capture a request charge from one SDK response."""
        raw_charge = next(
            (
                value
                for key, value in headers.items()
                if key.lower() == "x-ms-request-charge"
            ),
            None,
        )
        if raw_charge is None:
            return

        try:
            charge = float(raw_charge)
        except (TypeError, ValueError):
            return
        if not math.isfinite(charge) or charge < 0:
            return

        self.request_charge += charge
        self.request_count += 1

    def emit(
        self,
        callback: CosmosDBRequestChargeCallback | None,
        *,
        operation: str,
    ) -> None:
        """Emit the aggregate without disrupting a successful operation."""
        if callback is None or self.request_count == 0:
            return

        event = CosmosDBRequestCharge(
            operation=operation,
            request_charge=self.request_charge,
            request_count=self.request_count,
        )
        try:
            callback(event)
        except Exception:
            logger.warning("Cosmos DB request charge callback failed", exc_info=True)
