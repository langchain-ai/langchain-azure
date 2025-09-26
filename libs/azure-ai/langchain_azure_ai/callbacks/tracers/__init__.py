"""Tracing capabilities for Azure AI Foundry."""

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    AzureAIOpenTelemetryTracer,
)

__all__ = [
    "AzureAIOpenTelemetryTracer",
]
