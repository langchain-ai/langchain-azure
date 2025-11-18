"""Azure AI OpenTelemetry tracing for LangChain and LangGraph.

This module provides automatic tracing of LangChain/LangGraph executions with
OpenTelemetry spans aligned with GenAI semantic conventions.

Usage:
    from langchain_azure_ai.tracing import AzureAIOpenTelemetryTracer

    # Configure once at startup
    AzureAIOpenTelemetryTracer.set_app_insights(
        "InstrumentationKey=...;IngestionEndpoint=..."
    )
    AzureAIOpenTelemetryTracer.set_config({
        "provider_name": "azure.ai.openai",
        "redact_messages": False
    })
    
    # Enable automatic tracing
    AzureAIOpenTelemetryTracer.autolog()
    
    # All LangChain operations are now automatically traced
    result = chain.invoke(input)
"""

from langchain_azure_ai.tracing.tracer import AzureAIOpenTelemetryTracer

__all__ = ["AzureAIOpenTelemetryTracer"]
