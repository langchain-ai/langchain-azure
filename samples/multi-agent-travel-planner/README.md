# Nested Agent Travel Planner Sample

This sample demonstrates a multi-agent travel planning system using LangGraph with **automatic tracing** via the new `AzureAIOpenTelemetryTracer.autolog()` API.

## Features

- **Automatic Tracing**: Uses the new autolog API - no need to pass callbacks explicitly!
- **Multi-Agent System**: Coordinates multiple specialized agents (flight, hotel, activities, synthesis)
- **Nested Agents**: Demonstrates an inner agent (itinerary editor) running within another agent
- **OpenTelemetry Compliance**: Emits spans aligned with GenAI semantic conventions
- **Dual Export**: Sends telemetry to both Azure Application Insights and OTLP endpoints

## Setup

1. Install dependencies:
```bash
pip install langchain-azure-ai langchain-openai langgraph python-dotenv
```

2. Set environment variables:
```bash
# Required: Azure OpenAI configuration
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export OPENAI_MODEL="gpt-4"

# Required: Application Insights for tracing
export APPLICATION_INSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=..."

# Optional: Additional tracing configuration
export NESTED_SAMPLE_PROVIDER="azure.ai.openai"
export OTEL_SERVICE_NAME="nested-travel-sample"
```

## Running the Sample

```bash
python nested_agent_travel_planner.py
```

## New Autolog Tracing Pattern

This sample has been updated to use the new autolog API:

### Old Pattern (Manual Callback Passing)
```python
# Create tracer instance
tracer = AzureAIOpenTelemetryTracer(
    name="nested_travel_planner",
    provider_name="openai"
)

# Pass tracer explicitly to every invocation
result = agent.invoke(input, config={"callbacks": [tracer]})
```

### New Pattern (Automatic Tracing)
```python
# Configure once at startup
AzureAIOpenTelemetryTracer.set_app_insights(
    "InstrumentationKey=...;IngestionEndpoint=..."
)
AzureAIOpenTelemetryTracer.set_config({
    "provider_name": "azure.ai.openai",
    "patch_mode": "monkey"  # Enable automatic callback injection
})
AzureAIOpenTelemetryTracer.autolog()

# All LangChain operations are now automatically traced!
result = agent.invoke(input)  # No callbacks needed! 🎉
```

## What Gets Traced

The sample demonstrates tracing of:

1. **Top-level workflow**: The main LangGraph workflow
2. **Coordinator agent**: Extracts key details from user request
3. **Flight specialist**: Searches and recommends flights
4. **Hotel specialist**: Finds boutique hotels
5. **Activity specialist**: Curates activities
6. **Plan synthesizer**: Combines all information
7. **Nested itinerary editor**: Inner agent that polishes the final plan

All traces include:
- GenAI semantic convention attributes
- Token usage tracking and aggregation
- Agent metadata (name, id, description)
- Conversation/session tracking
- Request parameters (model, temperature, etc.)

## Architecture

The sample uses a state machine pattern with the following flow:

```
START → Coordinator → Flight Specialist → Hotel Specialist → Activity Specialist → Plan Synthesizer → END
                                                                                         ↓
                                                                              (Nested Itinerary Editor)
```

Each specialist agent:
- Uses specific tools to gather information
- Runs with different temperature settings for variety
- Contributes its findings to the shared state
- Is automatically traced without explicit callback passing

## Output

The sample produces a complete travel itinerary including:
- Flight recommendations
- Hotel suggestions
- Activity highlights
- Polished final itinerary

All execution is traced to Azure Application Insights with:
- Complete span hierarchy showing agent relationships
- Token usage aggregation
- GenAI semantic convention compliance

## Cleanup

The sample automatically shuts down tracing at the end:

```python
if AzureAIOpenTelemetryTracer.is_active():
    AzureAIOpenTelemetryTracer.shutdown()
```

This ensures all telemetry is flushed before the application exits.
