# What this sample demonstrates

A minimal [LangGraph](https://langchain-ai.github.io/langgraph/) agent
hosted using the **Responses protocol**, configured to emit
OpenTelemetry GenAI semantic-convention spans for every LangGraph node,
model call, and tool call.

The agent itself is intentionally trivial (one `create_agent` with no
custom tools) so the focus is on **how to wire observability** rather
than agent behavior. The full tracing wiring is also documented in the
parent README's [Tracing](../../README.md#tracing) section; this sample
makes it discoverable as a standalone, deployable scenario.

## How It Works

### Model Integration

The agent uses `langchain_openai.AzureChatOpenAI` against the Foundry
project endpoint, authenticated with
`DefaultAzureCredential`.

### Tracing wiring

The destination is picked **purely from environment variables**. First
matching rule wins:

| # | Env var(s) set | Where spans go |
|---|----------------|----------------|
| 0 | `OTEL_SDK_DISABLED=true` | nowhere — OTel SDK no-ops |
| 1 | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) | OTLP/HTTP collector at that URL |
| 2 | `APPLICATION_INSIGHTS_CONNECTION_STRING` | Azure Monitor directly |
| 3 | `FOUNDRY_PROJECT_ENDPOINT` (always set for the samples) | Foundry project's managed App Insights (connection string auto-resolved over the Foundry control plane; needs `az login` + Reader on the project) |
| 4 | none of the above | tracer attached but no exporter |

Set `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true` to also record
message content, tool arguments, and tool results on the spans
(default: redacted). This sample's `.env.example` enables it so you can
see full content end-to-end in App Insights.

The exact selection code in [main.py](main.py):

```python
if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT"
):
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    enable_auto_tracing()
else:
    enable_auto_tracing(auto_configure_azure_monitor=True)
```

Inside the **Foundry hosted-agent sandbox**, `azure-ai-agentserver-core`
has already configured the `TracerProvider` + Azure Monitor exporter
for its per-request server span; the langchain GenAI spans then nest
under that existing span. `enable_auto_tracing(auto_configure_azure_monitor=True)`
detects the existing provider and skips reconfiguring it, so the same
branch still works.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.LangGraphResponsesHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
README in the parent directory.

## Interacting with the agent

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Tell me a fun fact about distributed tracing."}'
```

## Inspecting the spans

After invoking the agent at least once, open the destination resource:

- **Foundry-managed App Insights**: in the Foundry portal, open the
  project, then the *Tracing* tab.
- **Standalone App Insights**: open the resource in the Azure portal
  and navigate to *Investigate → Transaction search*.
- **OTLP collector**: whichever backend you wired to the collector
  (Jaeger, Tempo, etc.).

You should see at least the following spans per request:

- One `invoke_agent` server span (the per-request root).
- Nested `chain` / `graph` spans for each LangGraph node.
- One `chat <model>` span per model call, with prompt / completion
  attributes when `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true`.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory. The Foundry hosting infrastructure
injects `APPLICATIONINSIGHTS_CONNECTION_STRING` at runtime, so spans
flow into the project's managed App Insights without any extra
configuration.
