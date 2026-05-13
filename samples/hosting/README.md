# Samples — LangGraph hosting (`langchain_azure_ai.agents.hosting`)

| # | File | What it shows |
|---|------|---------------|
| 1 | [sample_01_responses_basic.py](sample_01_responses_basic.py) | Simplest case: host a `create_react_agent` graph as the Responses API. |
| 2 | [sample_02_responses_tools.py](sample_02_responses_tools.py) | Same graph + a `@tool` function. Intermediate tool calls and tool results are surfaced as `function_call` / `function_call_output` output items in both non-streaming and streaming modes. |
| 3 | [sample_03_invocations_basic.py](sample_03_invocations_basic.py) | Host the same graph as the Invocations API, with a `MemorySaver` checkpointer for multi-turn continuity via `agent_session_id`. |
| 4 | [sample_04_invocations_tools.py](sample_04_invocations_tools.py) | Variant of #3 with a local `@tool` function — the agent runs a tool round-trip server-side and returns the final assistant text. Streaming returns per-token text deltas. |
| 5 | [sample_05_workflow_all_in_one.py](sample_05_workflow_all_in_one.py) | All-in-one: a custom multi-node `StateGraph` (plan → tools → synthesize) with two tools, hosted as **both** the Responses API and the Invocations API on the same port via the `app=` parameter. |
| 6 | [sample_06_responses_hitl.py](sample_06_responses_hitl.py) | Human-in-the-loop: the graph uses `langgraph.types.interrupt` to pause for user input. The pause is surfaced as a `function_call(name="__hosted_agent_adapter_interrupt__")` item; the client resumes by posting a matching `function_call_output` with a JSON `{"resume": ...}` payload on the same conversation. |
| 7 | [sample_07_responses_toolbox.py](sample_07_responses_toolbox.py) | Combines this hosting package with `langchain_azure_ai.tools.AzureAIProjectToolbox` — tools are loaded at startup from an Azure AI Foundry Toolbox (a managed multi-MCP gateway) and bound to a `create_react_agent` graph hosted as the Responses API. |
| 8 | [sample_08_responses_foundry_checkpointer.py](sample_08_responses_foundry_checkpointer.py) | Swaps `InMemorySaver` for `langchain_azure_ai.checkpointers.FoundryCheckpointSaver`, persisting LangGraph checkpoints to Azure AI Foundry's managed checkpoint storage (preview). State for a given `conversation.id` survives **server restarts**. |

## Setup

From the sample root (`samples/hosting/`):

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in at minimum
`AZURE_AI_PROJECT_ENDPOINT`. `AZURE_AI_MODEL_DEPLOYMENT_NAME` (defaults
to `gpt-4o`).

Authentication uses default azure auth — `az login` is the simplest setup.

## Running a sample

From the sample root, after the setup above:

```bash
python sample_01_responses_basic.py
```

Open Agent Inspector with the corresponding API and send a message to trigger the agent or use curl command embedded in each sample header.

## Tracing

Each sample emits OpenTelemetry GenAI semantic-convention spans for every
LangGraph node, model call, and tool call via
[`AzureAIOpenTelemetryTracer`](../../libs/azure-ai/langchain_azure_ai/callbacks/tracers/inference_tracing.py).
The destination is picked **purely from environment variables** — no code
edits needed. First matching rule wins:

| # | Env var(s) set | Where spans go |
|---|----------------|----------------|
| 0 | `OTEL_SDK_DISABLED=true` | nowhere — OTel SDK no-ops |
| 1 | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) | OTLP/HTTP collector at that URL |
| 2 | `APPLICATION_INSIGHTS_CONNECTION_STRING` | Azure Monitor directly |
| 3 | `AZURE_AI_PROJECT_ENDPOINT` (always set for the samples) | Foundry project's managed App Insights (connection string auto-resolved over the Foundry control plane; needs `az login` + Reader on the project) |
| 4 | none of the above | tracer attached but no exporter |

Set `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true` to also record
message content, tool arguments, and tool results on the spans (default:
redacted).

The selection logic in every sample's `main()`:

```python
if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT"
):
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    enable_auto_tracing()
else:
    # auto_configure_azure_monitor resolves App Insights from
    # APPLICATION_INSIGHTS_CONNECTION_STRING first, then falls back to
    # AZURE_AI_PROJECT_ENDPOINT (project-managed App Insights).
    enable_auto_tracing(auto_configure_azure_monitor=True)
```

### Quick recipes

**Foundry project's managed App Insights** (default for the samples,
because `AZURE_AI_PROJECT_ENDPOINT` is required for the chat model
anyway):

```bash
az login
# AZURE_AI_PROJECT_ENDPOINT already set in .env — nothing else to do.
python sample_01_responses_basic.py
```

**Direct App Insights connection string** (bypasses the Foundry resolver
— faster startup, no Reader RBAC needed):

```bash
export APPLICATION_INSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=..."
python sample_01_responses_basic.py
```

**Local OTel Collector**:

```bash
docker run --rm -p 4318:4318 otel/opentelemetry-collector:latest
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
python sample_01_responses_basic.py
```

**Off**:

```bash
export OTEL_SDK_DISABLED=true
python sample_01_responses_basic.py
```

Inside the **Foundry hosted-agent sandbox**, `azure-ai-agentserver-core`
has already configured the `TracerProvider` + Azure Monitor exporter for
its per-request server span; the langchain GenAI spans then nest under
that existing span. `enable_auto_tracing(auto_configure_azure_monitor=True)`
detects the existing provider and skips reconfiguring it, so the same
branch still works.
