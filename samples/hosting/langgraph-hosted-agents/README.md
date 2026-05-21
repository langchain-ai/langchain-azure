# LangGraph Hosted Agent Samples

This directory contains samples that demonstrate how to host
[LangGraph](https://langchain-ai.github.io/langgraph/) agents on
Microsoft Foundry using the
[`langchain_azure_ai.agents.hosting`](../../../libs/azure-ai/langchain_azure_ai/agents/hosting)
package. The hosting layer adapts any compiled LangGraph runnable into a
REST endpoint that speaks either the **Responses** protocol or the
**Invocations** protocol (or both, on the same port).

Each sample is a self-contained folder with its own `main.py`,
`README.md`, `requirements.txt`, `Dockerfile`, `agent.manifest.yaml`,
and `agent.yaml` so it can be run locally with `python main.py` or
deployed to Foundry with `azd`.

## Samples

### Responses protocol

| # | Sample | Description |
|---|--------|-------------|
| 1 | [Basic](responses/01_basic/) | Minimal `create_agent` graph hosted as the Responses API. |
| 2 | [Tools](responses/02_tools/) | `@tool` registration. Intermediate tool calls and results are surfaced as `function_call` / `function_call_output` output items. |
| 3 | [MCP](responses/03_mcp/) | Tools loaded at startup from a remote MCP server (GitHub by default) via `langchain_mcp_adapters.client.MultiServerMCPClient`. |
| 4 | [Foundry Toolbox](responses/04_foundry_toolbox/) | Tools loaded at startup from an Azure AI Foundry Toolbox (managed multi-MCP gateway) via `langchain_azure_ai.tools.AzureAIProjectToolbox`. |
| 5 | [Workflows](responses/05_workflows/) | Hand-built multi-node `StateGraph` (plan -> tools -> synthesize) hosted as **both** the Responses API and the Invocations API on the same port. |
| 6 | [Files](responses/06_files/) | Filesystem tools (`list_files`, `read_text_file`) that let the agent read files shipped with the container under a configurable data root. |
| 7 | [Observability](responses/07_observability/) | Standalone observability sample — minimal `create_agent` graph with GenAI OpenTelemetry tracing wired to the Foundry project's managed Application Insights, with `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true` enabled by default. |
| 8 | [HITL](responses/08_hitl/) | Approval-style human-in-the-loop using `langgraph.types.interrupt`. Pending interrupts surface as the standard OpenAI `mcp_approval_request` output item; the client approves (or rejects) via `mcp_approval_response` — the same flow OpenAI uses for MCP server tool approvals. A paired `function_call` item is also emitted for callers that need to override the proposed payload via `function_call_output`. |

> This tree follows the Agent Framework's [`foundry-hosted-agents`](https://github.com/microsoft/agent-framework/tree/main/python/samples/04-hosting/foundry-hosted-agents)
> folder structure and naming style. Observability coverage for the
> existing samples is documented in the [Tracing](#tracing) section
> below.

### Invocations protocol

| # | Sample | Description |
|---|--------|-------------|
| 1 | [Basic](invocations/01_basic/) | Minimal `create_agent` + `MemorySaver` checkpointer for multi-turn continuity via `agent_session_id` headers/URL params. |
| 2 | [Tools](invocations/02_tools/) | Repo-specific LangChain tools sample. The tool round-trip runs server-side and the client sees only the final assistant text (or token deltas when streaming). This differs from the Agent Framework's current `02_break_glass` sample. |

## Running the Agent Host Locally

You can run any sample two ways: through the Azure Developer CLI
(`azd ai agent run`, which builds the container and matches the
Foundry-hosted runtime), or directly with `python main.py` (faster
iteration, uses your local Python).

### Using `azd`

#### Prerequisites

1. **Azure Developer CLI (`azd`)**
   - [Install azd](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
     and the AI agent extension: `azd ext install azure.ai.agents`
   - Authenticate: `azd auth login`
2. **Azure subscription** with permission to create Foundry projects
   and model deployments (or an existing project you can re-use).
3. **Docker** running locally (required by `azd ai agent run` to build
   the container image declared in each sample's `Dockerfile`).

#### Create a new project

No cloning required. Make a new folder and point `azd` at the manifest
on GitHub for the sample you want:

```bash
mkdir my-langchain-agent && cd my-langchain-agent

# Initialize from the manifest (replace the URL with any sample's agent.manifest.yaml)
azd ai agent init -m https://github.com/langchain-ai/langchain-azure/blob/main/samples/hosting/langgraph-hosted-agents/responses/01_basic/agent.manifest.yaml
```

Follow the prompts from `azd ai agent init` to complete initialization.
If you don't have an existing Foundry project and a model deployment,
`azd ai agent init` will guide you through creating them.

#### Provision Azure resources

> Only needed if you don't have an existing Foundry project and model
> deployment.

```bash
azd provision
```

This creates a new resource group containing a Foundry account, a
Foundry project, the requested model deployment, an Application
Insights instance, and a container registry for the agent image.

#### Run the agent host

```bash
azd ai agent run
```

The host serves on `http://127.0.0.1:8088`.

#### Invoke the agent

In another terminal:

```bash
azd ai agent invoke --local "Hello!"
```

Or send a raw HTTP request — every sample's README contains specific
`curl` examples for the protocol(s) it exposes:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!"}'
```

PowerShell equivalent:

```powershell
(Invoke-WebRequest -Uri http://127.0.0.1:8088/responses `
  -Method POST -ContentType 'application/json' `
  -Body '{"input": "Hello!"}').Content
```

### Using `python`

#### Prerequisites

1. An existing Foundry project
2. A deployed model in your Foundry project
3. Azure CLI installed and authenticated (`az login`)
4. Python 3.10 or later

#### Environment setup

From the sample folder you want to run (e.g. `responses/01_basic/`):

```bash
cd responses/01_basic

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows (PowerShell)

# 2. Install per-sample dependencies
pip install -r requirements.txt

# 3. Copy and fill in env vars
cp .env.example .env
# edit FOUNDRY_PROJECT_ENDPOINT (and TOOLBOX_NAME for responses/04_foundry_toolbox)
```

Alternatively, install the shared dev dependencies once at this
directory's root using the editable install in
[`requirements.txt`](requirements.txt) and reuse the same virtualenv
across all samples.

#### Run the agent host

```bash
python main.py
```

The host serves on `http://127.0.0.1:8088`.

#### Invoke the agent

In another terminal, use the `curl` examples in the sample folder's
`README.md`. The minimal one:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!"}'
```

## Deploying the Agent to Foundry

Once you've tested locally, deploy to Microsoft Foundry.

### With an existing Foundry project

If you already have a Foundry project and the necessary Azure resources
provisioned, you can skip the provisioning step. After running
`azd ai agent init -m <agent.manifest.yaml>` and answering the prompts,
you will have an azd project ready for deployment.

### Setting up a new Foundry project

Follow the steps in [Using `azd`](#using-azd) above to set up the
project and provision the necessary Azure resources.

### Deploy

Once the project is set up and resources are provisioned:

```bash
azd deploy
```

This packages the agent into a container image (using each sample's
[`Dockerfile`](responses/01_basic/Dockerfile)), pushes it to the
provisioned container registry, and rolls it out to the Foundry
hosted-agent runtime.

> The Foundry hosting infrastructure injects the following environment
> variables into your agent at runtime:
>
> - `FOUNDRY_PROJECT_ENDPOINT` — the endpoint URL for the Foundry
>   project where the agent is deployed.
> - `AZURE_AI_MODEL_DEPLOYMENT_NAME` — the name of the model deployment
>   provisioned during `azd ai agent init`.
> - `APPLICATIONINSIGHTS_CONNECTION_STRING` — the connection string for
>   the project's Application Insights instance.
> - `TOOLBOX_NAME` — only for samples that declare a `kind: toolbox`
>   resource in their `agent.manifest.yaml` (e.g.
>   [responses/04_foundry_toolbox](responses/04_foundry_toolbox/)).
>
> Each sample's `main.py` reads these names directly, so no code
> changes are required between local-dev and deployed.

For the full deployment guide, see the [official deployment
docs](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent).
Once deployed, learn more about how to manage deployed agents in the
[official management guide](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-agent).

## Tracing

Each sample emits OpenTelemetry GenAI semantic-convention spans for every
LangGraph node, model call, and tool call via
[`AzureAIOpenTelemetryTracer`](../../../libs/azure-ai/langchain_azure_ai/callbacks/tracers/inference_tracing.py).
The destination is picked **purely from environment variables** — no code
edits needed. First matching rule wins:

| # | Env var(s) set | Where spans go |
|---|----------------|----------------|
| 0 | `OTEL_SDK_DISABLED=true` | nowhere — OTel SDK no-ops |
| 1 | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) | OTLP/HTTP collector at that URL |
| 2 | `APPLICATION_INSIGHTS_CONNECTION_STRING` | Azure Monitor directly |
| 3 | `FOUNDRY_PROJECT_ENDPOINT` (always set for the samples) | Foundry project's managed App Insights (connection string auto-resolved over the Foundry control plane; needs `az login` + Reader on the project) |
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
    # FOUNDRY_PROJECT_ENDPOINT (project-managed App Insights).
    enable_auto_tracing(auto_configure_azure_monitor=True)
```

### Quick recipes

**Foundry project's managed App Insights** (default for the samples,
because `FOUNDRY_PROJECT_ENDPOINT` is required for the chat model
anyway):

```bash
az login
# FOUNDRY_PROJECT_ENDPOINT already set in .env — nothing else to do.
python main.py
```

**Direct App Insights connection string** (bypasses the Foundry resolver
— faster startup, no Reader RBAC needed):

```bash
export APPLICATION_INSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=..."
python main.py
```

**Local OTel Collector**:

```bash
docker run --rm -p 4318:4318 otel/opentelemetry-collector:latest
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
python main.py
```

**Off**:

```bash
export OTEL_SDK_DISABLED=true
python main.py
```

Inside the **Foundry hosted-agent sandbox**, `azure-ai-agentserver-core`
has already configured the `TracerProvider` + Azure Monitor exporter for
its per-request server span; the langchain GenAI spans then nest under
that existing span. `enable_auto_tracing(auto_configure_azure_monitor=True)`
detects the existing provider and skips reconfiguring it, so the same
branch still works.
