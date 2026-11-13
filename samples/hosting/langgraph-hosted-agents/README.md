# LangGraph Hosted Agent Samples

This directory contains samples that demonstrate how to host
[LangGraph](https://langchain-ai.github.io/langgraph/) agents on
Microsoft Foundry using the
[`langchain_azure_ai.agents.hosting`](../../../libs/azure-ai/langchain_azure_ai/agents/hosting)
package. The hosting layer adapts any compiled LangGraph runnable into a
REST endpoint that speaks either the **Responses** protocol or the
**Invocations** protocol (or both, on the same port).

All samples use the Responses API for model calls, with
`use_responses_api=True` and `output_version="responses/v1"`. We recommend
Responses when your model and endpoint support it. If you prefer Chat
Completions and your model and endpoint support it, set
`use_responses_api=False` and remove `output_version="responses/v1"` from
the model configuration. This choice is independent of the host
protocol: both Responses and Invocations hosts can use either model API.

Each sample has its own code, dependencies, and README for local runs.

## Setup paths

Choose its `azd` setup path before running or deploying:

| Sample configuration | Setup path |
| --- | --- |
| `agent.manifest.yaml` and `agent.yaml` | Follow [Using azd](#using-azd) below. |
| `azure.yaml` in [Responses Resilient](responses/10_resilient/README.md#deploying-the-agent-to-foundry) or [Invocations Resilient](invocations/04_resilient/README.md#deploying-the-agent-to-foundry) | Initialize the independent azd project using that sample's deployment steps. |
| `azure.yaml` in [Deep Agents](responses/12_deep_agents/SETUP.md) | Complete its resource setup and deployment steps. |
| Both formats in [Branching](responses/13_branching/README.md#run-with-azd) | Follow the sample's existing-project setup. |

## Start here

**New to hosting? Start with [Responses Basic](responses/01_basic/README.md).**
It is the smallest end-to-end example. Use the
[LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents)
for the concepts and protocol comparison.

| Your goal | Start with |
| --- | --- |
| Host an existing compiled graph without changing its implementation | [Responses run entrypoint](responses/09_run/README.md) or [Invocations run entrypoint](invocations/03_run/README.md); point `langgraph.json` at your exported graph |
| Build a custom multi-node workflow | [Responses Workflows](responses/05_workflows/README.md) |
| Persist graph state in Foundry State Store | [Responses Workflows](responses/05_workflows/README.md) uses `FoundryCheckpointSaver` when hosted and an in-memory saver locally; see [durable graph state](../../../libs/azure-ai/README.md#durable-graph-state) |
| Use the Invocations protocol with session continuity | [Invocations Basic](invocations/01_basic/README.md) |
| Add local, remote MCP, or managed Toolbox tools | [Tools](responses/02_tools/README.md), [MCP](responses/03_mcp/README.md), or [Foundry Toolbox](responses/04_foundry_toolbox/README.md) |
| Add tracing or human approval | [Observability](responses/07_observability/README.md) or [HITL](responses/08_hitl/README.md) |
| Recover background work after a restart | [Responses Resilient](responses/10_resilient/README.md) or [Invocations Resilient](invocations/04_resilient/README.md) |
| Branch, rewind, or regenerate from a completed response | [Responses Branching](responses/13_branching/README.md) |

After choosing a sample, follow [local setup](#running-the-agent-host-locally),
use its README for requests, then follow [deployment](#deploying-the-agent-to-foundry).

## Samples

### Responses protocol

| # | Sample | Description |
|---|--------|-------------|
| 1 | [Basic](responses/01_basic/) | Minimal `create_agent` graph hosted as the Responses API. |
| 2 | [Tools](responses/02_tools/) | `@tool` registration. Intermediate tool calls and results are surfaced as `function_call` / `function_call_output` output items. |
| 3 | [MCP](responses/03_mcp/) | Tools loaded at startup from a remote MCP server (GitHub by default) via `langchain_mcp_adapters.client.MultiServerMCPClient`. |
| 4 | [Foundry Toolbox](responses/04_foundry_toolbox/) | Tools loaded at startup from an Azure AI Foundry Toolbox (managed multi-MCP gateway) via `langchain_azure_ai.tools.AzureAIProjectToolbox`. |
| 5 | [Workflows](responses/05_workflows/) | Hand-built multi-node `StateGraph` (plan -> tools -> synthesize) hosted as the Responses API. |
| 6 | [Files](responses/06_files/) | Image and file request attachments passed directly to a Responses-compatible model. |
| 7 | [Observability](responses/07_observability/) | Standalone observability sample — minimal `create_agent` graph with GenAI OpenTelemetry tracing wired to the Foundry project's managed Application Insights, with `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true` enabled by default. |
| 8 | [HITL](responses/08_hitl/) | Approval-style human-in-the-loop using `langgraph.types.interrupt`. Pending interrupts surface as the standard OpenAI `mcp_approval_request` output item; the client approves (or rejects) via `mcp_approval_response` — the same flow OpenAI uses for MCP server tool approvals. A paired `function_call` item is also emitted for callers that need to override the proposed payload via `function_call_output`. |
| 9 | [Run entrypoint](responses/09_run/) | The basic `create_agent` graph loaded from `langgraph.json` and hosted through `langchain_azure_ai.agents.hosting.run`. |
| 10 | [Resilient](responses/10_resilient/) | Resilient background Responses with replayable SSE, crash recovery, durable checkpoints, HITL approval, cancellation, and optional active-turn steering. Includes a Textual recovery client. |
| 11 | [LangChain middleware](responses/11_middleware/) | Native `HumanInTheLoopMiddleware` approve/edit/reject plus the official `@before_model` message-limit function. |
| 12 | [Deep Agents](responses/12_deep_agents/) | Official Deep Research workflow with planning, research subagents, Foundry web search, checkpoint-backed virtual files, and Azure Blob Storage integration for downloadable reports. |
| 13 | [Branching](responses/13_branching/) | Remember a number to demonstrate branching, rewind, and regeneration with `previous_response_id`. |

> This tree follows the Agent Framework's [`foundry-hosted-agents`](https://github.com/microsoft/agent-framework/tree/main/python/samples/04-hosting/foundry-hosted-agents)
> folder structure and naming style. Observability coverage for the
> existing samples is documented in the [Tracing](#tracing) section
> below.

### Invocations protocol

| # | Sample | Description |
|---|--------|-------------|
| 1 | [Basic](invocations/01_basic/) | Minimal `create_agent` + `MemorySaver` checkpointer for multi-turn continuity via `agent_session_id` headers/URL params. |
| 2 | [Tools](invocations/02_tools/) | Repo-specific LangChain tools sample. The tool round-trip runs server-side and the client sees only the final assistant text (or token deltas when streaming). This differs from the Agent Framework's current `02_break_glass` sample. |
| 3 | [Run entrypoint](invocations/03_run/) | The basic `create_agent` + `MemorySaver` graph loaded from `langgraph.json` and hosted through `langchain_azure_ai.agents.hosting.run`. |
| 4 | [Resilient](invocations/04_resilient/) | Resilient foreground and background Invocations with stable invocation IDs, crash recovery, durable checkpoints, HITL approval, retrieval, cancellation, and optional steering. Includes a Textual recovery client. |

## Running the Agent Host Locally

You can run any sample two ways: through the Azure Developer CLI
(`azd ai agent run`, which builds the container and matches the
Foundry-hosted runtime), or directly with `python main.py` (faster
iteration, uses your local Python).

### Using `azd`

The initialization steps below apply to samples with `agent.manifest.yaml`.
For `azure.yaml` projects, use the [setup paths](#setup-paths)
above after installing the prerequisites.

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
Foundry project, the requested model deployment, and supporting resources.
For both new and existing projects, [connect or verify Application
Insights](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-agent-setup#connect-application-insights-to-your-foundry-project)
before deploying. For local tracing, follow the sample's connection-string
configuration in [Tracing](#tracing).

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
4. Python 3.11 through 3.14

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

Most samples start with:

```bash
python main.py
```

The configuration-driven samples instead start with:

```bash
# responses/09_run
python -m langchain_azure_ai.agents.hosting.run --protocol responses

# invocations/03_run
python -m langchain_azure_ai.agents.hosting.run --protocol invocations
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
The steps below use `agent.manifest.yaml`. For `azure.yaml` projects,
follow the sample-specific [setup paths](#setup-paths).

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
> The samples and hosting SDK consume these values, so no code
> changes are required between local-dev and deployed.

For the full deployment guide, see the [official deployment
docs](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent).
Once deployed, learn more about how to manage deployed agents in the
[official management guide](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-agent).

## Tracing

The samples enable OpenTelemetry tracing for agent, model, and tool calls.
Most call
[`enable_auto_tracing()`](../../../libs/azure-ai/langchain_azure_ai/callbacks/tracers/auto_instrument.py)
to attach `AzureAIOpenTelemetryTracer`. The two resilient samples,
[Responses](responses/10_resilient/) and
[Invocations](invocations/04_resilient/), use Agent Server's built-in Microsoft
OpenTelemetry instrumentation instead. Internal graph nodes may be omitted.

For traces in Foundry, [connect Application Insights to the
project](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-agent-setup#connect-application-insights-to-your-foundry-project)
and grant the caller permission to query traces. Verify this after provisioning
a new project or selecting an existing one, before deploying the agent.

The connection-string variable depends on which instrumentation the sample uses:

| Destination | Samples calling `enable_auto_tracing()` | The two resilient samples |
| --- | --- | --- |
| OTLP/HTTP collector | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` | Same variables |
| Azure Monitor directly | `APPLICATION_INSIGHTS_CONNECTION_STRING` | `APPLICATIONINSIGHTS_CONNECTION_STRING` |
| Project-connected App Insights | Auto-resolved from `FOUNDRY_PROJECT_ENDPOINT` when no direct connection string is set; requires project read access | Foundry injects the connection string when hosted; set it explicitly for local runs |

In samples calling `enable_auto_tracing()`, an OTLP endpoint selects the
OTLP exporter instead of automatically configuring Azure Monitor. Agent Server
can export to both destinations when both are configured. These settings choose
where recorded spans are sent; they do not disable tracing.

The samples retain the SDK's default sampling policy. Sampling may omit some
spans while tracing remains enabled. Set
`OTEL_SDK_DISABLED=true` to disable tracing; no sample sets it by default.

Content capture is separate from tracing. For `AzureAIOpenTelemetryTracer`, use
`AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED`; for Agent Server's Microsoft
instrumentation, use `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
See each sample for its content-capture defaults.

### Quick recipes

**Project-connected App Insights for samples calling `enable_auto_tracing()`**:

```bash
az login
# Set FOUNDRY_PROJECT_ENDPOINT in .env and connect App Insights to that project.
python main.py
```

**Direct App Insights connection string** (bypasses the Foundry resolver
— faster startup, no Reader RBAC needed):

```bash
export APPLICATION_INSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=..."
python main.py
```

For either resilient sample, use `APPLICATIONINSIGHTS_CONNECTION_STRING`
(without the underscore between `APPLICATION` and `INSIGHTS`) instead.

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

After invoking a deployed agent, check **Build > Agents > Traces** in Foundry.
Open the new trace and check its agent, model, and tool spans. For a resilient
sample, approve the booking and confirm that the resumed turn includes the
`book_trip` tool call. Allow a few minutes for ingestion.

The local exporter regression check exercises every hosted sample through
HTTP with real Azure Monitor and OTLP exporters. It checks model spans, tool
spans for tool-using scenarios, and approval/resume spans where applicable.
The test explicitly sets `OTEL_TRACES_SAMPLER=always_on` to check all expected
spans; it does not verify full retention under the default sampling policy:

```bash
python -m pip install -r samples/hosting/tests/requirements.txt
python -m pytest samples/hosting/tests/test_tracing.py -q
```

Run these commands from the repository root. See the
[test guide](../tests/README.md) for coverage and adding new samples. The test
substitutes remote models, MCP/Toolbox services, and cloud storage and receives
exports locally; Azure ingestion and Foundry visibility require the
deployed-agent check above.
