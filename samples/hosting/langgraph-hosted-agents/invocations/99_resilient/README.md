# Sample 99 - Resilient Invocations with LangGraph checkpointing

> **Work in progress / experimental.** This sample demonstrates how
> `InvocationsHostServer` combines Agent Server's resilient Invocations
> protocol with LangGraph checkpoint recovery.

## Overview

The sample hosts a real-model trip-planning `StateGraph`. Flight and hotel
searches run automatically, while `book_trip` pauses at a durable LangGraph
`interrupt()` until the client approves or denies the tool call.

```text
START -> agent -> [search tools | approval] -> agent -> END
```

It demonstrates:

- background Invocations retrieved by stable invocation ID;
- exact LangGraph checkpoint recovery after a process restart;
- linear multi-turn sessions linked by `previous_invocation_id`;
- durable human approval before a sensitive tool executes;
- foreground streaming plus retrieval and cancellation routes; and
- client recovery from connection failures, interrupted SSE streams, and HTTP
  `5xx` responses.

Recovery depends on two persistent layers: Agent Server stores the durable
invocation and protocol events, while `AsyncSqliteSaver` stores LangGraph
workflow state. Both must remain available after the host restarts.

## Prerequisites

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/)
- Azure CLI authenticated with `az login`
- A Microsoft Foundry project and model deployment accessible through
  `DefaultAzureCredential`

Create a `.env` file in this directory:

```dotenv
FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4.1-mini"
```

See the [parent sample guide](../../README.md#running-the-agent-host-locally)
for general Foundry setup options.

## Run locally

From this directory, start the host:

```bash
uv sync
uv run python main.py
```

The Invocations endpoint is available at
`http://127.0.0.1:8088/invocations` by default.

In another terminal, start the Textual CUI:

```bash
cd client
uv sync
uv run python client.py --session-id trip-demo
```

Ask it to book a trip. The CUI displays the proposed `book_trip` arguments
when the graph pauses; choose **Approve** to continue or **Deny** to reject the
tool call.

The CUI creates a stable invocation ID for every turn, reuses the same
`agent_session_id`, and links turns with `previous_invocation_id`. It streams
foreground results and polls the same invocation ID when the connection or SSE
stream is interrupted. Omit `--session-id` to generate a random session ID.

Useful client options:

| Option | Purpose |
| --- | --- |
| `--url` | Host base URL or full Invocations endpoint. Defaults to the local host. |
| `--session-id` | Stable agent session ID shared by all turns. |
| `--auth` | Acquire an Azure AI bearer token for a deployed agent. |
| `--reconnect-timeout` | Seconds to keep recovering an interrupted turn. Defaults to 120. |

## Test crash recovery

Start the host with an isolated Agent Server state directory:

```bash
AGENTSERVER_STATE_ROOT="$PWD/.agentserver-demo" uv run python main.py
```

Start the CUI in another terminal:

```bash
cd client
uv run python client.py \
  --session-id crash-demo \
  --reconnect-timeout 300
```

Enter:

```text
Call simulate_crash, recover, and report the result.
```

The tool terminates the host on its first execution. Restart the host with the
same command, from the same directory, before the client timeout expires. The
CUI polls the same invocation and resumes it from the paired LangGraph
checkpoint; do not submit the original request again.

The local LangGraph database is `checkpoints.sqlite` in this directory. The
host reclaims a stale local replay-stream lock only after Agent Server re-enters
the owning invocation in recovered mode, so manual lock deletion is not
required.

## Protocol reference

### Create and retrieve an invocation

Choose the invocation ID before create and reuse it for all recovery requests:

```bash
curl -X POST \
  "http://127.0.0.1:8088/invocations?agent_session_id=trip-demo" \
  -H "Content-Type: application/json" \
  -H "x-agent-invocation-id: <invocation-id>" \
  -d '{
    "message": "Book a two-night trip to Paris",
    "background": true
  }'
```

A background create returns `202`. Poll the invocation by that same ID until
it reaches a terminal status:

```bash
curl "http://127.0.0.1:8088/invocations/<invocation-id>"
```

For foreground SSE output, send `"stream": true` instead of
`"background": true` and consume events through `event: done`. The two modes
cannot be combined. The sample CUI uses foreground streaming and falls back to
retrieval when the stream is interrupted.

### Approve the booking

The first invocation completes with an `mcp_approval_request` in its `output`
array. Use its `id` in the next invocation, keep the same session, and link the
completed turn with `previous_invocation_id`:

```bash
curl -X POST \
  "http://127.0.0.1:8088/invocations?agent_session_id=trip-demo" \
  -H "Content-Type: application/json" \
  -H "x-agent-invocation-id: <next-invocation-id>" \
  -d '{
    "message": [{
      "type": "mcp_approval_response",
      "approval_request_id": "<approval-request-id>",
      "approve": true
    }],
    "previous_invocation_id": "<invocation-id>",
    "background": true
  }'
```

The sample CUI constructs both approval and denial responses automatically.

### Cancel an invocation

```bash
curl -X POST \
  "http://127.0.0.1:8088/invocations/<invocation-id>/cancel"
```

Cancellation stops future work but does not roll back completed checkpoints or
external effects.

## Recovery contract

### Client behavior

| Condition | Required action |
| --- | --- |
| Connection failure, SSE termination without `event: done`, or HTTP `5xx` | Retrieve the same stable invocation ID until it becomes terminal or the reconnect timeout expires. |
| Retrieval returns `404` before create was admitted | Retry create with the same invocation ID. Never generate a replacement ID. |
| Other HTTP `4xx` or an explicit terminal protocol event | Treat the result as final; do not retry it. |
| Starting the next turn | Reuse the `agent_session_id` and send the latest invocation ID as `previous_invocation_id`. |

Each turn needs a stable `x-agent-invocation-id` chosen before create. Sessions
are linear: a new turn continues from the latest completed invocation rather
than forking an older checkpoint.

### Graph and handler behavior

- Compile the graph with a durable checkpointer that survives process
  replacement and is accessible to every recovering host instance.
- Keep durable workflow data in LangGraph state. Process memory, local caches,
  active HTTP requests, `InvocationContext`, and cancellation events are
  transient.
- Make nodes replay-safe. A crash after an external action but before the next
  paired checkpoint can execute that action again.
- Make external side effects idempotent, or deduplicate them with a stable
  operation key. At-least-once execution applies to writes, payments, email,
  queue publication, and other mutating tool calls.
- Keep checkpointed state serializable and compatible across deployments.

Before using this pattern in production, crash-test every node boundary and
both sides of each external side effect. Replace the sample's local SQLite and
file-backed stores with durable stores suitable for the deployment topology.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8088` | HTTP port for the agent host. |
| `AGENTSERVER_STATE_ROOT` | `~/.agentserver` | Local durable task, invocation, and protocol-event state. Reuse it across local restarts. |
| `CHECKPOINT_DB` | `checkpoints.sqlite` locally; `$HOME/checkpoints.sqlite` when hosted | LangGraph checkpoint database. An explicit value takes precedence. |
| `STEERABLE_CONVERSATIONS` | `false` | Enable server-side active-turn steering support. |
| `FOUNDRY_PROJECT_ENDPOINT` | None | Required Foundry project endpoint. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | None | Required Foundry model deployment name. |

## Deploy to Foundry

This directory is an independent `azd` project. The deployment script builds
the repository's current `libs/azure-ai` package into `vendor/`, provisions the
model declared in `azure.yaml`, and deploys the Invocations service.

For the first deployment, run in PowerShell:

```powershell
.\deploy.ps1 `
  -Environment resilient `
  -SubscriptionId "<subscription>" `
  -Location "<region>"
```

The script deploys `langchain-azure-resilient-invocations`. The provisioned
project and model outputs are stored in the `azd` environment. Subsequent
deployments can reuse them:

```powershell
.\deploy.ps1
```

Connect the CUI to the deployed Invocations endpoint with Azure
authentication:

```bash
cd client
uv run python client.py \
  --url <hosted-agent-invocations-url> \
  --auth
```