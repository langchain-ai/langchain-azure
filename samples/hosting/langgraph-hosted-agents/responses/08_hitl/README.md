# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) **human-in-the-loop**
agent hosted using the **Responses protocol**, modelled as a
**tool-call approval flow**: before any tool runs, the graph pauses
and surfaces the complete model-turn batch to the client as the **standard OpenAI
`mcp_approval_request` output item**. Any Responses-API client that
already supports MCP server approvals (e.g. the OpenAI Python SDK)
can drive this agent without code changes.

The host also emits a paired `function_call` item for the same interrupt, so
clients that need to override the approval payload (or send a richer
LangGraph `Command`) can use the standard `function_call_output` channel
as an alternative.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` with an Azure bearer token
provider from `DefaultAzureCredential` and an OpenAI-compatible endpoint
from `azure.ai.projects.AIProjectClient`.

See [main.py](main.py) for the full implementation.

### Approval-style HITL

The graph has two nodes:

- `agent` — invokes the chat model.
- `approve_and_call_tool` — pauses **for the whole model turn** with
  `langgraph.types.interrupt({"tool_calls": [...]})`, preserving every call's
  ID, name, and arguments. A standard MCP approval approves the whole batch.
  The rich channel accepts per-call approve/edit/reject decisions, matched by
  call ID rather than list position. The node validates the entire decision
  set and every accepted call's final arguments against its tool input schema
  before executing any tools, then emits one `ToolMessage` per original ID.

Do not loop over `interrupt()` in the same node to ask separate approval
questions: those pauses share an interrupt ID, so retrying the previous
approval could approve the next call. Every review here presents the **same
complete batch**; validation retries never advance to another individual call.
All tool execution remains after successful validation, so invalid decisions
cannot repeat earlier tool executions on [node replay](https://docs.langchain.com/oss/python/langgraph/interrupts#rules-of-interrupts).

When the graph pauses, the host serializes the pending interrupt as
**two** output items in the same response, both referring to the same
LangGraph interrupt:

1. An **`mcp_approval_request`** item with its own approval `id`,
   `server_label == "langgraph"`, `name ==
   "__hosted_agent_adapter_interrupt__"`, and `arguments` JSON of the
   form:

   ```json
   {"interrupt_id": "<id>", "value": {"tool_calls": [{"id": "<tool call id>", "name": "get_weather", "arguments": {"location": "Seattle"}}]}}
   ```

2. A `function_call` item with the same `name` and `call_id ==
   interrupt.id` carrying the same `arguments`. (Parallel rich channel
   for callers that need to drive an arbitrary LangGraph `Command`.)

State is persisted by a checkpointer keyed by the `conversation.id`, so the
second request continues the paused run from exactly where it left off. Local
runs use `InMemorySaver`; Foundry-hosted runs use `FoundryCheckpointSaver` so
an approval pause survives container replacement.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.ResponsesHostServer`](../../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

### Step 1 — ask the agent a question that requires a tool

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the weather in Seattle?", "conversation": {"id": "demo-hitl-1"}}'
```

The response `output` array will contain:

- a `function_call` item with `name == "get_weather"` — the LLM's own
  tool call. **Do not** reply to this one; the graph closes it itself
  on resume.
- an `mcp_approval_request` item — the OpenAI-standard approval
  prompt. **Copy its `id`** to use in Step 2.
- a paired `function_call` item with `name ==
  "__hosted_agent_adapter_interrupt__"` and `call_id` equal to the raw
  LangGraph interrupt ID (advanced
  channel — see "Advanced resume" below).

The approval item's `arguments` JSON describes every call from that model turn:

```json
{
  "interrupt_id": "<langgraph interrupt id>",
  "value": {
    "tool_calls": [
      {"id": "<tool call id>", "name": "get_weather", "arguments": {"location": "Seattle"}}
    ]
  }
}
```

Ask for Seattle and Paris together to see both calls in one approval prompt
when the model proposes them in the same turn.

### Step 2 — approve (or reject) the entire batch

#### Approve — `mcp_approval_response` with `approve: true`

Post an `mcp_approval_response` whose `approval_request_id` matches
the `id` of the `mcp_approval_request` item:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {"id": "demo-hitl-1"},
    "input": [
      {
        "type": "mcp_approval_response",
        "approval_request_id": "<id of the mcp_approval_request item>",
        "approve": true
      }
    ]
  }'
```

The host resumes the graph with the original `proposed` payload echoed
back; `approve_and_call_tool` invokes `get_weather`, and the agent
returns a final assistant `message` item.

This is the same wire flow OpenAI's Responses API uses for MCP server
tool approvals — any standard Responses client will already know how
to render it.

The host treats the complete LangChain `HumanInTheLoopMiddleware` request
shape (`action_requests` plus matching `review_configs`) as reserved for this
boolean approval shortcut. It expands `approve: true` or `false` into one
ordered middleware decision per action, and requires every action to allow
that decision. A custom `interrupt()` using the same value shape is interpreted
the same way; use the paired `function_call_output` channel when the custom
value must be returned unchanged.

#### Reject — `mcp_approval_response` with `approve: false`

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {"id": "demo-hitl-1"},
    "input": [
      {
        "type": "mcp_approval_response",
        "approval_request_id": "<id of the mcp_approval_request item>",
        "approve": false,
        "reason": "user canceled"
      }
    ]
  }'
```

The host short-circuits the turn into:

```json
{
  "status": "failed",
  "error": {
    "code": "interrupt_rejected",
    "message": "Interrupt '<id>' was rejected by the client: user canceled"
  }
}
```

The pending interrupt remains in the checkpoint, so the next request
can retry with a different decision.

### Advanced resume — `function_call_output` (override the proposed payload)

To decide per call, target the paired `function_call` item via
`function_call_output`. Include exactly one decision for every proposed tool
call ID. `approve` keeps the original arguments, `edit` supplies replacement
arguments without changing the tool name, and `reject` returns a rejection
`ToolMessage` without executing that call. Missing, duplicate, or unknown IDs
and invalid decision types are rejected before any tools execute. Edited
arguments must also satisfy the tool's input schema; a dictionary alone is
not enough (for example, `get_weather` requires `location`). Decisions may be
returned in any order.

An invalid review does **not** fail and strand the paused turn. The response
contains a new approval prompt with the same original `tool_calls` and a
`validation_error` explanation. Send a complete corrected decision set to that
prompt's paired `function_call.call_id`, or use the latest MCP approval to
approve the **original** proposed arguments, discarding the invalid edits.
Repeating an invalid review simply re-prompts for that same batch.

This preflight prevents argument-validation failures from partially executing
a batch. It is not a transaction or an exactly-once guarantee for failures
during tool execution.

For a batch containing a single Seattle call, this changes its arguments:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {"id": "demo-hitl-1"},
    "input": [
      {
        "type": "function_call_output",
        "call_id": "<call_id of the __hosted_agent_adapter_interrupt__ item>",
        "output": "{\"resume\": {\"decisions\": [{\"id\": \"<tool call id>\", \"type\": \"edit\", \"arguments\": {\"location\": \"Vancouver\"}}]}}"
      }
    ]
  }'
```

For a two-call batch, a combined per-call resume value could be:

```json
{
  "decisions": [
    {"id": "<Paris tool call id>", "type": "reject"},
    {"id": "<Seattle tool call id>", "type": "edit", "arguments": {"location": "Vancouver"}}
  ]
}
```

The standard `approve: false` shortcut rejects the entire batch at the host
boundary; a rich `reject` decision skips only that specific call. Do not send
partial decisions as separate resumes. A later model turn gets a new batch
interrupt, so retrying an earlier batch's approval cannot approve the new one.

#### Conflict resolution

If a single request contains **both** a matching `function_call_output`
and a matching `mcp_approval_response` for the same interrupt id, the
`function_call_output` wins (it carries the richer payload) and a
warning is logged server-side.

Without a valid matching `function_call_output`, if one request contains both
`approve: true` and `approve: false` for the same interrupt, the whole request
fails and all interrupts remain pending.

The same failure behavior applies when any action in a recognized
`HumanInTheLoopMiddleware` request does not allow the selected decision. The
shortcut is atomic: the host does not partially approve or reject its actions.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.

The deployment descriptors declare Responses protocol `2.0.0`, which supplies
the hosted request context required by `FoundryCheckpointSaver`.
