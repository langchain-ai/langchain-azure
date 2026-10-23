# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) **human-in-the-loop**
agent hosted using the **Responses protocol**, modelled as a
**tool-call approval flow**: before any tool runs, the graph pauses
and surfaces the proposed call to the client as the **standard OpenAI
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

The graph has three nodes:

- `agent` — invokes the chat model.
- `approve_and_call_tool` — when the model emits a tool call, this
  node pauses with `langgraph.types.interrupt(proposed)` where
  `proposed` is the proposed tool name and arguments. On resume, it
  invokes the tool and emits a `ToolMessage`.

When the graph pauses, the host serializes the pending interrupt as
**two** output items in the same response, both referring to the same
LangGraph interrupt:

1. An **`mcp_approval_request`** item with a host-generated `mcpr_*` ID,
   `server_label == "langgraph"`, `name ==
   "__hosted_agent_adapter_interrupt__"`, and `arguments` JSON of the
   form:

   ```json
   {"interrupt_id": "<id>", "value": {"tool": "get_weather", "arguments": {"location": "Seattle"}}}
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
  "__hosted_agent_adapter_interrupt__"` and `call_id == interrupt.id` (advanced
  channel — see "Advanced resume" below).

The approval item's `arguments` JSON describes the proposed action:

```json
{"interrupt_id": "<langgraph interrupt id>", "value": {"tool": "get_weather", "arguments": {"location": "Seattle"}}}
```

### Step 2 — approve (or reject)

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

When you need to **override** the proposed tool call (e.g. change the
arguments) or drive a LangGraph `Command` with `update`/`goto` fields,
target the paired `function_call` item via the standard
`function_call_output` channel:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {"id": "demo-hitl-1"},
    "input": [
      {
        "type": "function_call_output",
        "call_id": "<call_id of the __hosted_agent_adapter_interrupt__ item>",
        "output": "{\"resume\": {\"tool\": \"get_weather\", \"arguments\": {\"location\": \"Vancouver\"}}}"
      }
    ]
  }'
```

The graph resumes with the client-supplied payload (`Vancouver` instead
of `Seattle`) and the tool is invoked with the overridden arguments.

This channel requires a resume value, either a raw payload or
`{"resume": ...}`. A rich command may also include `update` and `goto`;
control-only commands do not answer an interrupt.
Empty and whitespace-only string outputs are valid answers. An interrupt-ID
map must include the pending ID even for a single interrupt; to pass an empty
object as the answer, use `{"resume": {"<call_id>": {}}}`.

#### Complete-batch continuation

Both Responses and Invocations hosts require an answer for **every interrupt
currently pending on the selected checkpoint**, in one request. Collect all
answers on the client before submitting them; the host no longer accepts or
saves partial batches. Use `input` for Responses and the structured `message`
list for Invocations.

For function-call responses, missing results return HTTP 400:

```json
{
  "error": {
    "message": "No tool output found for function call <missing-call-id>.",
    "type": "invalid_request_error",
    "param": "input",
    "code": null
  }
}
```

For MCP responses, the message is
`The following MCP approval requests do not have an approval: <missing-ids>`
and `param` is `null`. `approve: false` is a response, not a missing decision;
the existing rejection semantics described above are unchanged.

Incomplete requests do not advance the graph, start SSE, or admit background
work. Retry the complete batch against the same parent. A paired
`function_call` / `mcp_approval_request` represents one interrupt, so either
channel can answer it. A later sequential interrupt starts a new batch.

Both hosts expose `validate_resume_input` after input normalization. Custom
Responses resume protocols can keep their `build_resume_command` override;
Invocations can keep a `build_input` override returning a `Command`. These hooks
may run at admission and again at execution, so they must not mutate graph
state or retain partial answers.

For custom Invocations checkpoint selection, use `build_checkpoint_config`
rather than changing identity in the execution-only task config hook. See the
[hook contract and migration example](../../../../../libs/azure-ai/README.md#continuation-hooks).

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
