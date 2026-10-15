# LangChain middleware on Foundry

This sample demonstrates that an existing LangChain agent can keep its
middleware when hosted on Foundry. HITL is the built-in example, not a new
approval framework. The simulated email tool never sends real email.

## Official examples used

| LangChain reference | What stays native here |
| --- | --- |
| [HITL configuration and decisions](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) | `create_agent`, `HumanInTheLoopMiddleware`, approve/edit/reject, and checkpointed state. |
| [Custom middleware: node-style hooks](https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks) | A user-written `AgentMiddleware.before_model` message-limit hook and `jump_to="end"`. |
| [Custom middleware: state updates](https://docs.langchain.com/oss/python/langchain/middleware/custom#state-updates) | A user-written `@after_model` hook that persists `model_call_count` across the review pause. |

The hook patterns are adapted from those guides; the tool and prompt are
local demonstrations. The fixed LangChain version uses `action_requests[].args`
(see its [typed request/decision contract](https://github.com/langchain-ai/langchain/blob/f4bc5031dbcf24edb0374a07830915a285222567/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py)).
The live guide may show a newer or different payload spelling; use the
installed version's contract rather than translating the agent's middleware.

## Separation of responsibilities

- [agent.py](agent.py) contains only ordinary LangChain/LangGraph application
  code. It does not import the Foundry SDK, call `interrupt()`, or dispatch tools
  manually. An existing application can retain its own middleware and tools.
- [main.py](main.py) selects the Foundry-backed model, tracing, and an application
  checkpointer: local `InMemorySaver`, or `FoundryCheckpointSaver` in a hosted
  runtime. It runs the standard `ResponsesHostServer.run_async()` inside a
  resource context; it does not translate decisions or subclass a host.
- [client.py](client.py) is a reviewer for the existing Responses transport;
  decisions still use LangChain's schema.

The explicit startup owns the credential, model HTTP transports, and async
checkpoint lifetime. It releases them after normal shutdown, startup failure,
or cancellation. The project client used to discover the model endpoint is
closed before serving; the model's transports stay open until the host exits.
This is the async-resource lifecycle exception to configuration-only hosting,
not a middleware-specific adapter. An existing agent with externally managed
resources can continue using the configuration-driven run entrypoint.

The custom message limit is 50 accumulated messages, not a token limit. The
counter is observable in checkpoint state and logs, not an extra Responses
field. The tests assert its value, so a final answer alone cannot mask a
hook that never ran.

## Install and configure

Requires Python 3.11+ and an environment with access to a Foundry project
and a tool-capable model deployment. From this sample directory, use a
virtual environment and install the pinned public dependencies:

```bash
python -m pip install -r requirements.txt
```

The primary path uses published `langchain-azure-ai==1.2.10`; no source SDK
patch is required. The graph/model package versions are pinned; this file
is not a complete transitive dependency lock.

Copy [.env.example](.env.example) to `.env`, set the project endpoint and model
deployment, and authenticate with `az login`. Do not commit tokens or `.env`.
For a deployed agent, configure its identity to access the model and Foundry
state storage. The sample does not grant roles or create resources itself.

Start the standard host without adding a custom web server to the agent:

```bash
python main.py
```

Tracing follows the [shared SDK sample convention](../../README.md#tracing):
set an OTLP endpoint, an Application Insights connection string, or use the
Foundry project's managed Application Insights. Set `OTEL_SDK_DISABLED=true`
to skip tracing setup. Message content recording remains opt-in. A locally
created OTLP provider is shut down with the host; an existing platform
provider remains owned by the platform.

## Review an action

In another terminal, run the client from this directory:

```bash
python client.py
```

It prints the pending action and waits for a human to enter approve, edit,
or reject. Nothing executes before that decision. An edit changes only the
recipient, preserving the proposed subject/body; rejection supplies feedback
and does not execute the tool. The final simulated receipt shows what
actually ran. If the model returns no review, the client stops rather than
pretending approval happened.

For repeatable demonstrations, choose a decision explicitly. Each invocation
starts a fresh conversation; the second request uses the returned
`previous_response_id`. Optional `--stream` tests SSE on both requests:

```bash
python client.py --decision approve
python client.py --decision edit --recipient bob@example.com --stream
python client.py --decision reject
```

These flags apply the displayed decision to **all** actions in their original
order; they are not an unattended approval policy for production. Use an
appropriate reviewer/UI for real side effects. If a model requests further
review after the second response, inspect that response and review again;
the client deliberately does not auto-approve a loop.

### The transport boundary

The host exposes each pause as a paired approval item and a `function_call`
with name `__hosted_agent_adapter_interrupt__`. Read the latter's `call_id`
and `arguments.value` to inspect the native `action_requests`/`review_configs`.
The client sends `function_call_output` rather than using the MCP boolean
shortcut: that explicit rich channel works with the pinned SDK release
and preserves edit/reject and multiple-action decisions without a host patch.

One decision per reviewed action is required. The `output` field is a JSON
string containing the existing LangGraph command envelope. For one approval:

```json
{
  "type": "function_call_output",
  "call_id": "<interrupt-id>",
  "output": "{\"resume\": {\"<interrupt-id>\": {\"decisions\": [{\"type\": \"approve\"}]}}}"
}
```

The interrupt-keyed `resume` map is LangGraph's existing addressing convention,
not middleware logic invented by this sample. A rejected middleware action
is a normal completed response with feedback, not the hand-written-interrupt
`interrupt_rejected` transport error described in [08_hitl](../08_hitl/).
A completed response can still contain a pending interrupt: inspect output
items instead of equating response completion with tool execution.

## Run on Foundry

Use the normal [SDK sample deployment steps](../../README.md#deploying-the-agent-to-foundry)
with this folder's [agent.manifest.yaml](agent.manifest.yaml),
[agent.yaml](agent.yaml), and [Dockerfile](Dockerfile). Run `azd` initialization
and deployment only after selecting the intended subscription/project. The
image starts the same `python main.py` command as local development;
there is no middleware-specific server or platform plugin.

For this Docker-based sample, select container deployment so its command is
preserved. For example, from the sample folder:

```bash
azd ai agent init -m ./agent.manifest.yaml --deploy-mode container
azd deploy
```

If you choose code deployment instead, set the generated service entry point
to `main.py`. Provision resources only if the selected project needs them.

After deploying, invoke the existing agent through the project's Responses
endpoint (not the model endpoint):

```bash
python client.py --agent-name <deployed-agent-name> --decision approve
python client.py --agent-name <deployed-agent-name> --decision edit --recipient bob@example.com --stream
python client.py --agent-name <deployed-agent-name> --decision reject
```

The client uses `AIProjectClient.get_openai_client()` plus
`extra_body.agent_reference` and carries the same reference on resume.
The hosted branch uses Responses protocol 2.0.0 and durable Foundry-backed
checkpoints. Local memory checkpoints do not survive process restarts.
Existing applications retain their own persistence choices; this sample's
checkpointer selection is not a migration requirement.

## Verification and scope

From `libs/azure-ai` in a repository test environment with the hosting extra:

```bash
python -m pytest tests/unit_tests/agents/hosting/hitl/test_middleware_sample.py -q
```

The deterministic tests load the real sample and real native middleware,
using a scripted model and the HTTP host. They compare native graph execution
with approve/edit/reject over JSON/SSE, test conversation and response-ID
continuations, verify no tool call before approval, check decision order for
two actions, and verify the custom hook state and early-exit behavior.
They check startup resource cleanup on normal exit, failures, and cancellation,
plus tracing destination selection and local-provider cleanup.
They also recreate the graph/host and resume via the real
`FoundryCheckpointSaver` with an in-memory stand-in for Azure storage.
Azure storage/model calls are not part of those offline tests.

This proves these middleware patterns, not every LangChain middleware or
client UI. A cloud run additionally needs the target model, identity, and
storage configuration; run the three client cases above on the deployed
version before claiming live Foundry end-to-end acceptance.
