# Enable persistence and recovery

[Overview](../migrate-langgraph-to-foundry.md)

After verifying the remote deployment, configure durable checkpoints for state that must survive across turns. Unfinished-request recovery is needed when running requests must survive a host failure.

## Before you start

Complete [Deploy and verify the agent](04-deploy-and-verify.md). Keep the deployed agent name, invocation access, and application source available. Choose the checks below that match your state and recovery requirements.

## Persist graph state across turns

If your application needs state across turns, configure a durable checkpointer. In the Deep Research example, this state includes messages, reports, and task progress.

Choose a storage service based on what the agent needs to retain:

| Storage service | What to store | How the application uses it |
| --- | --- | --- |
| [Foundry State Store](https://learn.microsoft.com/azure/foundry/agents/concepts/agent-state-store) | LangGraph checkpoints; in this example, messages, TODOs, and virtual files | The graph uses `FoundryCheckpointSaver` to save and restore checkpoints. |
| [Azure Cosmos DB](https://learn.microsoft.com/azure/cosmos-db/gen-ai/integrations#langgraph) | LangGraph checkpoints or structured application data | The graph uses a Cosmos DB checkpointer for checkpoints; application code reads and writes other records. |
| [Azure Blob Storage](https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction) | Reports, downloaded sources, and other file artifacts | Application tools write and read blobs, while graph state can retain references to them. |

These services can be used together. This guide uses Foundry State Store through [`FoundryCheckpointSaver`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) for checkpoints. The [default Deep Agents backend](https://docs.langchain.com/oss/python/deepagents/backends) stores virtual files in graph state, so checkpoints preserve the report and TODOs across turns.

Foundry State Store uses the hosted agent's identity model and does not require provisioning a separate storage account. If you choose application-owned Cosmos DB or Blob Storage instead, configure the agent identity's access to that service using its data-plane permissions; the Foundry connection role in [Deploy remotely](04-deploy-and-verify.md#deploy-remotely) does not grant access to those stores.

> **Note:** The hosted session filesystem has a session-scoped lifecycle. Foundry preserves `$HOME` and uploaded files across turns, idle periods, and session stops. Reports that must remain available after session deletion or expiration need external storage. See [Manage hosted agent sessions](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions).

Configure the checkpointer in Python when you create the graph. In this example, the application builds its graph directly in the root `agent.py`. Add the `FoundryCheckpointSaver` import and pass the checkpointer to the existing `create_deep_agent(...)` call:

```python
from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver


agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    checkpointer=FoundryCheckpointSaver(user_isolation=True),
)
```

The existing root `langgraph.json` loads `./agent.py:agent`. The Responses host uses async graph execution, as required by `FoundryCheckpointSaver`.

> **Tip:** To continue using [LangGraph Studio](https://docs.langchain.com/langsmith/local-dev-testing) for local debugging, provide a separate graph entry point without a custom checkpointer. `langgraph dev` manages persistence itself and rejects graphs with an application-configured checkpointer. Keep `FoundryCheckpointSaver` on the Foundry entry point. Studio shows local graph state, not deployed Foundry checkpoints.

Redeploy the updated agent:

```powershell
azd deploy deep-research-foundry
```

### Verify persistence and user isolation

Verify that a later turn restores and updates the required application state. For this example, create a report and revise it in a later turn. For hosted tests, the client needs to retain the conversation identity and authenticated user across turns. Test continuation in the same session, then in a different session with the same conversation. The [session management guide](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions) explains sessions and conversations.

> **Tip:** Measure latency with representative conversation histories and state sizes, including continuation in a new session. Separate model and tool time from checkpoint and response-history I/O. A successful first turn does not establish acceptable multi-turn performance.

> **Warning:** User isolation must remain enabled for user-specific state. Test continuation with the same authenticated user and verify that another user cannot access that state.

## Recover unfinished requests

If unfinished requests must survive host failures, configure durable checkpoints and the Responses host's recovery settings. These settings allow execution to resume from persisted graph progress, including checkpoints saved partway through an interrupted request.

The graph still needs the durable checkpointer. The following configuration applies to the hosting runner used in this guide. To enable recovery for the deployed host, update the agent's `codeConfiguration` in `azure.yaml`:

```yaml
codeConfiguration:
  runtime: python_3_13
  entryPoint: >-
    -m langchain_azure_ai.agents.hosting.run
    --protocol responses --config langgraph.json
    --option resilient_background=true
  dependencyResolution: remote_build
```

Redeploy the updated entry point with `azd deploy deep-research-foundry`. Test recovery on the deployed agent in a test environment.

> **Note:** Foundry State Store's [user isolation](https://learn.microsoft.com/azure/foundry/agents/concepts/agent-state-store#caller-identity) is not supported in local runs. Keep `user_isolation=True` for the hosted agent; a local host test does not verify this configuration's isolation or recovery behavior.

The client must set `background` and `store` to `true` to make the request recoverable. The following PowerShell example uses the [hosted Responses endpoint](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions?pivots=rest#invoke-an-agent-and-let-the-platform-create-the-session):

> **Note:** Sign in with `az login` using an account with the invocation access described in [Verify the deployed agent](04-deploy-and-verify.md#verify-the-deployed-agent).

```powershell
$projectEndpoint = "https://<resource>.services.ai.azure.com/api/projects/<project>"
$responsesUrl = "$projectEndpoint/agents/deep-research-foundry/endpoint/protocols/openai/responses"
$token = az account get-access-token --resource https://ai.azure.com --query accessToken --output tsv
$headers = @{ Authorization = "Bearer $token" }
$body = @{
    input = "Research the difference between HTTP 429 and 503. Save a report with citations to /reports/http-errors.md."
    background = $true
    store = $true
    stream = $false
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "${responsesUrl}?api-version=v1" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body
$response.id
```

Retain the returned response ID. A background request can return before completion. Retrieve the same response until its status is terminal:

```powershell
$current = Invoke-RestMethod `
    -Uri "$responsesUrl/$($response.id)?api-version=v1" `
    -Headers $headers
$current | ConvertTo-Json -Depth 20
```

After a connection failure, retrieve the existing response instead of submitting the research request again. Refresh the access token if it expires.

### Verify unfinished-request recovery

Before relying on recovery, interrupt a running hosted request in a test environment and let the host restart. Retrieve the original response ID and verify completion, saved state, and the absence of duplicate external effects. Successful continuation after a completed turn does not verify crash recovery.

> **Warning:** A recovered operation may execute unfinished work again. Your external side effects must be idempotent.

The [recovery guide](https://learn.microsoft.com/azure/foundry/agents/how-to/recover-long-running-work?tabs=responses) describes the response lifecycle and request requirements.

## Next steps

Previous: [Deploy and verify the agent](04-deploy-and-verify.md) · Next: [Inspect logs and traces](06-diagnostics.md)
