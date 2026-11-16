# Host and test the agent locally

With the [resources prepared](01-prepare-resources.md), you can test your existing graph through the local Responses host. In this article, you will add the hosting dependency, load the graph, and verify a representative request before preparing a remote deployment. Keep the current model provider, tools, and prompts.

## Before you start

Complete [Prepare Foundry resources](01-prepare-resources.md). Run commands from the application root, `{local-root}/deep_research`, with the existing graph and local model and search credentials available.

## Add the hosting dependency

The hosting runner requires the `hosting` extra described in the [hosting package installation guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents#install-the-package). Add this dependency before starting the host:

```shell
uv add --python 3.13 --prerelease allow --upgrade-package langchain-azure-ai "langchain-azure-ai[hosting]"
uv pip check
```

The command selects Python 3.13, records the dependency in `pyproject.toml`, updates `uv.lock`, and installs the latest compatible version. The [uv dependency guide](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies) explains these project file updates. The upgrade option requests the latest compatible release when the package is already declared.

## Identify the existing graph entry point

The hosting runner loads an existing graph as described in [Host an existing agent](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents#host-an-existing-agent). In this example, `agent.py` exports the compiled graph as `agent`. The root `langgraph.json` references it as `./agent.py:agent`, telling the hosting runner which graph to load. The hosting runner reuses this graph entry point.

The local host reuses these application files:

| Location | Responsibility |
| --- | --- |
| `agent.py` | Exports the compiled research graph as `agent` |
| `research_agent/` | Existing prompts and tools |
| `langgraph.json` | Points the hosting runner to `./agent.py:agent` |

The hosting package works with the existing graph, tools, and prompts, handling Responses request conversion and streaming for a compiled graph. The runner loads the agent from `langgraph.json`. You can also create and run the host explicitly with `ResponsesHostServer(graph).run()` in Python. The [LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents) describes both approaches.

> **Note:** The Responses hosting path used in this guide does not currently expose the LangGraph Agent Server runs, threads, or state APIs. Reusing the graph does not preserve those client APIs. Adapt agent invocation to the hosted Responses API and preserve the conversation identity for multi-turn clients. The Responses API provides request and response records, not equivalent access to full graph state.

> **Note:** The default host requires a graph state schema with a `messages` field. Check any custom state schema before choosing this path.

### Preserve the public application interface

Map every existing public operation to its input, caller configuration, output or side effect, and usable target access path. Include supporting workflows used before or between conversations. Selecting one exported graph does not expose the others, and retaining `langgraph.json` does not deliver the original server API.

The runner selects and loads a graph. Its default Responses conversion supplies conversation messages; the default Invocations conversion also builds message-shaped graph input. Neither automatically decodes a JSON string into arbitrary graph state or forwards caller-supplied `RunnableConfig.configurable` fields. A `messages` field alone therefore does not establish compatibility. Inspect the installed SDK's request parser, input/configuration builders and output parser against the original interface before relying on the defaults.

Use configuration-only hosting when those mappings preserve the required interface. Otherwise adapt request parsing, operation routing, configuration propagation and output serialization at the hosting boundary, delegating business execution to the existing graphs. Resolve the actual exported runnable or factory rather than assuming a similarly named module attribute. Validate adapted input against the original schema; changing the protocol name or disabling a compatibility check does not implement a missing conversion. Deliver and document any separately required entry points or companion services.

Keep application configuration separate from platform authentication. Preserve per-request parameter semantics and validation; an authenticated user, session key or environment-wide fallback does not automatically replace an application-defined partition/configuration value. Resolve necessary mappings explicitly without inventing a new business or authorization policy.

## Prepare local configuration

Make your application's configuration and credentials available before SDK and graph initialization. Create or update the root `.env` with the values your application reads. The startup command below loads it with uv's `--env-file` option rather than relying on the `env` field in `langgraph.json`.

If you keep the original Deep Research model, configure `ANTHROPIC_API_KEY` in your local `.env`.

The following configuration assumes that you have completed the [Foundry model adaptation](../host-langgraph-on-foundry.md#model-choice). Use these same environment variable names in your model initialization:

```dotenv
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
AZURE_AI_MODEL_DEPLOYMENT_NAME=<deployment-name>
```

The local host test can use the application's existing search credentials. When deploying the agent to Foundry, we recommend storing these credentials in a Foundry project connection, as described in [Configure project connections](03-prepare-deployment.md#configure-project-connections-for-third-party-endpoints).

> **Note:** If your application uses `DefaultAzureCredential`, ensure its authentication dependencies are installed and sign in with `az login` or `azd auth login`. For this example's Foundry model calls, your account can use `Foundry User` on the parent Foundry resource. An administrator can grant it as shown below. This authorizes your local account, not the deployed agent; local success does not verify the agent's runtime access. Other model providers use their own authentication requirements.
>
> PowerShell (Windows):
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry User" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>"
> ```
>
> Bash (macOS and Linux):
>
> ```bash
> az role assignment create \
>     --assignee-object-id "<your-user-object-id>" \
>     --assignee-principal-type User \
>     --role "Foundry User" \
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>"
> ```

## Start and test the host

After `uv add` installs the declared dependencies, start the Responses host. The command uses `--no-sync` to run with the environment you just prepared:

```shell
uv run --no-sync --env-file .env python -m langchain_azure_ai.agents.hosting.run --protocol responses --config langgraph.json
```

## Verify local execution

Installation, sustained host startup and successful application execution are separate gates. Use the actual deployment entry point and effective configuration, not an independently constructed graph. Exercise every preserved operation and the workflow connecting them through the delivered interface, including follow-up requests when continuity matters. Check actual downstream effects, not only HTTP success or an acknowledgement: dropped input can produce a plausible success response without performing the requested operation.

When model, tool or external-service credentials are unavailable, use disposable, external-only substitutes to check a successful local request. Keep the submitted adapter, graph construction, routing, configuration propagation and application execution real; do not stub the delegate or return expected results from the host. Keep substitutes outside production code and remove them before delivery. This verifies the hosting boundary, not cloud integration or real-provider behavior; report unavailable checks and prerequisites separately.

In another terminal, send a representative request to the local `/responses` endpoint. The default port is `8088`; adjust the URL if you configured a different port:

PowerShell (Windows):

```powershell
$body = @{
    input = "Research when to use asyncio.TaskGroup versus asyncio.gather in Python. Use Tavily search and official Python documentation. Return a summary of at most 200 words with source links."
    stream = $false
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "http://localhost:8088/responses" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec 600
$response | ConvertTo-Json -Depth 20
```

Bash (macOS and Linux):

```bash
body=$(jq -n --arg input "Research when to use asyncio.TaskGroup versus asyncio.gather in Python. Use Tavily search and official Python documentation. Return a summary of at most 200 words with source links." \
    '{input: $input, stream: false}')

curl --silent --show-error --max-time 600 \
    --request POST "http://localhost:8088/responses" \
    --header "Content-Type: application/json" \
    --data "$body" | jq .
```

Verify that the response has `status: completed`, no error, and the expected search calls and cited answer. To inspect streamed events and tool execution interactively, use [Foundry Toolkit Agent Inspector](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-agent-inspector#connect-and-debug). For projects already configured for azd, [azd local invocation](https://learn.microsoft.com/azure/foundry/agents/how-to/invoke-hosted-agent#invoke-locally) is another option.

> **Warning:** HTTP success and a successful CLI exit code do not establish agent completion. Inspect the final Responses status and error.

## Next steps

Previous: [Prepare Foundry resources](01-prepare-resources.md) · [Migration overview](../host-langgraph-on-foundry.md) · Next: [Prepare the agent for deployment](03-prepare-deployment.md)
