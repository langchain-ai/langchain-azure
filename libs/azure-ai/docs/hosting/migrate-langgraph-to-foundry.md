# Host a LangGraph application on Microsoft Foundry

If you already have a LangGraph, LangChain, or Deep Agents application, you can host it as a hosted agent on Microsoft Foundry. This guide covers hosting setup, deployment, diagnostics, and evaluation, with code examples, tips, and notes.

This guide uses a Deep Research application adapted from the [Deep Research sample](https://github.com/langchain-ai/deepagents/tree/4ff926659cfb1281b01d6821052135cef151dde3/examples/deep_research) in the [Deep Agents](https://github.com/langchain-ai/deepagents) repository. In this application, a user asks the agent to research a topic, reviews the report, and requests a revision in a later turn.

The examples use Python, uv, the hosted Responses protocol, and source-code deployment. Adapt the commands and configuration to your application's dependencies, authentication, and deployment requirements.

> **Note:** This guide covers application hosting and runtime configuration. Moving existing LangGraph threads and checkpoints to another storage backend requires a separate data migration.

## Before you start

The following prerequisites apply before you prepare and deploy your application:

- This guide uses Python 3.13 for local testing and Foundry source-code deployment.
- You need access to the application source code and dependency definitions.
- Deployment requires [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), [Azure Developer CLI (`azd`)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd), and the Foundry agents extension described in the [deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent).
- The application's search provider requires credentials or connection information.

> **Note:** Permission notes below describe access for your account, separate from the agent's runtime access. Skip role assignments if existing or inherited access is sufficient. Role assignment commands require `Microsoft.Authorization/roleAssignments/write` at the target scope; ask an administrator to run them if needed. In user-assignment examples, replace `<your-user-object-id>` with the Microsoft Entra object ID of the account you use for that operation.

> **Tip:** Use Azure Developer CLI (`azd`) or [Microsoft Foundry Toolkit for VS Code](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-agents-workflow-pro-code) for local testing and deployment. If you use a coding agent, add the [Microsoft Foundry Skill](https://learn.microsoft.com/azure/foundry/how-to/develop/use-microsoft-foundry-skill) to assist with setup and troubleshooting.

This guide assumes that the Deep Research example from the Deep Agents repository is available at `{local-root}\deep_research`. All relative paths below start from that directory. The example includes `agent.py`, `research_agent/`, `langgraph.json`, and dependency files. The guide adds Foundry hosting and deployment files alongside them.

Your local `.env` must remain outside source control and source uploads. Step 3.1 describes the upload exclusions.

## 1. Prepare Foundry resources

Prepare a Foundry project and the model or external-service access your application requires. The following example uses a Foundry model deployment and Tavily search. You can reuse existing resources where possible.

**Foundry project.** The hosted agent needs a project in your Azure subscription. For setup instructions, see [Create a project for Microsoft Foundry](https://learn.microsoft.com/azure/foundry/how-to/create-projects). The deployment configuration uses the project endpoint, subscription ID, and project resource ID. The connection-access role assignment in step 3.4 also needs the parent Foundry resource ID.

**Model access.** The agent needs a model that supports the application's API and tool calls. Using a Foundry model is not a prerequisite for hosting your agent on Foundry; you can keep your existing model provider. This example uses a Foundry model deployment and its deployment name in the agent configuration. If you want to set up and use a Foundry model, see [Deploy a Foundry model](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/deploy-foundry-models) and [Call a Foundry model](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/generate-responses).

**Search tool.** This research application needs a web search tool to gather sources for its reports. You can keep the existing Tavily integration or replace it with search tools exposed through Foundry Toolbox. The rest of this guide uses Tavily's search endpoint. The [Tavily Python SDK quickstart](https://docs.tavily.com/sdk/python/quick-start) and [Build a toolbox and use it with a hosted agent](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-toolbox-agent) describe these two options.

> **Tip:** You can keep the existing search provider for the first deployment and replace it after agent hosting works.

**Runtime access.** The hosted agent uses its own identity to access models and connected resources. Model inference through its own project endpoint is available by default; configure additional access where the application needs it, such as reading connection credentials in step 3.4. See the [hosted agent permissions reference](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions).

> **Note:** If you create Foundry resources or deploy models instead of reusing existing resources, your account needs the corresponding resource-management permissions. For example, an administrator can grant `Contributor` on the existing resource group where you create them. This role does not grant role-assignment permissions. See [Azure resource setup](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions#azure-resource-setup).
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Contributor" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>"
> ```

Before continuing, have the project endpoint, subscription ID, project ARM ID, parent Foundry resource ARM ID, and your application's model and search configuration ready.

## 2. Host and test the agent locally

First verify that your existing graph works through the Responses API before adding remote deployment configuration. Keep the current model provider, tools, and prompts.

### Add the hosting dependency

The hosting runner requires the `hosting` extra described in the [hosting package installation guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents#install-the-package). Add this dependency before starting the host:

```powershell
uv add --python 3.13 --prerelease allow --upgrade-package langchain-azure-ai "langchain-azure-ai[hosting]"
uv pip check
```

The command selects Python 3.13, records the dependency in `pyproject.toml`, updates `uv.lock`, and installs the latest compatible version. The [uv dependency guide](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies) explains these project file updates. The upgrade option requests the latest compatible release when the package is already declared.

### Identify the existing graph entry point

The hosting runner loads an existing graph as described in [Host an existing agent](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents#host-an-existing-agent). In this example, `agent.py` exports the compiled graph as `agent`. The root `langgraph.json` references it as `./agent.py:agent`, telling the hosting runner which graph to load. No new application entry point is required.

The local host reuses these application files:

| Location | Responsibility |
| --- | --- |
| `agent.py` | Exports the compiled research graph as `agent` |
| `research_agent/` | Existing prompts and tools |
| `langgraph.json` | Points the hosting runner to `./agent.py:agent` |

The hosting package works with the existing graph, tools, and prompts, handling Responses request conversion and streaming for a compiled graph. The runner loads the agent from `langgraph.json`. You can also create and run the host explicitly with `ResponsesHostServer(graph).run()` in Python. The [LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents) describes both approaches.

> **Note:** The Responses hosting path used in this guide does not currently expose the LangGraph Agent Server runs, threads, or state APIs. Reusing the graph does not preserve those client APIs. Adapt agent invocation to the hosted Responses API and preserve the conversation identity for multi-turn clients. The Responses API provides request and response records, not equivalent access to full graph state.

> **Note:** The default host requires a graph state schema with a `messages` field. Check any custom state schema before choosing this path.

### Prepare local configuration

Make your application's configuration and credentials available before SDK and graph initialization. Create or update the root `.env` with the values your application reads. The startup command below loads it with uv's `--env-file` option rather than relying on the `env` field in `langgraph.json`.

For the Foundry model configuration used in this example, `.env` includes:

```dotenv
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
AZURE_AI_MODEL_DEPLOYMENT_NAME=<deployment-name>
```

The local host test can use the application's existing search credentials. When deploying the agent to Foundry, we recommend storing these credentials in a Foundry project connection, as described in [step 3.2](#32-configure-project-connections-for-third-party-endpoints).

> **Note:** If your application uses `DefaultAzureCredential`, ensure its authentication dependencies are installed and sign in with `az login` or `azd auth login`. For this example's Foundry model calls, your account can use `Foundry User` on the parent Foundry resource. An administrator can grant it as shown below. This authorizes your local account, not the deployed agent; local success does not verify the agent's runtime access. Other model providers use their own authentication requirements.
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry User" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>"
> ```

### Start and test the host

After `uv add` installs the declared dependencies, start the Responses host. The command uses `--no-sync` to run with the environment you just prepared:

```powershell
uv run --no-sync --env-file .env python -m langchain_azure_ai.agents.hosting.run --protocol responses --config langgraph.json
```

In another terminal, send a representative request to the local `/responses` endpoint. The default port is `8088`; adjust the URL if you configured a different port:

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

Verify that the response has `status: completed`, no error, and the expected search calls and cited answer. To inspect streamed events and tool execution interactively, use [Foundry Toolkit Agent Inspector](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-agent-inspector#connect-and-debug). For projects already configured for azd, [azd local invocation](https://learn.microsoft.com/azure/foundry/agents/how-to/invoke-hosted-agent#invoke-locally) is another option.

> **Warning:** HTTP success and a successful CLI exit code do not establish agent completion. Inspect the final Responses status and error.

## 3. Deploy the agent to Foundry

Once you have verified that the agent runs correctly with the local Responses host, proceed with deploying it to Foundry.

> **Note:** Your account needs `Foundry Project Manager` on the Foundry project to create project connections and deploy the source-code agent. An administrator can grant this access with the following command. This assignment is for your account, not the agent identity. See the [source-code deployment permissions](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code#required-permissions).
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry Project Manager" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>/projects/<project-name>"
> ```

### 3.1 Prepare deployment files

#### Configure azure.yaml

The `codeConfiguration.entryPoint` setting in `azure.yaml` specifies the hosting runner command, as shown in [Host an existing agent](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents#host-an-existing-agent).

`langgraph.json` tells the runner how to load the agent, both locally and in Foundry. `azure.yaml` tells azd how to deploy and start the hosted agent. The deployment package must include `langgraph.json` and the application files it references. See [Package the zip manually](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code#package-the-zip-manually) for the package layout and `requirements.txt`.

Initialize a source-code deployment from `{local-root}\deep_research`:

```powershell
azd auth login
azd ai agent init --protocol responses --deploy-mode code --runtime python_3_13 --entry-point '-m langchain_azure_ai.agents.hosting.run --protocol responses --config langgraph.json'
```

During initialization, select the existing project from step 1 and name the agent `deep-research-foundry`. The initialization flow creates `azure.yaml` and an azd environment. The agent service's `project` field should be `.` so the upload starts from the application root directory.

The following agent service configuration starts the hosting runner with the graph configuration tested in step 2:

```yaml
services:
  deep-research-foundry:
    host: azure.ai.agent
    kind: hosted
    project: .
    language: python
    protocols:
      - protocol: responses
        version: 2.0.0
    codeConfiguration:
      runtime: python_3_13
      entryPoint: >-
        -m langchain_azure_ai.agents.hosting.run
        --protocol responses --config langgraph.json
      dependencyResolution: remote_build
```

The Python runtime starts the module specified by `codeConfiguration.entryPoint`. The [azure.yaml authoring guide](https://learn.microsoft.com/azure/foundry/agents/how-to/author-azure-yaml) and [source-code deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code) describe the other deployment settings.

#### Prepare runtime dependencies

The remote build installs dependencies from the root `requirements.txt`, not your local `pyproject.toml` or `uv.lock`. Include the application's runtime dependencies and choose compatible versions. For this guide's Foundry-model adaptation, these include:

```text
deepagents>=0.6.12
langchain-openai>=1.3.4,<2.0
azure-identity>=1.15,<2.0
azure-ai-projects
langchain-azure-ai[hosting]
tavily-python>=0.7.26
httpx>=0.28.1
markdownify>=1.2.2
```

Check that these requirements install in a fresh Python 3.13 environment outside the project:

```powershell
$checkEnv = Join-Path $env:TEMP ("foundry-deps-" + [guid]::NewGuid().ToString("N"))
uv venv --python 3.13 $checkEnv
uv pip install --python "$checkEnv\Scripts\python.exe" --prerelease allow -r requirements.txt
uv pip check --python "$checkEnv\Scripts\python.exe"
```

Both installation and the dependency check must succeed before deployment. This verifies dependency compatibility, not application startup.

#### Configure upload exclusions

Configure a root `.agentignore` to keep local credentials, development files, and evaluation data out of source uploads. The following exclusions apply to this example's `requirements.txt`-based build. Adapt development-file exclusions to your build process; do not exclude files required at build time or runtime:

```text
.env
.env.*
.azure/
.venv/
.git
eval/
__pycache__/
*.pyc
*.log
uv.lock
pyproject.toml
research_agent.ipynb
```

Your `.gitignore` should also exclude local credentials, azd environment files, and generated results. In this example, the remote build uses `requirements.txt`; the upload excludes the development `pyproject.toml` and lockfile.

Review the upload contents before deployment. The upload must exclude `.env`, local caches, evaluation output, and development-only dependencies.

### 3.2 Configure project connections for third-party endpoints

Configure secure credential access for any third-party endpoints your application uses. This example stores the search API key in a Foundry project connection. Follow [Add a new connection to your project](https://learn.microsoft.com/azure/foundry/how-to/connections-add) to create the appropriate connection.

This example uses a `CustomKeys` connection named `tavily-search` with target `https://api.tavily.com`. You can obtain a key from the [Tavily Python SDK quickstart](https://docs.tavily.com/sdk/python/quick-start) and store it in the connection's secret field named `api_key`.

Add the project SDK to the dependencies before updating the Tavily client initialization:

```powershell
uv add --prerelease allow azure-ai-projects
uv pip check
```

In this example, the application reads the connection credentials before initializing the Tavily client. The project SDK supports this through [`connections.get(..., include_credentials=True)`](https://learn.microsoft.com/python/api/azure-ai-projects/azure.ai.projects.operations.connectionsoperations#azure-ai-projects-operations-connectionsoperations-get). Replace the existing `tavily_client = TavilyClient()` initialization in `research_agent/tools.py` with the following code:

```python
import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CustomCredential
from azure.identity import DefaultAzureCredential
from tavily import TavilyClient

with (
    DefaultAzureCredential() as credential,
    AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        credential=credential,
    ) as project,
):
    connection = project.connections.get(
        os.environ["TAVILY_CONNECTION_NAME"], include_credentials=True
    )
    credentials = connection.credentials
    if not isinstance(credentials, CustomCredential):
        raise ValueError("The Tavily connection must use CustomKeys credentials.")
    api_key = credentials.credential_keys.get("api_key")
    if not api_key:
        raise ValueError("The Tavily connection needs an api_key secret field.")
    tavily_client = TavilyClient(api_key=api_key)
```

The hosted agent identity also needs permission to read these connection credentials. Assign that access after deployment creates the identity, as shown in [step 3.4](#34-deploy-remotely).

> **Note:** Local tests that read the project connection use your developer credentials. Your account also needs connection-credential access on the parent Foundry resource; the `Foundry User` assignment in the local-testing note in step 2 covers this example. The agent's runtime assignment does not grant access to your account.

> **Tip:** In this example, the Tavily client is initialized when `research_agent.tools` is imported, so the connection lookup must complete before graph initialization. Other applications can retrieve credentials when initializing or invoking the dependent tool. The hosting runner does not resolve a connection name automatically.

### 3.3 Set environment variables

The hosted runtime uses deployment configuration, not your local `.env`. Pass the model deployment name and connection name to the agent; the tool code from step 3.2 retrieves the secret at runtime. Foundry supplies the project endpoint automatically.

This example uses the following values:

| Variable | Value |
| --- | --- |
| `FOUNDRY_PROJECT_ENDPOINT` | Foundry injects the project endpoint into the hosted runtime. Your local host reads it from `.env`. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Your selected model deployment name |
| `TAVILY_CONNECTION_NAME` | Your project connection name for Tavily credentials |

Set the model deployment name and connection name in the selected azd environment:

```powershell
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME "<deployment-name>"
azd env set TAVILY_CONNECTION_NAME tavily-search
```

Add these values to the agent service's `env` mapping in `azure.yaml`:

```yaml
env:
  AZURE_AI_MODEL_DEPLOYMENT_NAME: ${AZURE_AI_MODEL_DEPLOYMENT_NAME}
  TAVILY_CONNECTION_NAME: ${TAVILY_CONNECTION_NAME}
```

Add the connection name to the local `.env`:

```dotenv
TAVILY_CONNECTION_NAME=tavily-search
```

Restart the local host with the command from step 2 and verify that the application can search using the project connection before deployment. After this check succeeds, you can remove `TAVILY_API_KEY` from your local `.env`.

### 3.4 Deploy remotely

Deploy the configured application to create an agent version and its runtime identity. Run `azd provision` first if new resources or infrastructure changes are required. For an existing project with no infrastructure changes:

```powershell
azd deploy deep-research-foundry
$agentInfo = azd ai agent show deep-research-foundry --output json | ConvertFrom-Json
$agentInfo | ConvertTo-Json -Depth 10
```

#### Grant runtime connection access

Before invoking the agent, grant its instance identity `Foundry User` on the parent Foundry resource to read the connection credentials from step 3.2. Skip this assignment if the identity already has sufficient access. See the [hosted agent permissions reference](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions).

Use the identity returned above and the parent resource's ARM ID, not the project ID:

```powershell
$foundryResourceId = "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>"
$agentPrincipalId = $agentInfo.instance_identity.principal_id
az role assignment create `
    --assignee-object-id $agentPrincipalId `
    --assignee-principal-type ServicePrincipal `
    --role "Foundry User" `
    --scope $foundryResourceId
```

Confirm that the deployment reports the expected agent version and instance identity, and that the required role assignment succeeds. Then verify runtime access with a hosted request in step 3.5.

### 3.5 Verify the deployed agent

Verify that the deployed agent identity can access its model and connected resources. If local calls work and hosted calls return `PermissionDenied`, check the identity and endpoint scope.

> **Note:** Invoking the hosted agent requires permission for your account to interact with agents in the project. `Foundry Agent Consumer` provides this access; skip this assignment if your existing project role already includes it. An administrator can grant it with:
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry Agent Consumer" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>/projects/<project-name>"
> ```

Test the deployed agent with a representative request using the [Foundry Toolkit Hosted Agent Playground](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-agents-workflow-pro-code#inspect-and-update-the-deployed-agent), [azd remote invocation](https://learn.microsoft.com/azure/foundry/agents/how-to/invoke-hosted-agent#invoke-the-deployed-agent), or the [Foundry portal playground](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-hosted-agent#step-4-deploy-to-foundry-agent-service). For the Deep Research example, the following CLI check exercises search, report writing, and readback:

```powershell
$testQuery = "Research when to use asyncio.TaskGroup versus asyncio.gather in Python. Use Tavily search and official Python documentation. Save a report of at most 250 words with source links to /reports/asyncio.md, read it back, and include the verified report in your response."
azd ai agent invoke deep-research-foundry --new-session --new-conversation --timeout 600 $testQuery
```

Check the final response status, expected tool calls, and application outputs. For the Deep Research example, verify the actual search calls and report contents. If the application depends on sandbox files, the client needs to retain the hosted session as described in [Manage hosted agent sessions](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions).

## 4. Enable persistence and recovery

After verifying the remote deployment, configure durable checkpoints for state that must survive across turns. Unfinished-request recovery is needed when running requests must survive a host failure.

### 4.1 Persist graph state across turns

If your application needs state across turns, configure a durable checkpointer. In the Deep Research example, this state includes messages, reports, and task progress.

Choose a storage service based on what the agent needs to retain:

| Storage service | What to store | How the application uses it |
| --- | --- | --- |
| [Foundry State Store](https://learn.microsoft.com/azure/foundry/agents/concepts/agent-state-store) | LangGraph checkpoints; in this example, messages, TODOs, and virtual files | The graph uses `FoundryCheckpointSaver` to save and restore checkpoints. |
| [Azure Cosmos DB](https://learn.microsoft.com/azure/cosmos-db/gen-ai/integrations#langgraph) | LangGraph checkpoints or structured application data | The graph uses a Cosmos DB checkpointer for checkpoints; application code reads and writes other records. |
| [Azure Blob Storage](https://learn.microsoft.com/azure/storage/blobs/storage-blobs-introduction) | Reports, downloaded sources, and other file artifacts | Application tools write and read blobs, while graph state can retain references to them. |

These services can be used together. This guide uses Foundry State Store through [`FoundryCheckpointSaver`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) for checkpoints. The [default Deep Agents backend](https://docs.langchain.com/oss/python/deepagents/backends) stores virtual files in graph state, so checkpoints preserve the report and TODOs across turns.

Foundry State Store uses the hosted agent's identity model and does not require provisioning a separate storage account. If you choose application-owned Cosmos DB or Blob Storage instead, configure the agent identity's access to that service using its data-plane permissions; the Foundry connection role in step 3.4 does not grant access to those stores.

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

Verify that a later turn restores and updates the required application state. For this example, create a report and revise it in a later turn. For hosted tests, the client needs to retain the conversation identity and authenticated user across turns. Test continuation in the same session, then in a different session with the same conversation. The [session management guide](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions) explains sessions and conversations.

> **Tip:** Measure latency with representative conversation histories and state sizes, including continuation in a new session. Separate model and tool time from checkpoint and response-history I/O. A successful first turn does not establish acceptable multi-turn performance.

> **Warning:** User isolation must remain enabled for user-specific state. Test continuation with the same authenticated user and verify that another user cannot access that state.

### 4.2 Recover unfinished requests

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

> **Note:** Sign in with `az login` using an account with the invocation access described in step 3.5.

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

Before relying on recovery, interrupt a running hosted request in a test environment and let the host restart. Retrieve the original response ID and verify completion, saved state, and the absence of duplicate external effects. Successful continuation after a completed turn does not verify crash recovery.

> **Warning:** A recovered operation may execute unfinished work again. Your external side effects must be idempotent.

The [recovery guide](https://learn.microsoft.com/azure/foundry/agents/how-to/recover-long-running-work?tabs=responses) describes the response lifecycle and request requirements.

## 5. Observe and evaluate the agent

Logs and traces help you understand how the agent runs and diagnose failures or slow requests. Evaluation helps you measure answer quality and detect regressions as you update the agent.

### 5.1 Inspect logs and traces

You can inspect logs, traces, and application state through the following sources and tools:

| Source | Check | How to access |
| --- | --- | --- |
| Responses record | Final status, error, response ID, input, and output | Use `azd ai agent invoke deep-research-foundry --output raw "<query>"` to inspect the invocation response, or test in the Foundry portal playground or Foundry Toolkit **Hosted Agent Playground**. To retrieve a stored response by ID, use the Responses API or SDK. |
| Console logs | Startup, imports, authentication, SDK exceptions, and container output | Use `azd ai agent monitor deep-research-foundry --session-id "<session-id>" --tail 300`; use `--type system` for container lifecycle events. |
| Application Insights traces | Model calls, search calls, subagent execution, and latency | Follow [Enable and verify traces](#enable-and-verify-traces) below to configure tracing. Open **Agents** > **Traces** in the Foundry portal, use [Foundry Toolkit hosted-agent traces](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-tracing#view-hosted-agent-traces), or query the connected Application Insights resource in Azure Monitor. `azd ai agent monitor` does not query these spans. |
| Checkpoints | Persisted graph state, checkpoint history, and execution progress | Inspect checkpoint records in the configured storage backend, such as Foundry State Store or Azure Cosmos DB, with the required data-access permissions. For Cosmos DB, use Data Explorer in the Azure portal. |

> **Note:** The Foundry portal and Inspector do not currently provide a Studio-style graph-state viewer. The sandbox Files view does not show virtual files stored in graph state.

To inspect graph state, use the configured checkpointer's read APIs or its backing storage with the required permissions. Raw records may need decoding. [Get graph state](https://docs.langchain.com/oss/python/langgraph/checkpointers#get-state) describes application-side access, not a Foundry remote state endpoint.

> **Tip:** For simpler diagnostics, add application code to record selected state fields in storage or traces. Exclude credentials and sensitive content.

> **Note:** `azd ai agent state-stores items list` and `azd ai agent state-stores items show <key>` can inspect items only in non-user-isolated stores.

#### Inspect a request and its console logs

Use a representative application request; the Deep Research query from step 3.5 is one example. Retain the agent name and version, request time, response ID, conversation ID, and hosted session ID from the invocation. Check the final Responses status and error before interpreting the logs. For a background request, retrieve the response until it reaches a terminal status, as described in step 4.2.

Use the hosted session ID from the invocation output to inspect its console logs:

```powershell
azd ai agent monitor deep-research-foundry --session-id "<session-id>" --tail 300
```

For startup failures or container restarts, also inspect system events:

```powershell
azd ai agent monitor deep-research-foundry --session-id "<session-id>" --type system --tail 300
```

A hosted session can contain several requests. Use request times and response IDs, where logged, to distinguish turns within the session. The [hosted agent logs guide](https://learn.microsoft.com/azure/foundry/agents/how-to/monitor-hosted-agent-logs) describes log types and streaming.

#### Enable and verify traces

With LangChain instrumentation enabled, the hosting package supports exporting LangChain execution traces as GenAI-compatible OpenTelemetry (OTel) spans to Azure Monitor Application Insights. The hosted server adds project and agent identity attributes to associate these traces with the deployed agent.

Use the following sequence for the hosted application:

1. If LangChain tracing is not already configured, use the hosting libraries' built-in Microsoft OpenTelemetry integration. No additional instrumentation package is required. See the [hosted-agent tracing quickstart](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-tracing-hosted-agent).

   > **Warning:** Check for existing LangChain instrumentation before enabling another tracer. Avoid tracing the same graph with both the host's Microsoft OpenTelemetry integration and `AzureAIOpenTelemetryTracer`, which can produce duplicate spans. Do not initialize a second tracer provider in an already configured host.

2. Connect an Application Insights resource to the Foundry project. Reuse an existing resource or create one if needed. Follow [Set up tracing in Microsoft Foundry](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-agent-setup#connect-application-insights-to-your-foundry-project) for resource creation and connection. If the project already has a connection, confirm that it points to the intended resource.

   If your organization requires Microsoft Entra-authenticated ingestion, follow [Configure Microsoft Entra authentication for trace ingestion (preview)](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-ingestion-entra-authentication) to configure the connection. The linked portal workflow grants `Monitoring Metrics Publisher` to the project managed identity for server-side traces. Also grant this role to the agent identity for traces emitted by its code:

   ```powershell
   $appInsightsResourceId = "<application-insights-resource-arm-id>"
   $agentInfo = azd ai agent show deep-research-foundry --output json | ConvertFrom-Json
   $agentPrincipalId = $agentInfo.instance_identity.principal_id
   az role assignment create `
       --assignee-object-id $agentPrincipalId `
       --assignee-principal-type ServicePrincipal `
       --role "Monitoring Metrics Publisher" `
       --scope $appInsightsResourceId
   ```

   This ingestion assignment is not required for key-based ingestion.

   > **Note:** Creating a new Application Insights resource or Log Analytics workspace needs resource-management permissions, such as the resource-group `Contributor` assignment shown in step 1. Connecting it to the project uses the project permissions described in step 3. The ingestion role assignment above requires role-assignment permission on Application Insights; ask an administrator to run it if needed.

3. If runtime configuration has changed, redeploy with `azd deploy deep-research-foundry` to apply it. The hosted runtime supplies `APPLICATIONINSIGHTS_CONNECTION_STRING` for the connected resource. If the deployed application already exports traces to the intended destination, no redeployment is needed for this check.
4. Open **Agents** > **Traces** in the Foundry portal and select an existing trace for the deployed agent version and request time. If no suitable trace is available, invoke the agent with a representative request, such as the Deep Research query from step 3.5, and allow time for telemetry ingestion. Confirm that the trace includes the model and tool operations expected for the request, not just an `invoke_agent` entry span. For the Deep Research example, verify the actual search-tool execution. Inspect subagent spans when the request delegates work. Retain the trace ID and any available response, conversation, or session attributes with the invocation record.

You can also find these traces in the Azure portal: open the Log Analytics workspace linked to the Application Insights resource, select **Logs**, and query the telemetry using the trace ID and request time.

> **Note:** To query traces, your account needs `Log Analytics Reader` on the connected Application Insights resource. An administrator can grant it with the command below. For workspace-scoped queries, repeat the assignment using the linked workspace's ARM ID as the scope. Protected tables can require additional access, as described in the [tracing prerequisites](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-agent-setup#prerequisites). These read permissions are separate from the agent's ingestion permissions.
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Log Analytics Reader" `
>     --scope "<application-insights-resource-arm-id>"
> ```

> **Tip:** A successful outer span or a completed function-call item does not prove that the operation succeeded. Inspect the final response, child-span errors, exception records, and actual tool results. Tools that return failures as ordinary text can still produce successful spans.

> **Warning:** The hosting libraries use `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` to control message-content recording. Keep it `false` in production; enable it with `true` only for controlled tests after reviewing the data being captured. Configure this value in the service's `env` mapping in `azure.yaml`, or in `.env` for local execution, before starting the host. This setting does not sanitize custom logs or metadata; keep credentials out of both. See the [hosted-agent tracing quickstart](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-tracing-hosted-agent#troubleshooting) for details.

### 5.2 Evaluate the agent

Logs and traces help verify that the agent executes as expected, but successful execution does not establish answer quality. Evaluation is therefore important to check whether the migrated agent still produces useful, accurate answers that meet user needs.

Reuse your existing evaluation dataset, evaluators, and pass criteria. Change how the evaluation calls the agent, not what it measures. You can continue to use your local evaluation tools to test the agent deployed in Foundry. You can also run managed evaluations in Foundry and set up continuous evaluation of live agent traffic.

> **Note:** Your existing evaluation runner needs the invocation access from step 3.5 and access to its judge model, if used.

> **Note:** For Foundry portal or SDK evaluations, your account needs `Foundry User` on the project. Foundry model inference also needs account-scoped access; the assignment below covers both for this example. For CI, assign access to the workload identity instead. See [evaluation permissions](https://learn.microsoft.com/azure/foundry/observability/how-to/evaluation-permissions) for additional workflows.
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry User" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>"
> ```

#### Connect your existing evaluation tools to the deployed agent

1. Retain a baseline run from the original application, including its dataset version, evaluator configuration, and per-case results.
2. Replace the evaluation runner's local graph invocation with a call to the deployed Foundry agent. Follow the [hosted Responses SDK and REST invocation examples](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions?pivots=python#invoke-an-agent-and-let-the-platform-create-the-session), targeting the agent version verified in [step 3.5](#35-verify-the-deployed-agent). Wait for each response to complete before scoring it.
3. Map the hosted response back to the fields your existing evaluators expect. Preserve reference answers and case metadata. If an evaluator previously read graph state or a saved report, provide equivalent evidence from the hosted execution; response text alone might not be sufficient.
4. Run the same cases and scoring logic against the hosted agent. Use a separate conversation for each independent case. For multi-turn cases, preserve the authenticated user and conversation as described in [step 4.1](#41-persist-graph-state-across-turns).

Custom evaluators can remain in your existing runner. Evaluators that depend on local callbacks, tool outputs, or traces need an adapter for hosted execution. Use the model and tool traces from step 5.1 where applicable, and report missing evidence rather than treating it as a pass.

> **Tip:** Optionally, upload saved queries, responses, and reference data to [evaluate the saved responses in Foundry](https://learn.microsoft.com/azure/foundry/observability/how-to/cloud-evaluation-datasets). This creates evaluation runs for centralized tracking and [comparison in the portal](https://learn.microsoft.com/azure/foundry/how-to/evaluate-results#compare-the-evaluation-results). It runs evaluators on saved responses without invoking the agent again; it does not import scores already calculated by your local tools.

#### Compare results before and after migration

Run the same test cases before and after migration, keeping the agent model, judge model (if used), search provider, and evaluation settings unchanged. Compare per-case scores, failures, and latency against your existing pass criteria. If live search returns different evidence, review it before attributing a score change to migration.

Save both runs with the dataset version, evaluation settings, agent version, and available trace IDs. Include failed or incomplete cases, and keep credentials and sensitive content out of the results.

> **Tip:** As a next step, consider [running managed evaluations in Foundry](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-evaluate-hosted-agent) and setting up [continuous evaluation of live agent traffic](https://learn.microsoft.com/azure/foundry/observability/how-to/how-to-monitor-agents-dashboard#set-up-continuous-evaluation).

## Migration acceptance checklist

Use this checklist to verify the migration behaviors covered by this guide. Apply the checks that match your application's requirements; the **Further reading** links cover optional next steps.

> **Note:** Confirm your account's access using the permission notes in the relevant steps. These permissions are separate from the agent's runtime access checked below.

| Check | Action and verification | Further reading |
| --- | --- | --- |
| Single-turn execution | Invoke the deployed agent with a representative request. Verify the final status, required tool calls, and application outputs ([step 3.5](#35-verify-the-deployed-agent)). | [Hosted agent invocation](https://learn.microsoft.com/azure/foundry/agents/how-to/invoke-hosted-agent) |
| Multi-turn behavior | When required, retain the authenticated user and conversation across turns. Verify that follow-up requests use prior context and update application state as intended ([step 4.1](#41-persist-graph-state-across-turns)). | [Session lifecycle and management](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions) |
| Durable continuation | When required, configure a durable checkpointer. Continue the same conversation in a different hosted session and verify that application state is restored ([step 4.1](#41-persist-graph-state-across-turns)). | [Checkpoint configuration](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) |
| User isolation | When required, enable user isolation, for example with `FoundryCheckpointSaver(user_isolation=True)`. Verify that another authenticated user cannot access that user's saved state ([step 4.1](#41-persist-graph-state-across-turns)). | [User-isolated checkpoint access](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) |
| Unfinished-request recovery | When required, configure a durable checkpointer and `resilient_background=true`, and send requests with `background=true` and `store=true`. Interrupt a running request in a test environment and verify that it resumes without duplicate external effects ([step 4.2](#42-recover-unfinished-requests)). | [Long-running work recovery](https://learn.microsoft.com/azure/foundry/agents/how-to/recover-long-running-work?tabs=responses) |
| Deployment | Prepare dependencies and the graph entry point ([step 3.1](#31-prepare-deployment-files)). Deploy and verify that the agent loads and runs ([steps 3.4](#34-deploy-remotely) and [3.5](#35-verify-the-deployed-agent)). | [Source-code deployment](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code) |
| Runtime permissions | Assign the required roles at the correct scopes to the agent's runtime identity ([step 3.4](#34-deploy-remotely)). Verify that the deployed agent can access its model, connections, and storage without relying on the developer's credentials ([step 3.5](#35-verify-the-deployed-agent)). | [Hosted agent permissions](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions) |
| Diagnostics | Enable framework tracing and connect Application Insights. Inspect console logs, system events, and correlated model and tool spans; retain request identifiers and identify gaps ([step 5.1](#51-inspect-logs-and-traces)). | [Hosted-agent tracing](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-tracing-hosted-agent); optional [Entra-authenticated ingestion](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-ingestion-entra-authentication) |
| Evaluation | Connect the existing evaluation runner to the deployed agent and rerun baseline cases. Compare per-case scores, failures, and latency; retain versioned results ([step 5.2](#52-evaluate-the-agent)). | [Managed evaluation](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-evaluate-hosted-agent); [Continuous evaluation](https://learn.microsoft.com/azure/foundry/observability/how-to/how-to-monitor-agents-dashboard#set-up-continuous-evaluation) |
