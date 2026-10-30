# Prepare the agent for deployment

[Overview](../migrate-langgraph-to-foundry.md)

Once you have verified that the agent runs correctly with the local Responses host, proceed with deploying it to Foundry.

## Before you start

Complete [Host and test the agent locally](02-run-locally.md). Keep the verified graph entry point and the resource identifiers from [Prepare Foundry resources](01-prepare-resources.md) available.

> **Note:** Your account needs `Foundry Project Manager` on the Foundry project to create project connections and deploy the source-code agent. An administrator can grant this access with the following command. This assignment is for your account, not the agent identity. See the [source-code deployment permissions](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code#required-permissions).
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry Project Manager" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>/projects/<project-name>"
> ```

## Prepare deployment files

### Configure azure.yaml

The `codeConfiguration.entryPoint` setting in `azure.yaml` specifies the hosting runner command, as shown in [Host an existing agent](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents#host-an-existing-agent).

`langgraph.json` tells the runner how to load the agent, both locally and in Foundry. `azure.yaml` tells azd how to deploy and start the hosted agent. The deployment package must include `langgraph.json` and the application files it references. See [Package the zip manually](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code#package-the-zip-manually) for the package layout and `requirements.txt`.

Initialize a source-code deployment from `{local-root}\deep_research`:

```powershell
azd auth login
azd ai agent init --protocol responses --deploy-mode code --runtime python_3_13 --entry-point '-m langchain_azure_ai.agents.hosting.run --protocol responses --config langgraph.json'
```

During initialization, select the existing project from [Prepare Foundry resources](01-prepare-resources.md) and name the agent `deep-research-foundry`. The initialization flow creates `azure.yaml` and an azd environment. The agent service's `project` field should be `.` so the upload starts from the application root directory.

The following agent service configuration starts the hosting runner with the graph configuration tested in [Host and test the agent locally](02-run-locally.md):

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

### Prepare runtime dependencies

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

### Configure upload exclusions

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

## Configure project connections for third-party endpoints

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

The hosted agent identity also needs permission to read these connection credentials. Assign that access after deployment creates the identity, as shown in [Deploy remotely](04-deploy-and-verify.md#deploy-remotely).

> **Note:** Local tests that read the project connection use your developer credentials. Your account also needs connection-credential access on the parent Foundry resource; the `Foundry User` assignment in the local-testing note in [Host and test the agent locally](02-run-locally.md) covers this example. The agent's runtime assignment does not grant access to your account.

> **Tip:** In this example, the Tavily client is initialized when `research_agent.tools` is imported, so the connection lookup must complete before graph initialization. Other applications can retrieve credentials when initializing or invoking the dependent tool. The hosting runner does not resolve a connection name automatically.

## Set environment variables

The hosted runtime uses deployment configuration, not your local `.env`. Pass the model deployment name and connection name to the agent; the tool code from [Configure project connections](#configure-project-connections-for-third-party-endpoints) retrieves the secret at runtime. Foundry supplies the project endpoint automatically.

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

## Verify deployment preparation

Restart the local host with the command from [Host and test the agent locally](02-run-locally.md) and verify that the application can search using the project connection before deployment. After this check succeeds, you can remove `TAVILY_API_KEY` from your local `.env`.

## Next steps

Previous: [Host and test the agent locally](02-run-locally.md) · Next: [Deploy and verify the agent](04-deploy-and-verify.md)
