# Prepare the agent for deployment

After [verifying the local host](02-run-locally.md), prepare the files and configuration Foundry will use to build and start the same application. This article covers deployment files, runtime dependencies, upload exclusions, project connections, and environment variables. By the end, the application will be ready for the deployment and runtime-access checks in the next article.

## Before you start

Complete [Host and test the agent locally](02-run-locally.md). Keep the verified graph entry point and the resource identifiers from [Prepare Foundry resources](01-prepare-resources.md) available.

> **Note:** Your account needs `Foundry Project Manager` on the Foundry project to create project connections and deploy the source-code agent. An administrator can grant this access with the following command. This assignment is for your account, not the agent identity. See the [source-code deployment permissions](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code#required-permissions).
>
> PowerShell (Windows):
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry Project Manager" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>/projects/<project-name>"
> ```
>
> Bash (macOS and Linux):
>
> ```bash
> az role assignment create \
>     --assignee-object-id "<your-user-object-id>" \
>     --assignee-principal-type User \
>     --role "Foundry Project Manager" \
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>/projects/<project-name>"
> ```

## Prepare deployment files

### Add a deployment launcher

Create `main.py` in the application root, `{local-root}/deep_research`. It calls the hosting runner with an explicit argument list and reuses the graph entry point in `langgraph.json`:

```python
from langchain_azure_ai.agents.hosting.run import main

if __name__ == "__main__":
    main(["--protocol", "responses", "--config", "langgraph.json"])
```

The launcher passes runner options as separate arguments. The deployment entry point stays a file path.

### Configure azure.yaml

`langgraph.json` tells the runner how to load the agent, both locally and in Foundry. `azure.yaml` tells azd how to deploy and start the hosted agent. The deployment package must include `main.py`, `langgraph.json`, and the application files it references. See [Package the zip manually](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code#package-the-zip-manually) for the package layout and `requirements.txt`.

Initialize a source-code deployment from `{local-root}/deep_research`:

```shell
azd auth login
azd ai agent init --protocol responses --deploy-mode code --runtime python_3_13 --entry-point main.py
```

During initialization, select the existing project from [Prepare Foundry resources](01-prepare-resources.md) and name the agent `deep-research-foundry`. The initialization flow creates `azure.yaml` and an azd environment. Retain the agent service's generated `name` when adapting `azure.yaml`; the top-level project `name` and service key do not replace it. The agent service's `project` field should be `.` so the upload starts from the application root directory.

The following agent service configuration starts the hosting runner with the graph configuration tested in [Host and test the agent locally](02-run-locally.md):

```yaml
services:
  deep-research-foundry:
    host: azure.ai.agent
    kind: hosted
    name: deep-research-foundry
    project: .
    language: python
    startupCommand: python main.py
    protocols:
      - protocol: responses
        version: 2.0.0
    codeConfiguration:
      runtime: python_3_13
      entryPoint: main.py
      dependencyResolution: remote_build
```

The remote Python runtime starts the file specified by `codeConfiguration.entryPoint`. The explicit `startupCommand` is for the local CLI path; it runs the same launcher from the application root using the prepared Python environment on `PATH`. Install the runtime dependencies before using this command. If you use uv to create or sync the local project environment, keep the `.python-version` pin alongside the local project files and verify that the local run uses the selected target Python minor version. A successful dependency install in a separate environment does not verify the submitted startup path. See the [azure.yaml reference](https://learn.microsoft.com/azure/foundry/agents/concepts/azure-yaml-reference) and [source-code deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code).

Declare application environment variables in the agent service's `env` map in `azure.yaml`. Do not copy environment fields from the deployed API representation into that file. Include the variables the application actually requires; do not make an optional setting mandatory by adding an unresolved placeholder. Run `azd ai agent doctor --local-only` to check the delivered service definition and configuration before packaging. This check does not establish remote deployment or runtime behavior.

### Prepare runtime dependencies

In this example, the remote build installs dependencies from the root `requirements.txt`. A locally installable PEP 621 `pyproject.toml` does not by itself establish that the selected remote builder supports that dependency workflow. Include the application's runtime dependencies and choose compatible versions. For the Foundry-model path used in the configuration examples, these include:

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

If you keep the original Anthropic model, also include `langchain-anthropic>=1.5.4,<2.0.0` in `requirements.txt`. Keep deployment dependencies consistent with the final application code.

Check that these requirements install in a fresh Python 3.13 environment outside the project:

PowerShell (Windows):

```powershell
$checkEnv = Join-Path $env:TEMP ("foundry-deps-" + [guid]::NewGuid().ToString("N"))
uv venv --python 3.13 $checkEnv
uv pip install --python "$checkEnv\Scripts\python.exe" --prerelease allow -r requirements.txt
uv pip check --python "$checkEnv\Scripts\python.exe"
```

Bash (macOS and Linux):

```bash
checkEnv=$(mktemp -d "${TMPDIR:-/tmp}/foundry-deps-XXXXXX")
uv venv --python 3.13 "$checkEnv"
uv pip install --python "$checkEnv/bin/python" --prerelease allow -r requirements.txt
uv pip check --python "$checkEnv/bin/python"
```

Both installation and the dependency check must succeed before deployment. This verifies dependency compatibility, not application startup.

Start `main.py` with the fresh environment's interpreter and the required environment variables, then send a representative request that exercises the research tools.

### Configure upload exclusions

Before writing `.agentignore`, identify the dependency inputs the remote builder uses and keep them in the upload. For example, keep `pyproject.toml` and the lockfile when the builder relies on them. Start with exclusions for local credentials and generated files, then adapt them to your application:

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
```

Your `.gitignore` should also exclude local credentials, azd environment files, and generated results. In this example, the remote build uses `requirements.txt`. Keep it in the upload; omit the development manifest, lockfile, or notebook only if the remote builder and application do not use them.

Run `azd package` before deployment and inspect the upload for the declared launcher, referenced application files, and dependency inputs. Packaging verifies the upload contents, not a successful remote build. The upload must exclude `.env`, local caches, evaluation output, and development-only dependencies.

## Configure project connections for third-party endpoints

Preserve the application's existing credential inputs unless a change is required. If it reads `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`, keep those names and supply values through an approved runtime secret delivery mechanism. Do not copy a local `.env`, hard-code keys in `azure.yaml`, or add unresolved secret placeholders to the application. Verify the selected mechanism for the deployment mode before claiming hosted credential access works.

The remainder of this section is an optional application-managed connection example. Use it if you choose to retrieve secrets through the project SDK instead of preserving the existing secret delivery path. Follow [Add a new connection to your project](https://learn.microsoft.com/azure/foundry/how-to/connections-add). If you explicitly choose this pattern for an external model too, use a separate connection for its credentials.

This example uses a `CustomKeys` connection named `tavily-search` with target `https://api.tavily.com`. You can obtain a key from the [Tavily Python SDK quickstart](https://docs.tavily.com/sdk/python/quick-start) and store it in the connection's secret field named `api_key`.

Add the project SDK to the dependencies before updating the Tavily client initialization:

```shell
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

The hosted runtime uses deployment configuration, not your local `.env`. Foundry supplies the project endpoint automatically. The configuration below applies only to the Foundry-model and project-connection example. Omit settings for paths your application does not use.

If you keep another model provider, use the existing configuration required by that model client. Only set `TAVILY_CONNECTION_NAME` if you implemented the optional [project-connection path](#configure-project-connections-for-third-party-endpoints).

This example uses the following values:

| Variable | Value |
| --- | --- |
| `FOUNDRY_PROJECT_ENDPOINT` | Foundry injects the project endpoint into the hosted runtime. Your local host reads it from `.env`. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Your selected Foundry model deployment name |
| `TAVILY_CONNECTION_NAME` | Your project connection name for Tavily credentials |

Set the model deployment name and connection name in the selected azd environment:

```shell
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME "<deployment-name>"
azd env set TAVILY_CONNECTION_NAME tavily-search
```

For this source-code example, add these values to the agent service's `env` map in `azure.yaml`:

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

Restart the local host with the configuration your application actually uses and verify a request that exercises its model and search tool. If you switched to an application-managed connection, verify that path before removing the old local API key. Verify hosted access separately after deployment.

## Next steps

Previous: [Host and test the agent locally](02-run-locally.md) · [Migration overview](../host-langgraph-on-foundry.md) · Next: [Deploy and verify the agent](04-deploy-and-verify.md)
