# What this sample demonstrates

The official LangChain [Deep Research example](https://github.com/langchain-ai/deepagents/tree/b6504f6654ee6f25e1cd923087291de9e51572cc/examples/deep_research),
adapted to Microsoft Foundry and hosted over the **Responses protocol**.
It demonstrates planning, delegated web research, checkpoint-backed working files,
and downloadable reports without changing the native Deep Agents hosting contract.

## How It Works

### Deep Research

[agent.py](agent.py) uses `create_deep_agent` with `TodoListMiddleware` and
the upstream workflow: plan, save the question, delegate research, consolidate
citations, write `/final_report.md`, and read back the files to check coverage.
[research_agent/prompts.py](research_agent/prompts.py) condenses the upstream
prompts while preserving these stages and delegation guidance. Search and
delegation budgets are prompt guidance, not enforced execution limits.

[research_agent/tools.py](research_agent/tools.py) replaces upstream Tavily
search and arbitrary webpage fetching with the Foundry Toolbox `web_search`
tool. Only that tool is exposed, even if the Toolbox has other tools.
`think_tool` records concise factual progress, not private reasoning. No Tavily
key, local shell, sandbox, or custom hosting middleware is required.

[report_export.py](report_export.py) adds `publish_report` to the coordinator's
tools, not the explicit `research-agent` tools. It reads the actual saved report
from injected `ToolRuntime` state; the model cannot supply different upload text,
a destination, or permissions. Each publication creates a new private Blob
snapshot and returns its SHA256, download URL, and expiry. The working file stays
in conversation state for revisions. No Blob-backed filesystem is introduced.

### Model Integration

[main.py](main.py) connects `ChatOpenAI` to the Foundry project with
`DefaultAzureCredential`. It uses `use_responses_api=True` and
`output_version="responses/v1"` for LangChain message history; this setting is
independent of the hosted Responses protocol version.

### Agent Hosting and State

`ResponsesHostServer` hosts the ordinary compiled graph. `StateBackend` keeps
working files in LangGraph state rather than as ordinary workspace files.
`FoundryCheckpointSaver` persists that state in local files outside Foundry and
in remote State Store when hosted. It remains open until host shutdown through
`AsyncExitStack`. Local state survives a process restart when the same
`AGENTSERVER_STATE_ROOT` is retained (default: `~/.agentserver`). Continue with
`previous_response_id` to read the saved report. A new conversation does not
share these files. This is not long-term memory or in-flight crash recovery.
The host does not serve virtual files directly; `publish_report` explicitly
copies the completed report to your configured Blob container.

## Prerequisites and setup paths

Use Python 3.11 or later, Azure CLI, and Azure Developer CLI with the
`azure.ai.agents` and `azure.ai.toolboxes` extensions. Follow the parent's
[local setup instructions](../../README.md#running-the-agent-host-locally).
The sample pins `langchain-azure-ai==1.2.12`, which includes the checkpoint
selection/history-ordering fix needed for reliable local continuation.

Choose one Foundry setup path; both require the separate report Storage setup:

- **Local host, existing Foundry project:** use your existing project and model,
  then create the Toolbox below. `python main.py` does not provision them.
- **New Foundry deployment:** [azure.yaml](azure.yaml) creates the project, model,
  and Toolbox through `azd up`. Skip manual Toolbox creation and follow
  [deployment](#deploying-the-agent-to-foundry) after setting up report Storage.

### Toolbox for an existing project

Authenticate with `az login` and `azd auth login` if needed. From this directory:

```bash
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="<existing-model-deployment>"
export TOOLBOX_NAME="deep-research-tools"
azd extension install azure.ai.toolboxes
azd ai toolbox create "$TOOLBOX_NAME" --from-file toolbox.yaml --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT"
azd ai toolbox show "$TOOLBOX_NAME" --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --output json
```

Confirm `toolbox.default_version` points to a version whose `tools` includes
`{"type":"web_search","name":"web_search"}`. The first version is promoted
automatically. For an existing Toolbox, inspect it instead of recreating it;
if the desired version is not the default, explicitly promote that version:

```bash
azd ai toolbox publish "$TOOLBOX_NAME" <version> --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT"
```

The sample loads the default version at startup and exposes only `web_search`.
Restart the host after changing the default. This built-in web search needs no
Tavily key or separate project connection. See the official
[Toolbox guide](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/toolbox).

### Private report Storage

Use a dedicated Storage account/container for this sample. Set a globally unique
lowercase account name (3–24 letters/digits). These Bash commands create it with
anonymous Blob access and shared-key authentication disabled:

```bash
export SUBSCRIPTION_ID="<subscription-id>"
export REPORT_RESOURCE_GROUP="<report-resource-group>"
export LOCATION="<region>"
export REPORT_STORAGE_ACCOUNT="<unique-storage-account>"
export REPORT_STORAGE_CONTAINER="reports"
az group create --subscription "$SUBSCRIPTION_ID" --name "$REPORT_RESOURCE_GROUP" --location "$LOCATION"
az storage account create --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --name "$REPORT_STORAGE_ACCOUNT" --location "$LOCATION" --sku Standard_LRS --kind StorageV2 --https-only true --min-tls-version TLS1_2 --allow-blob-public-access false --allow-shared-key-access false
az storage container-rm create --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --storage-account "$REPORT_STORAGE_ACCOUNT" --name "$REPORT_STORAGE_CONTAINER" --public-access off
export REPORT_STORAGE_ACCOUNT_URL="$(az storage account show --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --name "$REPORT_STORAGE_ACCOUNT" --query primaryEndpoints.blob -o tsv)"
export REPORT_STORAGE_ID="$(az storage account show --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --name "$REPORT_STORAGE_ACCOUNT" --query id -o tsv)"
export DEVELOPER_OBJECT_ID="$(az ad signed-in-user show --query id -o tsv)"
az role assignment create --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role "Storage Blob Data Contributor" --scope "$REPORT_STORAGE_ID"
```

The last command needs an RBAC administrator; resource Contributor alone cannot
grant Blob access. Wait for role propagation before running. Use an equivalent
service-principal assignment when `DefaultAzureCredential` uses a service principal.

### Required roles and scopes

| Identity / operation | Required role and scope | Who configures it |
|---|---|---|
| Local developer: project model calls and Toolbox use | **Foundry User** on the Foundry project; **Reader** on the parent Foundry account for discovery | Administrator; do not assume CLI/SDK project creation granted these |
| Developer: deploy an agent and configure its project access | **Foundry Project Manager** on the Foundry project (covers Foundry User operations) | Administrator |
| Provisioner: create resources/model deployment | **Contributor** on the target resource group, or equivalent narrower control-plane permissions | Administrator |
| Administrator: manual role assignments | **Role Based Access Control Administrator** or **Owner** at each target scope | Administrator; Foundry Project Manager cannot assign Blob roles |
| Project managed identity: proxy inference to account model deployments | **Foundry User** on the parent Foundry account | `azd`/platform setup when the creator has role-assignment permission; verify for existing projects |
| Deployed agent instance identity: own-project model, Toolbox and session State Store | Platform-managed own-project access | Foundry creates the agent identity and supplies its default project/session access; advanced explicit project grants use Foundry User on the project |
| Local developer **and**, separately, deployed agent instance identity: upload and sign report links | **Storage Blob Data Contributor** on the dedicated report Storage **account** | Manual; `azure.yaml` does not create this account or these grants |
| Download recipient | No Azure role while holding a valid SAS URL | The read-only URL grants access to that one Blob until expiry |

Foundry roles were formerly named Azure AI User / Azure AI Project Manager;
their IDs are unchanged. To assign project access, set the actual resource IDs
and have an administrator run:

```bash
export FOUNDRY_ACCOUNT_ID="/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<account>"
export FOUNDRY_PROJECT_ID="$FOUNDRY_ACCOUNT_ID/projects/<project>"
az role assignment create --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role Reader --scope "$FOUNDRY_ACCOUNT_ID"
az role assignment create --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role 53ca6127-db72-4b80-b1b0-d745d6d5456d --scope "$FOUNDRY_PROJECT_ID"
```

For a deployer, use role `eadc314b-1a2d-4efa-be10-5d325db5065e` (Foundry Project
Manager) instead of the Foundry User grant. If the project managed identity lacks
its account grant, retrieve its principal and assign it separately:

```bash
export PROJECT_PRINCIPAL_ID="$(az resource show --ids "$FOUNDRY_PROJECT_ID" --api-version 2025-06-01 --query identity.principalId -o tsv)"
az role assignment create --assignee-object-id "$PROJECT_PRINCIPAL_ID" --assignee-principal-type ServicePrincipal --role 53ca6127-db72-4b80-b1b0-d745d6d5456d --scope "$FOUNDRY_ACCOUNT_ID"
```

Storage's user-delegation-key operation requires **account**, resource-group or
subscription scope. Container-only Blob Data Contributor is insufficient to sign
links. For tighter data scope, use Blob Data Contributor on the container plus
**Storage Blob Delegator** on the account instead. See
[hosted agent permissions](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions)
and [user-delegation SAS permissions](https://learn.microsoft.com/rest/api/storageservices/create-user-delegation-sas).

## Running the Agent Host

Install this sample's dependencies, copy [.env.example](.env.example) to `.env`,
and fill in the project endpoint, model deployment, Toolbox name, Storage account
URL and container from setup. Shell variables override `.env` values.

```bash
python -m pip install -r requirements.txt
python main.py
```

Both content-recording switches default to `false` before host construction.
Startup uses the default Azure Monitor tracing path. See the
[observability sample](../07_observability/README.md) for other exporters.
Opting in can export prompts, research queries, tool results, and reports.
That also includes bearer download URLs; keep both switches off when sharing
telemetry and do not treat opt-in content traces as safe to publish.
Search still sends queries to the configured search service, and checkpoints
still store conversation content; telemetry redaction does not disable either.

## Generate, read, revise and download

Run this Python example against the local host. Each continuation uses the
**latest** response ID, not the initial research response:

```python
import re
from pathlib import Path
from urllib.request import urlopen

from openai import OpenAI

with OpenAI(base_url="http://127.0.0.1:8088", api_key="local", timeout=600) as client:
    response = client.responses.create(
        input="Research LangGraph checkpoint recovery. Use one researcher, cite primary sources, save /final_report.md, and publish it."
    )
    if response.status != "completed":
        raise RuntimeError(response.error)
    print(response.output_text)
    response = client.responses.create(
        previous_response_id=response.id,
        input="Read all of /final_report.md verbatim without researching or exporting again.",
    )
    print(response.output_text)
    response = client.responses.create(
        previous_response_id=response.id,
        input="Add a Recovery limitations section using edit_file. Explain that saved checkpoints do not resume arbitrary in-flight work. Keep the existing findings and citations, read the revised file, and publish it again.",
    )
    print(response.output_text)
    download = re.search(
        r"^Download: (https://\S+)$", response.output_text, re.MULTILINE
    )
    if response.status != "completed" or download is None:
        raise RuntimeError(
            "Report export did not complete; check the response and Blob permissions."
        )
    download_url = download.group(1)
    destination = Path.home() / "Downloads" / "deep-research-revised.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(download_url, timeout=30) as downloaded_report:
        destination.write_bytes(downloaded_report.read())
    response = client.responses.create(
        previous_response_id=response.id,
        input="Read the entire updated /final_report.md verbatim without researching, editing or exporting again.",
    )
    print(response.output_text)
    print(f"Downloaded revised report: {destination}")
```

`publish_report` returns a **one-hour**, HTTPS-only, read-only user-delegation SAS
for one Blob. No account key or public container is used. Anyone holding the URL
can read that snapshot until expiry; treat it as a bearer secret, including in
response histories, logs and acceptance transcripts. Storage firewall/private
endpoint restrictions still apply. Request a new link with `publish_report` if
the previous one expires. Each revision gets a new Blob; old links still return
the old report. SAS expiry does not delete Blobs: apply a retention policy or
delete exports when no longer needed. The five-minute start-time backdating
only tolerates clock skew; expiry remains one hour after publication.

For deployed calls, use an `AIProjectClient`-created OpenAI client instead:

```python
import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

with (
    DefaultAzureCredential() as credential,
    AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
    ) as project,
    project.get_openai_client(agent_name=os.environ["DEEP_AGENTS_NAME"]) as client,
):
    response = client.responses.create(
        input="Research checkpoint recovery and publish a cited report."
    )
    session_id = response.model_extra["agent_session_id"]
    response = client.responses.create(
        previous_response_id=response.id,
        extra_body={"agent_session_id": session_id},
        input="Read /final_report.md verbatim without researching or exporting again.",
    )
```

Apply that same `extra_body` to **every** subsequent revision/read turn from the
local example while continuing to update `previous_response_id=response.id`.
`agent_session_id` binds deployed compute; `previous_response_id` chains the
conversation and checkpoints. A new conversation in the same session still
cannot read the previous conversation's virtual files. See
[session binding](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions).

### Streaming

Use `stream=True` with `client.responses.create` and consume the events through
`response.completed`. Reject `error`, `response.failed` and `response.incomplete`;
HTTP 200 or an SSE connection ending is not proof of completion. With raw HTTP,
send `"stream": true` in the request body and use `curl -N`.

## Deploying the Agent to Foundry

[azure.yaml](azure.yaml) declares a model, a web-search-only Toolbox, and a
Python 3.13 hosted agent. It does **not** provision report Storage. Complete
the Storage setup first, review resources/costs, and deploy from this directory:

```bash
azd env new deep-research --subscription <subscription-id> --location <region>
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME gpt-4.1-mini
azd env set REPORT_STORAGE_ACCOUNT_URL "$REPORT_STORAGE_ACCOUNT_URL"
azd env set REPORT_STORAGE_CONTAINER "$REPORT_STORAGE_CONTAINER"
azd up
```

`azd up` provisions the model and creates/promotes the Toolbox before deploying
the agent. Foundry creates an **agent instance identity**, separate from the
project managed identity. After deployment, retrieve it and ask your Storage
RBAC administrator to grant access before generating a report:

```bash
export AGENT_PRINCIPAL_ID="$(azd ai agent show deep-research --output json | python -c 'import json,sys; print(json.load(sys.stdin)["instance_identity"]["principal_id"])')"
az role assignment create --assignee-object-id "$AGENT_PRINCIPAL_ID" --assignee-principal-type ServicePrincipal --role "Storage Blob Data Contributor" --scope "$REPORT_STORAGE_ID"
az role assignment list --scope "$REPORT_STORAGE_ID" --include-inherited --query "[?principalId=='$AGENT_PRINCIPAL_ID'].{role:roleDefinitionName,scope:scope}" -o table
```

The role must be assigned to `instance_identity.principal_id`, not the project's
managed identity or the blueprint. It is not inherited from the deploying user.
Allow propagation, and re-check the principal after redeployment. A deployment
can be `active` even when this external Blob permission is missing. See the
[source-code deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code).

## Acceptance and cleanup

Run the shared JSON/SSE checks for generation, complete report delivery, restart
recovery (local), revision, Blob downloads, and conversation isolation:

```bash
export FOUNDRY_PROJECT_ENDPOINT="$(azd env get-value FOUNDRY_PROJECT_ENDPOINT)"
export DEEP_AGENTS_NAME=langchain-azure-deep-research
export REPORT_STORAGE_ACCOUNT_URL="$(azd env get-value REPORT_STORAGE_ACCOUNT_URL)"
export REPORT_STORAGE_CONTAINER="$(azd env get-value REPORT_STORAGE_CONTAINER)"
python ../../tests/run_samples_e2e.py responses/12_deep_agents
```

For local checks, unset `DEEP_AGENTS_NAME` and set `FOUNDRY_PROJECT_ENDPOINT`,
`TOOLBOX_NAME` and both `REPORT_STORAGE_*` values in the shell. The runner launches
the local host. Both modes make real model and search calls and can incur charges.
Local checks restart the host
with the same temporary checkpoint storage before reading the report back.
Install this sample's requirements first; the runner skips these local checks
when Deep Agents dependencies are missing. Deployed checks keep continuation
and new-conversation isolation in the same compute session, then stop only the
test sessions returned by those requests, including when a check fails.
The runner compares saved file pages, final assistant text, SHA256 and downloaded
Blob bytes for both versions, and checks that the original link is unchanged.
It deletes only the export names observed in those test tool results. The runner's
developer identity therefore also needs the Blob role, even for deployed tests.
Treat its local logs/report as sensitive because they contain temporary SAS URLs.

After testing, list compute sessions with `azd ai agent sessions list` and stop
each test session with `azd ai agent sessions stop <session-id>`. Stopping compute
does not delete the agent, model, Toolbox, or saved conversation state.
Delete report snapshots separately, or remove the dedicated report resource
group when finished. `azd down` only removes azd-managed resources; it does not
remove this manually created report account.
