# Resource setup and operations

Commands below use Bash. Run them from this sample directory.

## Prerequisites and setup paths

Return to [the short research workflow](README.md) after completing this setup.

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
Local checks restart the host with the same temporary checkpoint storage before
reading the report back.
Install this sample's requirements first; the runner skips these local checks
when Deep Agents dependencies are missing. Deployed checks keep continuation
and new-conversation isolation in the same compute session, then stop only the
test sessions returned by those requests, including when a check fails.
The runner compares saved file pages, final assistant text, SHA256 and downloaded
Blob bytes for both versions, and checks that the original link is unchanged.
It also checks that `/report-v1.md` remains intact after creating and editing
`/report-v2.md`, including their chosen attachment filenames.
It deletes only the export names observed in those test tool results. The runner's
developer identity therefore also needs the Blob role, even for deployed tests.
Treat its local logs/report as sensitive because they contain temporary SAS URLs.

After testing, list compute sessions with `azd ai agent sessions list` and stop
each test session with `azd ai agent sessions stop <session-id>`. Stopping compute
does not delete the agent, model, Toolbox, or saved conversation state.
Delete report snapshots separately, or remove the dedicated report resource
group when finished. `azd down` only removes azd-managed resources; it does not
remove this manually created report account.

## Privacy and tracing

Both `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED` and
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` default to `false` before
host construction. Startup uses the default Azure Monitor tracing path. See the
[observability sample](../07_observability/README.md) for other exporters.
Opting in can export prompts, search queries, tool results, report text and bearer
SAS URLs. Telemetry redaction does not disable search or checkpoint persistence.

## Downloads and named working versions

`client.py` creates `/report-v1.md`, keeps it unchanged, and revises a copy at
`/report-v2.md`. `publish_report` accepts the selected filename without a leading
slash. Names must start with an ASCII letter/digit and contain only ASCII
letters, digits, hyphens or underscores (1-64 characters), followed by `.md`.
Directories, traversal, whitespace and other extensions are rejected.

Each publication copies that conversation's saved file to
`<random-prefix>/<report_name>` with a matching attachment filename. The original
working files remain editable in State Store, so a later request can select and
publish either version. Each Blob is a separate snapshot (`overwrite=False`),
not a new working-file backend.

The URL grants one-hour, read-only, HTTPS access to that one Blob to anyone who
holds it. No account key or anonymous container is used. Storage firewall/private
endpoint restrictions still apply. The five-minute start-time backdating only
tolerates clock skew; expiry is one hour after publication. Ask to publish the
same saved filename again after expiry. SAS expiry does not remove Blobs; delete
exports or configure retention separately. Keep response histories, client output
and acceptance transcripts private because they contain bearer links.

Expected Blob service or network failures are returned as handled tool errors,
not successful publication records. The agent can explain that the named report
remains saved for editing and no download link was created. The demo client stops
without writing a local download; use its latest response ID and deployed session
ID to continue the same conversation after permissions/connectivity are repaired.

The same sample has directly runnable offline checks:

```bash
python -m unittest discover -s tests
```

These simulate external model/Blob calls while exercising the real graph/host;
successful offline checks do not establish real Blob authorization or downloads.

## Deployed continuation

The client chains every turn using the latest `previous_response_id=response.id`
and sends `extra_body={"agent_session_id": session_id}` for deployed continuations.
The response ID selects conversation history/checkpoints; the session ID binds
compute. A new conversation in the same session still cannot read the previous
conversation's working files. See [session binding](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions).

For streaming, consume through `response.completed` and reject `error`,
`response.failed` and `response.incomplete`. HTTP 200 or an SSE connection ending
alone does not prove research/publication completed.
