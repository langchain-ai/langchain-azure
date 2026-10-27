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

### Execution identities and required roles

Confirm these prerequisites **before running resource commands**. The examples
can involve different people: the local **developer** calls Foundry and publishes
reports, a **provisioner** creates resources, and an **RBAC administrator** grants
access. A person may fill several roles, but permission to create a resource does
not imply permission to grant roles or access its Blob data.

| Identity / operation | Required role and scope | Who configures it |
|---|---|---|
| Local developer: project model calls and Toolbox use | **Foundry User** on the Foundry project; **Reader** on the parent Foundry account for discovery | Administrator; do not assume CLI/SDK project creation granted these |
| Developer: deploy an agent and configure its project access | **Foundry Project Manager** on the Foundry project (covers Foundry User operations) | Administrator |
| Provisioner: create resources/model deployment | **Contributor** on the target resource group, or equivalent narrower control-plane permissions | Administrator; creating the resource group itself also needs resource-group creation permission at subscription scope |
| Administrator: manual role assignments | **Role Based Access Control Administrator** or **Owner** at each target scope | Administrator; Foundry Project Manager cannot assign Blob roles |
| Project managed identity: proxy inference to account model deployments | **Foundry User** on the parent Foundry account | `azd`/platform setup when the creator has role-assignment permission; verify for existing projects |
| Deployed agent instance identity: own-project model, Toolbox and session State Store | Platform-managed own-project access | Foundry creates the identity and supplies its default project/session access; advanced explicit project grants use Foundry User on the project |
| Local developer **and**, separately, deployed agent instance identity: upload and sign report links | **Storage Blob Data Contributor** on the dedicated report Storage **account** | Manual; `azure.yaml` does not create this account or these grants |
| Download recipient | No Azure role while holding a valid SAS URL | The read-only URL grants access to that one Blob until expiry |

The project managed identity and the deployed agent's `instance_identity` are
different principals. Do not grant the agent's external Blob access to the project
identity or blueprint instead. Foundry roles were formerly called Azure AI User /
Azure AI Project Manager; their IDs are unchanged. See the official
[hosted agent permissions](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions).

Authenticate the CLI identities with `az login` and `azd auth login` if needed,
then set shared, non-secret values. The examples use an interactive user as the
developer; use the actual service-principal object ID and `ServicePrincipal`
instead of `User` for automation or a different `DefaultAzureCredential` source.

```bash
export SUBSCRIPTION_ID="<subscription-id>"
export LOCATION="<region>"
export DEVELOPER_OBJECT_ID="$(az ad signed-in-user show --query id -o tsv)"
```

If an administrator is executing grants in a different login, carry forward the
**developer's** object ID; do not replace it with the administrator's. Stop if the
role-assignment commands below are denied and ask that administrator to run them.

### Toolbox for an existing project

Before creating or using the Toolbox, have the RBAC administrator grant project
access to the developer. Set the existing account/project resource IDs:

```bash
export FOUNDRY_ACCOUNT_ID="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<account>"
export FOUNDRY_PROJECT_ID="$FOUNDRY_ACCOUNT_ID/projects/<project>"
```

**RBAC administrator:** grant the developer account discovery and project use:

```bash
az role assignment create --subscription "$SUBSCRIPTION_ID" --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role Reader --scope "$FOUNDRY_ACCOUNT_ID"
az role assignment create --subscription "$SUBSCRIPTION_ID" --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role 53ca6127-db72-4b80-b1b0-d745d6d5456d --scope "$FOUNDRY_PROJECT_ID"
```

The existing project's managed identity also needs its account inference grant.
If it is missing, the administrator retrieves and grants that distinct principal:

```bash
export PROJECT_PRINCIPAL_ID="$(az resource show --ids "$FOUNDRY_PROJECT_ID" --api-version 2025-06-01 --query identity.principalId -o tsv)"
az role assignment create --subscription "$SUBSCRIPTION_ID" --assignee-object-id "$PROJECT_PRINCIPAL_ID" --assignee-principal-type ServicePrincipal --role 53ca6127-db72-4b80-b1b0-d745d6d5456d --scope "$FOUNDRY_ACCOUNT_ID"
```

**Developer:** wait for the grants to propagate, then create/configure the Toolbox
from this directory. Existing model deployments are not created by this path:

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

Use a dedicated Storage account/container. The **provisioner** needs Contributor
on the target resource group before these create commands; an administrator must
arrange that access on an existing group. If creating a new resource group, run
the first command with an identity allowed to create groups at subscription scope,
or ask the platform administrator to create the group instead. Set a globally
unique lowercase account name (3–24 letters/digits):

```bash
export REPORT_RESOURCE_GROUP="<report-resource-group>"
export REPORT_STORAGE_ACCOUNT="<unique-storage-account>"
export REPORT_STORAGE_CONTAINER="reports"
az group create --subscription "$SUBSCRIPTION_ID" --name "$REPORT_RESOURCE_GROUP" --location "$LOCATION"
```

**Provisioner:** create the account and private container with anonymous Blob
access and shared-key authentication disabled:

```bash
az storage account create --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --name "$REPORT_STORAGE_ACCOUNT" --location "$LOCATION" --sku Standard_LRS --kind StorageV2 --https-only true --min-tls-version TLS1_2 --allow-blob-public-access false --allow-shared-key-access false
az storage container-rm create --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --storage-account "$REPORT_STORAGE_ACCOUNT" --name "$REPORT_STORAGE_CONTAINER" --public-access off
export REPORT_STORAGE_ACCOUNT_URL="$(az storage account show --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --name "$REPORT_STORAGE_ACCOUNT" --query primaryEndpoints.blob -o tsv)"
export REPORT_STORAGE_ID="$(az storage account show --subscription "$SUBSCRIPTION_ID" --resource-group "$REPORT_RESOURCE_GROUP" --name "$REPORT_STORAGE_ACCOUNT" --query id -o tsv)"
```

**RBAC administrator:** resource creation does not grant Blob data access. Before
the developer runs the host or client, grant upload and SAS-signing access to
the actual developer credential:

```bash
az role assignment create --subscription "$SUBSCRIPTION_ID" --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role "Storage Blob Data Contributor" --scope "$REPORT_STORAGE_ID"
```

Wait for propagation; resource Contributor alone cannot make this assignment or
upload Blob data. The hosted agent needs a separate grant after its identity is
created in the [deployment](#deploying-the-agent-to-foundry) step.

Storage's user-delegation-key operation requires **account**, resource-group or
subscription scope. Container-only Blob Data Contributor is insufficient to sign
links. For tighter data scope, use Blob Data Contributor on the container plus
**Storage Blob Delegator** on the account instead. See
[hosted agent permissions](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions)
and [user-delegation SAS permissions](https://learn.microsoft.com/rest/api/storageservices/create-user-delegation-sas).

## Deploying the Agent to Foundry

[azure.yaml](azure.yaml) declares a model, a web-search-only Toolbox, and a
Python 3.13 hosted agent. It does **not** provision report Storage. Complete
the Storage setup first and review resources/costs.

**Provisioner/deployer:** before `azd up`, confirm Contributor on the intended
resource group (and subscription-level resource-group creation permission if
needed). For a new project, the provisioning identity also needs role-assignment
authority so the project/model grants can be configured. If you only have
Contributor, have an administrator provision/configure access first rather than
expecting Contributor to assign roles. Deployment to an existing project needs
Foundry Project Manager on that project; the RBAC administrator can grant it with
the existing-project IDs from above:

```bash
az role assignment create --subscription "$SUBSCRIPTION_ID" --assignee-object-id "$DEVELOPER_OBJECT_ID" --assignee-principal-type User --role eadc314b-1a2d-4efa-be10-5d325db5065e --scope "$FOUNDRY_PROJECT_ID"
```

This grant covers Foundry User operations too. For the new-project route, the
administrator arranges the equivalent access as the project is provisioned.
**Provisioner/deployer:** only after those prerequisites are in place, run:

```bash
azd env new deep-research --subscription "$SUBSCRIPTION_ID" --location "$LOCATION"
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME gpt-4.1-mini
azd env set REPORT_STORAGE_ACCOUNT_URL "$REPORT_STORAGE_ACCOUNT_URL"
azd env set REPORT_STORAGE_CONTAINER "$REPORT_STORAGE_CONTAINER"
azd up
```

`azd up` provisions the model and creates/promotes the Toolbox before deploying
the agent. Foundry creates an **agent instance identity**, separate from the
project managed identity. **Deployer:** retrieve the instance principal after
deployment (this grant cannot be made before the identity exists):

```bash
export AGENT_PRINCIPAL_ID="$(azd ai agent show deep-research --output json | python -c 'import json,sys; print(json.load(sys.stdin)["instance_identity"]["principal_id"])')"
```

**Storage RBAC administrator:** grant the instance identity's external Blob
access before the developer runs the deployed client or generates a report:

```bash
az role assignment create --subscription "$SUBSCRIPTION_ID" --assignee-object-id "$AGENT_PRINCIPAL_ID" --assignee-principal-type ServicePrincipal --role "Storage Blob Data Contributor" --scope "$REPORT_STORAGE_ID"
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
