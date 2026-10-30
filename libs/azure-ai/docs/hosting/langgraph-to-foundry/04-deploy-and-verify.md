# Deploy and verify the agent

With the [deployment files and configuration ready](03-prepare-deployment.md), you can deploy the application to Foundry. This article walks through deployment, granting the agent's runtime identity access to its project connection, and testing a hosted request. The final check verifies that the deployed agent can use its model and search tool and return the expected output.

## Before you start

Complete [Prepare the agent for deployment](03-prepare-deployment.md), including the account permissions in its [prerequisites](03-prepare-deployment.md#before-you-start). Run commands from the application root in the configured azd environment.

## Deploy remotely

Deploy the configured application to create an agent version and its runtime identity. Run `azd provision` first if new resources or infrastructure changes are required. For an existing project with no infrastructure changes:

```powershell
azd deploy deep-research-foundry
$agentInfo = azd ai agent show deep-research-foundry --output json | ConvertFrom-Json
$agentInfo | ConvertTo-Json -Depth 10
```

### Grant runtime connection access

Before invoking the agent, grant its instance identity `Foundry User` on the parent Foundry resource to read the connection credentials from [Configure project connections](03-prepare-deployment.md#configure-project-connections-for-third-party-endpoints). Skip this assignment if the identity already has sufficient access. See the [hosted agent permissions reference](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions).

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

Confirm that the deployment reports the expected agent version and instance identity, and that the required role assignment succeeds. Then verify runtime access with a hosted request in [Verify the deployed agent](#verify-the-deployed-agent).

## Verify the deployed agent

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

## Next steps

Choose [persistence and recovery](05-persistence-and-recovery.md) when your application needs them. Use [diagnostics](06-diagnostics.md) to inspect requests and [evaluation](07-evaluation.md) to compare migration results.

Previous: [Prepare the agent for deployment](03-prepare-deployment.md) · [Migration overview](../migrate-langgraph-to-foundry.md) · Next: [Enable persistence and recovery](05-persistence-and-recovery.md)
