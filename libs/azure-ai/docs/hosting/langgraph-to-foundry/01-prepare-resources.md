# Prepare Foundry resources

Before hosting your existing LangGraph application on Foundry, prepare the resources it needs. This first step in the [migration guide](../migrate-langgraph-to-foundry.md) covers a Foundry project, model access, and search access. The Deep Research example uses a Foundry model deployment and Tavily search. You can reuse existing resources where possible.

## Before you start

Review the [common prerequisites](../migrate-langgraph-to-foundry.md#before-you-start). Have access to the application source code and your Azure subscription.

## Prepare resources

**Foundry project.** The hosted agent needs a project in your Azure subscription. For setup instructions, see [Create a project for Microsoft Foundry](https://learn.microsoft.com/azure/foundry/how-to/create-projects). The deployment configuration uses the project endpoint, subscription ID, and project resource ID. The connection-access role assignment in [Deploy remotely](04-deploy-and-verify.md#deploy-remotely) also needs the parent Foundry resource ID.

**Model access.** The agent needs a model that supports the application's API and tool calls. Using a Foundry model is not a prerequisite for hosting your agent on Foundry; you can keep your existing model provider. This example uses a Foundry model deployment and its deployment name in the agent configuration. If you want to set up and use a Foundry model, see [Deploy a Foundry model](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/deploy-foundry-models) and [Call a Foundry model](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/generate-responses).

**Search tool.** This research application needs a web search tool to gather sources for its reports. You can keep the existing Tavily integration or replace it with search tools exposed through Foundry Toolbox. The rest of this guide uses Tavily's search endpoint. The [Tavily Python SDK quickstart](https://docs.tavily.com/sdk/python/quick-start) and [Build a toolbox and use it with a hosted agent](https://learn.microsoft.com/azure/foundry/agents/quickstarts/quickstart-toolbox-agent) describe these two options.

> **Tip:** You can keep the existing search provider for the first deployment and replace it after agent hosting works.

**Runtime access.** The hosted agent uses its own identity to access models and connected resources. Model inference through its own project endpoint is available by default; configure additional access where the application needs it, such as reading connection credentials in [Deploy remotely](04-deploy-and-verify.md#deploy-remotely). See the [hosted agent permissions reference](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions).

> **Note:** If you create Foundry resources or deploy models instead of reusing existing resources, your account needs the corresponding resource-management permissions. For example, an administrator can grant `Contributor` on the existing resource group where you create them. This role does not grant role-assignment permissions. See [Azure resource setup](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions#azure-resource-setup).
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Contributor" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>"
> ```

## Verify resource preparation

Before continuing, have the project endpoint, subscription ID, project ARM ID, parent Foundry resource ARM ID, and your application's model and search configuration ready.

## Next steps

Previous: [Migration overview](../migrate-langgraph-to-foundry.md) · Next: [Host and test the agent locally](02-run-locally.md)
