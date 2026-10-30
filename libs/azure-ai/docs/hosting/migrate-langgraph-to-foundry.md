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

> **Note:** Permission notes in the linked articles describe access for your account, separate from the agent's runtime access. Skip role assignments if existing or inherited access is sufficient. Role assignment commands require `Microsoft.Authorization/roleAssignments/write` at the target scope; ask an administrator to run them if needed. In user-assignment examples, replace `<your-user-object-id>` with the Microsoft Entra object ID of the account you use for that operation.

> **Tip:** Use Azure Developer CLI (`azd`) or [Microsoft Foundry Toolkit for VS Code](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-agents-workflow-pro-code) for local testing and deployment. If you use a coding agent, add the [Microsoft Foundry Skill](https://learn.microsoft.com/azure/foundry/how-to/develop/use-microsoft-foundry-skill) to assist with setup and troubleshooting.

This guide assumes that the Deep Research example from the Deep Agents repository is available at `{local-root}\deep_research`. All application paths in this guide are relative to that directory. The example includes `agent.py`, `research_agent/`, `langgraph.json`, and dependency files. The guide adds Foundry hosting and deployment files alongside them.

Your local `.env` must remain outside source control and source uploads. See [Configure upload exclusions](langgraph-to-foundry/03-prepare-deployment.md#configure-upload-exclusions).

## Migration steps

Complete the first four articles to prepare, test, and deploy your application. Then use the remaining articles for persistence and recovery, diagnostics, and evaluation as your application requires.

| Article | Outcome |
| --- | --- |
| [1. Prepare Foundry resources](langgraph-to-foundry/01-prepare-resources.md) | The project, model, search access, and required configuration values are ready. |
| [2. Host and test the agent locally](langgraph-to-foundry/02-run-locally.md) | The existing graph runs through the local Responses host and completes a representative request. |
| [3. Prepare the agent for deployment](langgraph-to-foundry/03-prepare-deployment.md) | Deployment files, runtime dependencies, upload exclusions, connections, and environment variables are ready. |
| [4. Deploy and verify the agent](langgraph-to-foundry/04-deploy-and-verify.md) | The deployed agent has the required runtime access and completes a representative hosted request. |
| [5. Enable persistence and recovery](langgraph-to-foundry/05-persistence-and-recovery.md) | Required state survives across turns and sessions; unfinished-request recovery is verified separately when needed. |
| [6. Inspect logs and traces](langgraph-to-foundry/06-diagnostics.md) | Logs and correlated traces show the expected model and tool operations for a request. |
| [7. Evaluate the migrated agent](langgraph-to-foundry/07-evaluation.md) | The existing evaluation cases compare the original and hosted applications against the same criteria. |

## Migration acceptance checklist

Use the stage column to find the checks for resource preparation, local testing, deployment, or production readiness. Diagnostics can also help during local development and deployment. The production-readiness checks below cover the behaviors in this guide; apply them to your application's requirements. The **Further reading** links cover optional next steps.

> **Note:** Confirm your account's access using the permission notes in the linked articles. These permissions are separate from the agent's runtime access checked below.

| Stage | Check | Applies to | Action and verification | Further reading |
| --- | --- | --- | --- | --- |
| Prepare | Resource preparation | All applications | Prepare or reuse the project, model, and search services. Collect the required endpoints, resource IDs, and application configuration ([Prepare Foundry resources](langgraph-to-foundry/01-prepare-resources.md#verify-resource-preparation)). | [Create a Foundry project](https://learn.microsoft.com/azure/foundry/how-to/create-projects) |
| Local | Local execution | All applications | Load the existing graph through the Responses host. Verify completion, expected tool calls, and application output ([Host and test the agent locally](langgraph-to-foundry/02-run-locally.md#verify-local-execution)). | [LangGraph hosting](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents) |
| Deploy | Deployment | All applications | Prepare dependencies and the graph entry point ([Prepare deployment files](langgraph-to-foundry/03-prepare-deployment.md#prepare-deployment-files)). Deploy and verify that the agent loads and runs ([Deploy remotely](langgraph-to-foundry/04-deploy-and-verify.md#deploy-remotely) and [Verify the deployed agent](langgraph-to-foundry/04-deploy-and-verify.md#verify-the-deployed-agent)). | [Source-code deployment](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent-code) |
| Deploy | Runtime permissions | All applications | Assign the required roles at the correct scopes to the agent's runtime identity ([Deploy remotely](langgraph-to-foundry/04-deploy-and-verify.md#deploy-remotely)). Verify that the deployed agent can access its model, connections, and storage without relying on the developer's credentials ([Verify the deployed agent](langgraph-to-foundry/04-deploy-and-verify.md#verify-the-deployed-agent)). | [Hosted agent permissions](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agent-permissions) |
| Deploy | Single-turn execution | All applications | Invoke the deployed agent with a representative request. Verify the final status, required tool calls, and application outputs ([Verify the deployed agent](langgraph-to-foundry/04-deploy-and-verify.md#verify-the-deployed-agent)). | [Hosted agent invocation](https://learn.microsoft.com/azure/foundry/agents/how-to/invoke-hosted-agent) |
| Production readiness | Multi-turn behavior | Applications that retain context across turns | When required, retain the authenticated user and conversation across turns. Verify that follow-up requests use prior context and update application state as intended ([Persist graph state across turns](langgraph-to-foundry/05-persistence-and-recovery.md#persist-graph-state-across-turns)). | [Session lifecycle and management](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions) |
| Production readiness | Durable continuation | State must survive a hosted session | When required, configure a durable checkpointer. Continue the same conversation in a different hosted session and verify that application state is restored ([Persist graph state across turns](langgraph-to-foundry/05-persistence-and-recovery.md#persist-graph-state-across-turns)). | [Checkpoint configuration](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) |
| Production readiness | User isolation | Applications with user-specific state | When required, enable user isolation, for example with `FoundryCheckpointSaver(user_isolation=True)`. Verify that another authenticated user cannot access that user's saved state ([Persist graph state across turns](langgraph-to-foundry/05-persistence-and-recovery.md#persist-graph-state-across-turns)). | [User-isolated checkpoint access](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) |
| Production readiness | Unfinished-request recovery | Running requests must survive host failures | When required, configure a durable checkpointer and `resilient_background=true`, and send requests with `background=true` and `store=true`. Interrupt a running request in a test environment and verify that it resumes without duplicate external effects ([Recover unfinished requests](langgraph-to-foundry/05-persistence-and-recovery.md#recover-unfinished-requests)). | [Long-running work recovery](https://learn.microsoft.com/azure/foundry/agents/how-to/recover-long-running-work?tabs=responses) |
| Production readiness | Diagnostics | Applications preparing for production | Enable framework tracing and connect Application Insights. Inspect console logs, system events, and correlated model and tool spans; retain request identifiers and identify gaps ([Inspect logs and traces](langgraph-to-foundry/06-diagnostics.md)). | [Hosted-agent tracing](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-tracing-hosted-agent); optional [Entra-authenticated ingestion](https://learn.microsoft.com/azure/foundry/observability/how-to/trace-ingestion-entra-authentication) |
| Production readiness | Evaluation | Applications preparing for production | Connect the existing evaluation runner to the deployed agent and rerun baseline cases. Compare per-case scores, failures, and latency; retain versioned results ([Evaluate the migrated agent](langgraph-to-foundry/07-evaluation.md)). | [Managed evaluation](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-evaluate-hosted-agent); [Continuous evaluation](https://learn.microsoft.com/azure/foundry/observability/how-to/how-to-monitor-agents-dashboard#set-up-continuous-evaluation) |
