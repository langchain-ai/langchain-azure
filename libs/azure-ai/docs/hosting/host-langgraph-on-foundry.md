# Host a LangGraph application on Microsoft Foundry

If you already have a LangGraph, LangChain, or Deep Agents application, you can host it as a hosted agent on Microsoft Foundry. This guide covers hosting setup, deployment, diagnostics, and evaluation, with code examples, tips, and notes.

This guide uses a Deep Research application adapted from the [Deep Research sample](https://github.com/langchain-ai/deepagents/tree/4ff926659cfb1281b01d6821052135cef151dde3/examples/deep_research) in the [Deep Agents](https://github.com/langchain-ai/deepagents) repository. In this application, a user asks the agent to research a topic, reviews the report, and requests a revision in a later turn.

The examples use Python, uv, the hosted Responses protocol, and source-code deployment. Adapt the commands and configuration to your application's dependencies, authentication, and deployment requirements.

> **Note:** This guide covers application hosting and runtime configuration. Moving existing LangGraph threads and checkpoints to another storage backend requires a separate data migration.

Clients that use LangGraph Server APIs, custom HTTP routes, or Studio views need separate adaptation. See the [Responses hosting limitations](langgraph-to-foundry/02-run-locally.md#identify-the-existing-graph-entry-point).

## Preserve runtime-owned state

Identify required state from the original launch path and documented client workflow, not only the exported graph. LangGraph development/server runtimes can supply checkpointing outside that graph; `langgraph dev` manages persistence and rejects application-configured checkpointers. Another host does not inherit that runtime service.

| Original state owner | Re-hosting responsibility |
| --- | --- |
| The source server/runtime | Supply an equivalent server-side owner and lifetime for the state required by existing workflows. Returning updated state and expecting callers to resubmit it is a client-contract change, not equivalent server-managed continuation. |
| The application or an established external backend | Preserve that backend and its state semantics, including framework- and tool-managed state beyond messages. |
| The caller, under the existing contract | Preserve that contract; do not introduce server-side storage or recovery requirements unnecessarily. |

Ensure the serving and delegated execution paths preserve required state through the chosen owner. Conversation/thread identifiers are keys, not storage; for Invocations, a reused session does not reconstruct transcript history without a checkpointer. See [conversation and state](invocations.md#3-conversation-and-state) and [persistence configuration](langgraph-to-foundry/05-persistence-and-recovery.md#persist-graph-state-across-turns). Keep necessary construction/lifecycle adaptations focused on preserving behavior; they do not authorize business-logic redesign or historical checkpoint migration. Keep development and hosted construction separate where their ownership rules differ. Change client/server state responsibilities only with explicit user intent. For stateless operations, do not introduce continuity requirements.

## Map the application interface

Before choosing a hosting path, record a concrete contract map for every public operation in the application's original launch path that falls within the user's requested migration scope, including workflows used before or between conversations. For each operation, identify the target artifact and an actual invocation, not only a graph symbol. Graph importability and a `messages` field prove neither request nor response compatibility.

| Contract | Evidence required before choosing configuration-only hosting |
| --- | --- |
| Input | The selected protocol converts the caller's payload into the original graph input schema; a JSON string is not automatically arbitrary graph state. |
| Caller configuration | Preserve caller-provided `RunnableConfig.configurable` values and missing-value validation. Reject missing required values as the original application does; do not invent defaults or substitute platform/session identity for an application-defined parameter without an explicit approved mapping. |
| Result | The caller can obtain the operation's original result or observable side effect through the delivered interface, not just model/tool events or an HTTP success. |
| Access path | Every required operation has a runnable target command, supported endpoint and documented request. Keeping another graph in `langgraph.json` does not expose it through the selected host or preserve the original server API. |

Use configuration-only hosting only when the installed SDK's defaults implement every required mapping. Otherwise adapt input, configuration, routing and output at the hosting boundary while delegating business execution to the original graphs. Resolve the actual runnable or factory and validate adapted input against its schema; changing the protocol name or disabling a compatibility check does not implement a conversion. A companion service is a valid access path only when its required runtime and invocation are delivered and verified, not assumed to remain available from the source environment. When the user requests a whole-application migration, do not silently narrow it to its primary conversational operation: a required operation left available only in the old environment is unfinished migration work, unless the user explicitly accepts that split. Complete every required operation's target access path before declaring the whole application migrated. For a partial migration request, preserve the required operations within that scope without implicitly expanding it to the whole application.

Check each operation's managed client path as well as its local HTTP path. Foundry forwards the declared protocol requests to their protocol endpoints; adding a server route inside a container does not establish a managed access path for that operation. Expose required operations through a supported declared protocol or a separately supported ingress/service, and document the corresponding native invocation. See the [hosted agent runtime contract](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agent-contract#protocol-endpoints).

When composing a hosting boundary, preserve the selected host's graph and streaming contract, not just its input/output shape. `ResponsesHostServer` expects a `CompiledStateGraph`; an arbitrary runnable wrapper is not a drop-in replacement. Use a host-compatible compiled boundary or a supported custom handler, and preserve the invocation configuration and context when delegating to the original graphs.

When customizing a host, keep its documented hook contracts. In particular, Invocations `build_input` converts text to graph input; structured parser results have HITL semantics and are not an arbitrary graph-state transport. Use the supported input/configuration/output hooks rather than changing parser return types to bypass conversion. Read the [Invocations hook contracts](invocations.md#overridable-methods) or [Responses hosting interface](responses.md#overridable-methods) for the installed package before adapting the boundary.

## Why host your LangGraph application on Foundry?

Microsoft Foundry combines agent hosting with application interfaces, enterprise services, security controls, and diagnostics and evaluation in one platform to help enterprises deliver production-quality AI agents.

- **Managed execution that scales with your application.** Foundry manages compute, session lifecycles, and scaling around your deployed agent. Its session-isolated runtime and hosting libraries support background execution, configurable recovery, stream replay, and optional steering, helping your application handle concurrent demand, interruptions, and new user input.

- **Protocol integration built into Foundry hosting.** Expose your agent through OpenAI-compatible Responses for compatible clients, or Invocations for custom inputs and outputs. Foundry's LangGraph hosting adapters handle request conversion, response mapping, and streaming, reducing the custom integration code between your agent and client applications.

- **Managed memory, knowledge, and tools for your workflows.** Foundry Memory, Foundry IQ, and Toolbox can extend your LangGraph workflows with long-term context, enterprise knowledge, and business tools. These integrations let your application retain useful context, ground answers in organizational data, and act through connected systems.

- **Production safety and governance.** Foundry brings Entra identities, RBAC, private networking options, and configurable guardrails to hosted agents. Your organization can control who invokes the application, which resources it can access, and the security and safety policies applied during execution.

- **Diagnostics and evaluation integrated with your deployment.** Foundry connects hosted-agent execution with model and tool tracing, Application Insights, and managed evaluation. Teams can use data from real user interactions to assess execution behavior and output quality, investigate failures, validate changes, and continuously optimize their agents before and after release.

## Before you start

### Model choice

Hosting on Foundry does not require changing your model provider. You can keep your application's existing model, with its provider dependencies and credentials configured for both local and hosted execution.

To switch to a model deployed in Foundry, follow [Use LangChain with models in Microsoft Foundry](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-models) to update model initialization, authentication, and dependencies before local testing. The Foundry model configuration examples in this guide assume that this adaptation is complete.

### Common prerequisites

The following prerequisites apply before you prepare and deploy your application:

- This guide uses Python 3.13 for local testing and Foundry source-code deployment.
- You need access to the application source code and dependency definitions.
- Deployment requires [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), [Azure Developer CLI (`azd`)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd), and the Foundry agents extension described in the [deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent).
- The application's search provider requires credentials or connection information.

> **Note:** Permission notes in the linked articles describe access for your account, separate from the agent's runtime access. Skip role assignments if existing or inherited access is sufficient. Role assignment commands require `Microsoft.Authorization/roleAssignments/write` at the target scope; ask an administrator to run them if needed. In user-assignment examples, replace `<your-user-object-id>` with the Microsoft Entra object ID of the account you use for that operation.

> **Tip:** Use Azure Developer CLI (`azd`) or [Microsoft Foundry Toolkit for VS Code](https://learn.microsoft.com/azure/foundry/how-to/develop/vs-code-agents-workflow-pro-code) for local testing and deployment. If you use a coding agent, add the [Microsoft Foundry Skill](https://learn.microsoft.com/azure/foundry/how-to/develop/use-microsoft-foundry-skill) to assist with setup and troubleshooting.

The commands in this guide run on Windows, macOS, and Linux. Blocks labeled `shell` work in PowerShell and Bash. Where the syntax differs, the guide shows a PowerShell (Windows) block and a Bash (macOS and Linux) block. The Bash examples use `curl` and [`jq`](https://jqlang.org/download/) to send requests and read JSON output.

This guide assumes that the Deep Research example from the Deep Agents repository is available at `{local-root}/deep_research`. All application paths in this guide are relative to that directory. The example includes `agent.py`, `research_agent/`, `langgraph.json`, and dependency files. The guide adds Foundry hosting and deployment files alongside them.

Your local `.env` must remain outside source control and source uploads. See [Configure upload exclusions](langgraph-to-foundry/03-prepare-deployment.md#configure-upload-exclusions).

## Migration steps

Plan required state ownership before local hosting or deployment, then complete the first four articles to prepare, test and deploy the application. Apply persistence where existing behavior requires it; unfinished-request recovery, diagnostics and evaluation remain requirements-dependent.

| Stage | Article | Outcome |
| --- | --- | --- |
| Prepare | [1. Prepare Foundry resources](langgraph-to-foundry/01-prepare-resources.md) | The project, model, search access, and required configuration values are ready. |
| Local | [2. Host and test the agent locally](langgraph-to-foundry/02-run-locally.md) | The existing graph runs through the local Responses host and completes a representative request. |
| Deploy | [3. Prepare the agent for deployment](langgraph-to-foundry/03-prepare-deployment.md) | Deployment files, runtime dependencies, upload exclusions, connections, and environment variables are ready. |
| Deploy | [4. Deploy and verify the agent](langgraph-to-foundry/04-deploy-and-verify.md) | The deployed agent has the required runtime access and completes a representative hosted request. |
| Production readiness | [5. Enable persistence and recovery](langgraph-to-foundry/05-persistence-and-recovery.md) | Required state survives across turns and sessions; unfinished-request recovery is verified separately when needed. |
| Production readiness | [6. Inspect logs and traces](langgraph-to-foundry/06-diagnostics.md) | Logs and correlated traces show the expected model and tool operations for a request. |
| Production readiness | [7. Evaluate the migrated agent](langgraph-to-foundry/07-evaluation.md) | The existing evaluation cases compare the original and hosted applications against the same criteria. |

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
