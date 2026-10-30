# Inspect logs and traces

[Overview](../migrate-langgraph-to-foundry.md)

Logs and traces help you understand how the agent runs and diagnose failures or slow requests.

## Before you start

Have the deployed agent's name and version and access to invoke it. Use [Verify the deployed agent](04-deploy-and-verify.md#verify-the-deployed-agent) for a representative request. Checkpoint inspection applies when you have [configured persistence](05-persistence-and-recovery.md).

## Choose a diagnostic source

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

## Inspect a request and its console logs

Use a representative application request; the Deep Research query from [Verify the deployed agent](04-deploy-and-verify.md#verify-the-deployed-agent) is one example. Retain the agent name and version, request time, response ID, conversation ID, and hosted session ID from the invocation. Check the final Responses status and error before interpreting the logs. For a background request, retrieve the response until it reaches a terminal status, as described in [Recover unfinished requests](05-persistence-and-recovery.md#recover-unfinished-requests).

Use the hosted session ID from the invocation output to inspect its console logs:

```powershell
azd ai agent monitor deep-research-foundry --session-id "<session-id>" --tail 300
```

For startup failures or container restarts, also inspect system events:

```powershell
azd ai agent monitor deep-research-foundry --session-id "<session-id>" --type system --tail 300
```

A hosted session can contain several requests. Use request times and response IDs, where logged, to distinguish turns within the session. The [hosted agent logs guide](https://learn.microsoft.com/azure/foundry/agents/how-to/monitor-hosted-agent-logs) describes log types and streaming.

## Enable and verify traces

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

   > **Note:** Creating a new Application Insights resource or Log Analytics workspace needs resource-management permissions, such as the resource-group `Contributor` assignment shown in [Prepare Foundry resources](01-prepare-resources.md). Connecting it to the project uses the project permissions described in [Deployment prerequisites](03-prepare-deployment.md#before-you-start). The ingestion role assignment above requires role-assignment permission on Application Insights; ask an administrator to run it if needed.

3. If runtime configuration has changed, redeploy with `azd deploy deep-research-foundry` to apply it. The hosted runtime supplies `APPLICATIONINSIGHTS_CONNECTION_STRING` for the connected resource. If the deployed application already exports traces to the intended destination, no redeployment is needed for this check.
4. Open **Agents** > **Traces** in the Foundry portal and select an existing trace for the deployed agent version and request time. If no suitable trace is available, invoke the agent with a representative request, such as the Deep Research query from [Verify the deployed agent](04-deploy-and-verify.md#verify-the-deployed-agent), and allow time for telemetry ingestion. Confirm that the trace includes the model and tool operations expected for the request, not just an `invoke_agent` entry span. For the Deep Research example, verify the actual search-tool execution. Inspect subagent spans when the request delegates work. Retain the trace ID and any available response, conversation, or session attributes with the invocation record.

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

## Next steps

Previous: [Enable persistence and recovery](05-persistence-and-recovery.md) · Next: [Evaluate the migrated agent](07-evaluation.md)
