# Evaluate the migrated agent

[Overview](../migrate-langgraph-to-foundry.md)

Logs and traces help verify that the agent executes as expected, but successful execution does not establish answer quality. Evaluation is therefore important to check whether the migrated agent still produces useful, accurate answers that meet user needs.

## Before you start

Have a [verified deployment](04-deploy-and-verify.md#verify-the-deployed-agent), an existing evaluation dataset, and a baseline from the original application. For multi-turn cases, review the [conversation identity and state requirements](05-persistence-and-recovery.md#persist-graph-state-across-turns).

Reuse your existing evaluation dataset, evaluators, and pass criteria. Change how the evaluation calls the agent, not what it measures. You can continue to use your local evaluation tools to test the agent deployed in Foundry. You can also run managed evaluations in Foundry and set up continuous evaluation of live agent traffic.

> **Note:** Your existing evaluation runner needs the invocation access from [Verify the deployed agent](04-deploy-and-verify.md#verify-the-deployed-agent) and access to its judge model, if used.

> **Note:** For Foundry portal or SDK evaluations, your account needs `Foundry User` on the project. Foundry model inference also needs account-scoped access; the assignment below covers both for this example. For CI, assign access to the workload identity instead. See [evaluation permissions](https://learn.microsoft.com/azure/foundry/observability/how-to/evaluation-permissions) for additional workflows.
>
> ```powershell
> az role assignment create `
>     --assignee-object-id "<your-user-object-id>" `
>     --assignee-principal-type User `
>     --role "Foundry User" `
>     --scope "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<resource-name>"
> ```

## Connect your existing evaluation tools to the deployed agent

1. Retain a baseline run from the original application, including its dataset version, evaluator configuration, and per-case results.
2. Replace the evaluation runner's local graph invocation with a call to the deployed Foundry agent. Follow the [hosted Responses SDK and REST invocation examples](https://learn.microsoft.com/azure/foundry/agents/how-to/manage-hosted-sessions?pivots=python#invoke-an-agent-and-let-the-platform-create-the-session), targeting the agent version verified in [Verify the deployed agent](04-deploy-and-verify.md#verify-the-deployed-agent). Wait for each response to complete before scoring it.
3. Map the hosted response back to the fields your existing evaluators expect. Preserve reference answers and case metadata. If an evaluator previously read graph state or a saved report, provide equivalent evidence from the hosted execution; response text alone might not be sufficient.
4. Run the same cases and scoring logic against the hosted agent. Use a separate conversation for each independent case. For multi-turn cases, preserve the authenticated user and conversation as described in [Persist graph state across turns](05-persistence-and-recovery.md#persist-graph-state-across-turns).

Custom evaluators can remain in your existing runner. Evaluators that depend on local callbacks, tool outputs, or traces need an adapter for hosted execution. Use the model and tool traces from [Inspect logs and traces](06-diagnostics.md) where applicable, and report missing evidence rather than treating it as a pass.

> **Tip:** Optionally, upload saved queries, responses, and reference data to [evaluate the saved responses in Foundry](https://learn.microsoft.com/azure/foundry/observability/how-to/cloud-evaluation-datasets). This creates evaluation runs for centralized tracking and [comparison in the portal](https://learn.microsoft.com/azure/foundry/how-to/evaluate-results#compare-the-evaluation-results). It runs evaluators on saved responses without invoking the agent again; it does not import scores already calculated by your local tools.

## Compare results before and after migration

Run the same test cases before and after migration, keeping the agent model, judge model (if used), search provider, and evaluation settings unchanged. Compare per-case scores, failures, and latency against your existing pass criteria. If live search returns different evidence, review it before attributing a score change to migration.

Save both runs with the dataset version, evaluation settings, agent version, and available trace IDs. Include failed or incomplete cases, and keep credentials and sensitive content out of the results.

> **Tip:** As a next step, consider [running managed evaluations in Foundry](https://learn.microsoft.com/azure/foundry/observability/quickstarts/quickstart-evaluate-hosted-agent) and setting up [continuous evaluation of live agent traffic](https://learn.microsoft.com/azure/foundry/observability/how-to/how-to-monitor-agents-dashboard#set-up-continuous-evaluation).

## Next steps

Previous: [Inspect logs and traces](06-diagnostics.md) · Next: [Migration acceptance checklist](../migrate-langgraph-to-foundry.md#migration-acceptance-checklist)
