# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent whose
tools are loaded from an **Azure AI Foundry Toolbox** — a managed
multi-MCP gateway that aggregates many tool servers (custom MCP, OpenAPI
tools, SharePoint search, etc.) behind one URL. The toolbox tools are
fetched once at startup and bound to a `create_agent` graph hosted using
the **Responses protocol**.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` against the Foundry
project endpoint, authenticated with
`DefaultAzureCredential`.

See [main.py](main.py) for the full implementation.

### Tools — loaded from a Foundry Toolbox

[`langchain_azure_ai.tools.AzureAIProjectToolbox`](../../../../libs/azure-ai/langchain_azure_ai/tools):

- authenticates with `DefaultAzureCredential` (`az login` works),
- injects the required `Foundry-Features` header,
- sanitizes the tool schemas, and
- returns standard LangChain `BaseTool` instances ready to plug into any
  LangGraph / LangChain agent.

Because `AzureAIProjectToolbox.get_tools()` is asynchronous, this sample
fetches the tools once at startup with `asyncio.run(...)` and then
builds the `create_agent` graph synchronously — the same shape as
[`responses/02_tools`](../02_tools/) but with Foundry-managed tools
instead of a local `@tool`.

The set of tools available is whatever you configured on the toolbox
named `TOOLBOX_NAME` (default `agent-tools`). The shipped
[agent.manifest.yaml](agent.manifest.yaml) declares two managed tools to
get you started:

```yaml
resources:
  - kind: toolbox
    name: agent-tools
    tools:
      - type: web_search
        name: web_search
      - type: code_interpreter
        name: code_interpreter
```

Edit the manifest to add or remove tools before deploying.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.ResponsesHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory. This sample additionally requires `TOOLBOX_NAME` to be
set to a toolbox configured in your Foundry project.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

Ask the agent something that exercises one of your toolbox tools:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "What tools do you have available?"}'
```

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "<a question your toolbox can answer>", "stream": true}'
```

Intermediate `function_call` / `function_call_output` items are
surfaced for every toolbox tool the agent invokes — same shape as the
local-tools sample, but the tool execution happens inside the toolbox
server.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory. When you `azd provision`, the
toolbox resource declared in [agent.manifest.yaml](agent.manifest.yaml)
is created in your Foundry project and `TOOLBOX_NAME` is auto-injected
into the container.
