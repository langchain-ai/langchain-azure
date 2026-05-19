# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent whose
tools are loaded from a **remote MCP server** via
[`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters)
and hosted over the **Responses protocol**. By default the sample
connects to GitHub's remote MCP server at
`https://api.githubcopilot.com/mcp/`, but `MCP_SERVER_URL` can point at
any HTTP-transport MCP endpoint.

## How It Works

### Model Integration

The agent uses `langchain_openai.AzureChatOpenAI` against the Foundry
project endpoint, authenticated with
`DefaultAzureCredential`.

See [main.py](main.py) for the full implementation.

### Tools — loaded from a remote MCP server

[`langchain_mcp_adapters.client.MultiServerMCPClient`](https://github.com/langchain-ai/langchain-mcp-adapters):

- opens a streamable-HTTP session against the configured MCP server,
- forwards an `Authorization: Bearer <token>` header sourced from
  `GITHUB_PAT`, and
- returns standard LangChain `BaseTool` instances ready to plug into any
  LangGraph / LangChain agent.

Because `MultiServerMCPClient.get_tools()` is asynchronous, this sample
fetches the tools once at startup with `asyncio.run(...)` and then
builds the `create_agent` graph synchronously — the same shape as
[`responses/04_foundry_toolbox`](../04_foundry_toolbox/) but pointed at
a generic remote MCP endpoint instead of a Foundry-managed gateway.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.LangGraphResponsesHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol.

## Prerequisites

1. A [GitHub Personal Access Token](https://github.com/settings/tokens)
   with the scopes needed by the GitHub MCP tools you want the agent to
   call (e.g. `repo`, `read:user`).
2. Set the token in `.env` as `GITHUB_PAT`.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
README in the parent directory. This sample additionally requires
`GITHUB_PAT` (and optionally `MCP_SERVER_URL` if you
target a different MCP server).

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details.

Ask the agent a question that exercises one of the GitHub MCP tools:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "List my 5 most recently updated GitHub repos."}'
```

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Search GitHub issues mentioning langchain-azure.", "stream": true}'
```

Intermediate `function_call` / `function_call_output` items are
surfaced for every MCP tool the agent invokes — same shape as the
local-tools sample, but the tool execution happens inside the remote
MCP server.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory. Make sure
`GITHUB_PAT` is set on the deployed container — either
through `azd env set GITHUB_PAT <token>` before
`azd deploy`, or by wiring it to a secret store referenced from
[agent.manifest.yaml](agent.manifest.yaml).
