# 🦜️🔗 LangChain Azure

This repository contains the following packages with Azure integrations with LangChain:

- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-compute](./libs/azure-compute)
- [langchain-azure-cosmosdb](https://pypi.org/project/langchain-azure-cosmosdb/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)
- [langchain-azure-postgresql](https://pypi.org/project/langchain-azure-postgresql/)
- [langchain-azure-storage](https://pypi.org/project/langchain-azure-storage/)

**Note**: This repository will replace all Azure integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## LangGraph on Azure

- **Models & Hosted Agents (Microsoft Foundry):** Build with Azure models using the [LangChain and LangGraph development guide](https://aka.ms/azureai/langchain), or bring an existing compiled graph to Microsoft Foundry with the [LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents) and the [hosted-agent samples](./samples/hosting/langgraph-hosted-agents/README.md#start-here) (including [hosting an existing graph without changing its implementation](./samples/hosting/langgraph-hosted-agents/responses/09_run/README.md)).
- **Checkpointers & Long-Term Memory:** For durable graph state, use [FoundryCheckpointSaver backed by Foundry State Store](./libs/azure-ai/README.md#durable-graph-state) in Foundry-hosted agents, or the [Azure Cosmos DB checkpointer](./libs/azure-cosmosdb/README.md#langgraph-checkpointer). Cosmos DB also provides a [long-term memory store](./libs/azure-cosmosdb/README.md#langgraph-store-long-term-memory).
- **Durable Execution (Azure Durable Task Scheduler):** Run any compiled LangGraph `StateGraph`, LangChain `create_agent`, or Deep Agent as a fault-tolerant durable workflow using [`langchain_azure_compute.graph`](./libs/azure-compute/README.md#durable-langgraph-execution-durable-task-scheduler) backed by the [Azure Durable Task Scheduler](https://learn.microsoft.com/azure/durable-task/scheduler/durable-task-scheduler) — with per-step and per-tool fault isolation (`@durable_tool`, `@durable_node`), automatic worker crash failover, and zero-compute human-in-the-loop `interrupt()` / `resume()` waits. Explore the [10 runnable Durable Task LangGraph samples](./samples/langgraph-durabletask/README.md).

## Samples

- [SDK samples](https://github.com/langchain-ai/langchain-azure/tree/main/samples) demonstrate SDK usage, including hosting, memory, and storage integrations.
- [Durable Task Scheduler (DTS) LangGraph samples](./samples/langgraph-durabletask/README.md) provide 10 runnable examples covering durable graphs, `create_agent` ReAct agents, per-tool retry policies, human-in-the-loop approvals, multi-turn SQLite/Cosmos state, and Azure Functions HTTP hosting.
- [SDK notebooks](https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/docs) cover Foundry models, Agent Service, Content Safety, and Content Understanding.
- [Foundry deployment samples](https://github.com/microsoft-foundry/foundry-samples/tree/main/samples/python/hosted-agents/langgraph) demonstrate platform configuration and deployment with LangGraph.

# Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the Microsoft Foundry family of SDKs and client libraries for Azure to provide first-class support of Microsoft Foundry capabilities in LangChain and LangGraph.

This package includes:

* [Microsoft Agent Service](./libs/azure-ai/langchain_azure_ai/agents)
* [Microsoft Foundry Models inference](./libs/azure-ai/langchain_azure_ai/chat_models)
* [Microsoft Foundry Content Safety](./libs/azure-ai/langchain_azure_ai/agents/middleware)
* [Microsoft Foundry Tools](./libs/azure-ai/langchain_azure_ai/tools)
* [Azure AI Search](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Azure AI Services tools](./libs/azure-ai/langchain_azure_ai/tools)

Here's a quick start example to show you how to get started with a chat model. For more details and tutorials see [Develop with LangChain and LangGraph and models from Azure AI Foundry](https://aka.ms/azureai/langchain).

### Install langchain-azure

```bash
pip install -U langchain-azure-ai
```

### Microsoft Foundry Models

Use any Foundry Model with OpenAI-compatible APIs:

We recommend the [Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses) when your model and endpoint support it. Set `use_responses_api=True` as shown below. If you prefer Chat Completions and your model and endpoint support it, set `use_responses_api=False` instead.

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIOpenAIApiChatModel(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential(), # requires Azure AI Developer role. If using keys, use parameter `endpoint` instead of `project_endpoint`.
    model="gpt-5",                       # your model deployment name
    use_responses_api=True,
)

messages = [
    SystemMessage(
      content="Translate the following from English into Italian"
    ),
    HumanMessage(content="hi!"),
]

model.invoke(messages).pretty_print()
```

```output
================================== Ai Message ==================================
Ciao!
```

Use the `azure_ai` provider with `init_chat_model`. Choose one of the
following authentication options.

**Option A: Project endpoint with Microsoft Entra ID**

Set `AZURE_AI_PROJECT_ENDPOINT` and sign in with `az login` for local
development:

```python
import os

from langchain.chat_models import init_chat_model

os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "https://{your-resource-name}.services.ai.azure.com/api/projects/<project>"

model = init_chat_model("azure_ai:gpt-5-mini")
model.invoke("Hello!").pretty_print()
```

**Option B: OpenAI-compatible endpoint with an API key**

Set `OPENAI_API_BASE` and `OPENAI_API_KEY`. Leave `AZURE_AI_PROJECT_ENDPOINT`
unset when using this option:

```python
import os

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_BASE"] = "https://{your-resource-name}.services.ai.azure.com/openai/v1"
os.environ["OPENAI_API_KEY"] = "{your-key}"

model = init_chat_model("azure_ai:gpt-5-mini")
model.invoke("Hello!").pretty_print()
```

### Microsoft Foundry Agent Service

Use `AgentServiceFactory` to create a managed Foundry agent and add it as a
node in a LangGraph graph.

For local development, install the required extra and sign in:

```bash
pip install -U "langchain-azure-ai[opentelemetry]"
az login
```

Set your project endpoint and model deployment name. This creates a version
of `my-echo-agent` in your project:

```python
from azure.identity import DefaultAzureCredential
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_azure_ai.agents import AgentServiceAgentState, AgentServiceFactory
from langchain_azure_ai.utils.agents import pretty_print

factory = AgentServiceFactory(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential()
)

echo_node = factory.create_prompt_agent_node(
    name="my-echo-agent",
    model="gpt-4.1",  # your model deployment name
    instructions="Always reply with the opposite of what the user says.",
)
```

Agent Service nodes run in Microsoft Foundry but can be added to any graph:

```python
graph = StateGraph(AgentServiceAgentState)
graph.add_node("expert_node", echo_node)
graph.add_edge(START, "expert_node")
graph.add_edge("expert_node", END)
```

Use the graph as usual:

```python
agent = graph.compile()
messages = [HumanMessage(content="I'm a genius and I love programming!")]
response = agent.invoke({"messages": messages})

pretty_print(response)
```

```output
================================ Human Message =================================

I'm a genius and I love programming!
================================== Ai Message ==================================
Name: my-echo-agent

You're not a genius and you don't love programming!
```

### Durable Execution with Azure Durable Task Scheduler

To run workflows on the [Azure Durable Task Scheduler](https://learn.microsoft.com/azure/durable-task/scheduler/durable-task-scheduler), install the durable execution extra alongside `langchain-azure-ai` from the quick start above:

```bash
pip install -U "langchain-azure-compute[durabletask]"
```

Set `AZURE_AI_PROJECT_ENDPOINT` and `AZURE_AI_MODEL_DEPLOYMENT_NAME` for your Foundry
model and authenticate with Microsoft Entra ID (`az login` for local development).
For the scheduler, set `DURABLE_TASK_SCHEDULER_CONNECTION_STRING`, or set
`DTS_ENDPOINT=127.0.0.1:8080` when using a running local DTS emulator.
`sdk_options()` resolves the scheduler connection and authentication settings.

Wrap an existing compiled `StateGraph`, LangChain agent, or Deep Agent with
`attach_durability`. This example creates a support agent whose read-only tool
runs as an isolated activity with up to three attempts:

```python
import os

from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain.agents import create_agent
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_azure_compute.graph import attach_durability, durable_tool, sdk_options
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@durable_tool(retries=3)
@tool
def get_order_status(order_id: str) -> str:
    """Look up an order's status (replace with your order service)."""
    return f"Order {order_id} has shipped."


model = AzureAIOpenAIApiChatModel(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
    model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
    use_responses_api=True,
)
agent = create_agent(model=model, tools=[get_order_status])
reg = attach_durability(agent, name="support_agent")

opts = sdk_options()
worker = DurableTaskSchedulerWorker(**opts)
sdk = DurableTaskSchedulerClient(**opts)
try:
    reg.register(worker)
    worker.start()
    run = reg.client(sdk).start(
        {"messages": [HumanMessage(content="What is the status of order 42?")]}
    )
    result = run.wait_result(timeout=60)
    print(result["messages"][-1].content)
finally:
    worker.stop()
    sdk.close()
```

For production setup, checkpoint storage, idempotency, and human-in-the-loop
patterns, see the [full `langchain-azure-compute` documentation](./libs/azure-compute/README.md#durable-langgraph-execution-durable-task-scheduler)
and the [10 runnable samples](./samples/langgraph-durabletask/README.md).

# Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
