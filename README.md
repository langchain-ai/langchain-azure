# 🦜️🔗 LangChain Azure

This repository contains the following packages with Azure integrations with LangChain:

- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-cosmosdb](https://pypi.org/project/langchain-azure-cosmosdb/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)
- [langchain-azure-postgresql](https://pypi.org/project/langchain-azure-postgresql/)
- [langchain-azure-storage](https://pypi.org/project/langchain-azure-storage/)

**Note**: This repository will replace all Azure integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

# Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the Microsoft Foundry family of SDKs and client libraries for Azure to provide first-class support of Microsoft Foundry capabilities in LangChain and LangGraph.

This package includes:

* [Microsoft Agent Service](./libs/azure-ai/langchain_azure_ai/agents)
* [Microsoft Foundry Models inference](./libs/azure-ai/langchain_azure_ai/chat_models)
* [Microsoft Foundry Content Safety](./libs/azure-ai/langchain_azure_ai/agents/middleware)
* [Microsoft Foundry Tools](./libs/azure-ai/langchain_azure_ai/tools)
* [Azure AI Search](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Azure AI Services tools](./libs/azure-ai/langchain_azure_ai/tools)

Here's a quick start example to show you how to get started with the Chat Completions model. For more details and tutorials see [Develop with LangChain and LangGraph and models from Azure AI Foundry](https://aka.ms/azureai/langchain).

### Install langchain-azure

```bash
pip install -U langchain-azure-ai
```

### Microsoft Foundry Models

Use any Foundry Model with OpenAI-compatible APIs:

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIOpenAIApiChatModel(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential(), # requires Azure AI Developer role. If using keys, use parameter `endpoint` instead of `project_endpoint`.
    model="gpt-5"                        # use any OpenAI-compatible model, like Mistral-Large-3
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

To use `init_chat_model` you must set the `AZURE_AI_PROJECT_ENDPOINT`, and (optional) `OPENAI_API_KEY` environment variables. Use the provider `azure_ai`:

```python 
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
load_dotenv()

# Option A) Using project endpoint
os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "https://{your-resource-name}.services.ai.azure.com/api/projects/<project>"

# Option B) Using OpenAI endpoint
os.environ["OPENAI_API_BASE"] = "https://{your-resource-name}.services.ai.azure/openai/v1"
os.environ["OPENAI_API_KEY"] = "{your-key}"

model = init_chat_model("azure_ai:gpt-5-mini")
```

### Microsoft Foundry Agent Service

You can build multi agent graphs in LangGraph by using the integration with Microsoft Foundry Agent Service. The class `AgentServiceFactory` allows you to create agents and nodes that can be used to compose graphs.

```python
from azure.identity import DefaultAzureCredential
from langchain_core.messages import AIMessage, HumanMessage
from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.utils.agents import pretty_print

factory = AgentServiceFactory(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential()
)

echo_node = factory.get_agent_node(name="my-echo-agent", version="latest")
```

Agent Service nodes run in Microsoft Foundry but can be added to any graph:

```python
graph.add_node("expert_node", echo_node)
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

# Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

