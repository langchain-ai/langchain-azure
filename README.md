# 🦜️🔗 LangChain Azure

Python integrations for building LangGraph, LangChain, and Deep Agents applications on Azure.
Each package is maintained independently and supports Python 3.11 through 3.14.

## Packages

Install the packages for the Azure services your application uses.
Package names link to their documentation;
links in the Purpose column lead to usage guides and samples.

| Package | Purpose |
| --- | --- |
| [langchain-azure-ai](./libs/azure-ai/README.md) | Call Microsoft Foundry [chat models](#get-started-with-microsoft-foundry-models) and [embedding models](https://docs.langchain.com/oss/python/integrations/providers/azure_ai#embedding-models).<br>Use [existing Foundry agents as graph nodes](./libs/azure-ai/README.md#microsoft-foundry-agent-service).<br>Access [tools, skills, MCP, memory, guardrails, tracing, and Azure AI Search](#explore-azure-integrations).<br>Host [LangGraph applications as Foundry hosted agents](#host-applications-on-microsoft-foundry). |
| [langchain-azure-compute](./libs/azure-compute/README.md) | Run agent-generated code with [Azure Container Apps dynamic sessions](./libs/azure-compute/README.md#dynamic-sessions).<br>Provide Deep Agents compute backends with [dynamic sessions](./libs/azure-compute/README.md#deep-agents-backend) and [Azure Container Apps sandboxes](./libs/azure-compute/README.md#sandboxes-deep-agents-backend). |
| [langchain-azure-cosmosdb](./libs/azure-cosmosdb/README.md) | Use [Azure Cosmos DB for vector search, semantic caching, and chat history](./samples/cosmosdb-nosql/README.md#langchain-integrations).<br>Persist [LangGraph checkpoints, node caches, and data shared across threads](./samples/cosmosdb-nosql/README.md#langgraph-integrations) in Azure Cosmos DB. |
| [langchain-sqlserver](./libs/sqlserver/README.md) | Use [SQL Server and Azure SQL for vector search and chat history](https://github.com/Azure-Samples/azure-sql-langchain). |
| [langchain-azure-postgresql](./libs/azure-postgresql/README.md) | Use [Azure Database for PostgreSQL for vector search with DiskANN indexing](./libs/azure-postgresql/README.md#usage). |
| [langchain-azure-storage](./libs/azure-storage/README.md) | Load documents with the [Azure Blob Storage document loader](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage).<br>Persist Deep Agents files with the [Azure Blob Storage filesystem backend](./samples/deepagents-storage-backend/README.md). |
| [langchain-azure-dynamic-sessions](./libs/azure-dynamic-sessions/README.md) | **Deprecated.** Azure Container Apps dynamic sessions integrations. [Migrate to langchain-azure-compute](./libs/azure-dynamic-sessions/README.md). |

For help choosing between `langchain-azure-ai` and `langchain-openai`,
see the [Microsoft integrations guide](https://docs.langchain.com/oss/python/integrations/providers/microsoft#choose-a-package).

## Get started with Microsoft Foundry models

Call [Microsoft Foundry models](https://learn.microsoft.com/azure/foundry/concepts/foundry-models-overview) through LangChain's chat model interface.
The example below shows how to authenticate and make your first model call with `langchain-azure-ai`.

### Install langchain-azure-ai

Install the package and sign in to Azure with the Azure CLI.

```bash
pip install -U langchain-azure-ai
az login
```

### Call a model

This example uses the Responses API.
Use a Microsoft Foundry project with a model deployment that supports the Responses API.
Your identity needs the [**Foundry User**](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-models#prerequisites) role on the project.
Replace the project endpoint and model deployment name below.

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

model = AzureAIOpenAIApiChatModel(
    project_endpoint=(
        "https://{your-resource-name}.services.ai.azure.com"
        "/api/projects/{your-project}"
    ),
    credential=DefaultAzureCredential(),
    model="{your-model-deployment-name}",
    use_responses_api=True,
)

model.invoke("Translate 'Hello!' into Italian.").pretty_print()
```

For endpoint and authentication options, see the [model quick start](./libs/azure-ai/README.md#microsoft-foundry-models)
and the [LangChain model guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-models).

## Host applications on Microsoft Foundry

Deploy your LangChain, LangGraph, or Deep Agents application to Microsoft Foundry as a [hosted agent](https://learn.microsoft.com/azure/foundry/agents/concepts/hosted-agents).
The `langchain-azure-ai` hosting adapter connects your application to Foundry's agent APIs.

For an existing LangGraph application, point `langgraph.json` at its compiled graph.
With the application's dependencies, credentials, and model settings configured,
run these commands from the application directory to start a local Responses endpoint:

```bash
pip install -U "langchain-azure-ai[hosting]"
python -m langchain_azure_ai.agents.hosting.run --protocol responses
```

See the [hosting quick start](./libs/azure-ai/README.md#hosting-langgraph-agents-on-microsoft-foundry)
for configuration and examples, and the [LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents)
for Azure deployment.

## Explore Azure integrations

Connect your application to Azure services for search, document loading, tools, memory, and tracing.
Browse the guides and samples by integration type or agent capability:

- **Chat models**: Generate responses with models deployed in Microsoft Foundry. See the [Foundry model quick start](#get-started-with-microsoft-foundry-models).
- **Graph nodes**: Use [an agent already running in Microsoft Foundry as a LangGraph node](./libs/azure-ai/README.md#microsoft-foundry-agent-service). Also see the [Document Intelligence sample](./samples/react-agent-docintelligence/README.md).
- **Tools, skills and MCPs**: Load [MCP tools](https://reference.langchain.com/python/langchain-azure-ai/tools/_toolbox/AzureAIProjectToolbox/get_tools) and [agent skills](https://reference.langchain.com/python/langchain-azure-ai/tools/_toolbox/AzureAIProjectToolbox/get_resources) from Foundry Toolbox. Also see the [Document Intelligence tool sample](./samples/react-agent-docintelligence/README.md).
- **Sandbox**: Use [Azure Container Apps dynamic sessions and sandboxes](./libs/azure-compute/README.md) to run agent-generated codes.
- **Backends**: Use [Azure Blob Storage](./samples/deepagents-storage-backend/README.md) as storage backends.
- **Checkpointers**: Persist graph execution state with [FoundryCheckpointSaver backed by Foundry State Store](./libs/azure-ai/README.md#durable-graph-state) in Foundry-hosted agents, or the [Azure Cosmos DB checkpointer](./samples/cosmosdb-nosql/README.md#langgraph-integrations).
- **Stores**: Store data shared across threads in [Azure Cosmos DB](./samples/cosmosdb-nosql/README.md#langgraph-integrations).
- **Memory**: Extract and recall long-term information across sessions with [Foundry Memory](./samples/foundry-memory/README.md).
- **Embedding models**: Convert text into vectors for similarity search. See the [embedding model guide](https://docs.langchain.com/oss/python/integrations/providers/azure_ai#embedding-models).
- **Vector stores**: Store vectors and retrieve relevant documents. See the [Azure AI Search sample](./samples/rag-storage-document-loaders/README.md), [Azure Cosmos DB samples](./samples/cosmosdb-nosql/README.md#langchain-integrations), [Azure SQL and SQL Server samples](https://github.com/Azure-Samples/azure-sql-langchain), and [Azure Database for PostgreSQL quick start](./libs/azure-postgresql/README.md#usage).
- **Document loaders**: Load documents from [Azure Blob Storage](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage). Also see the [Azure AI Content Understanding loader](./libs/azure-ai/docs/content_understanding_loader_demo.ipynb).
- **Middleware**: Apply Content Safety guardrails to agent inputs, outputs, and tool calls. See the [Content Safety middleware sample](./samples/unified-moderation/basic_usage.py).
- **Observability**: Trace model calls and agent execution. See the [Azure Application Insights tracing sample](./samples/enable_auto_tracing_appinsights.py).

## Samples

Use the [samples](./samples) to try individual integrations and build applications with Azure services.
They cover databases, memory, storage, tools, guardrails, and tracing.

## Contributing

We welcome contributions!
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pull request guidelines.
