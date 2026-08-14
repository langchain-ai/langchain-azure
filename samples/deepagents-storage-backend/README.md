# Deep Agents Azure Blob Storage Backend Samples

Runnable examples for using
[`AzureBlobBackend`](../../libs/azure-storage/langchain_azure_storage/deepagents/backend.py)
as a [Deep Agents](https://github.com/langchain-ai/deepagents) filesystem backend, so an
agent's workspace lives in Azure Blob Storage instead of process memory.

## Resources these samples create

Each script creates a blob container named **`agent-workspace`** in your storage account,
or reuses it if it already exists. Nothing else is provisioned, and nothing is deleted —
the files each agent writes are left in place so you can inspect them afterwards.

Each sample writes under its own prefix, so they don't interfere with each other:

| Sample | Prefix in `agent-workspace` |
|---|---|
| `basic_agent.py` | `session-001/` |
| `resume_workspace.py` | `research-session/` |
| `composite_with_memories.py` | `composite-demo/memories/`, `composite-demo/workspace/` |

To clean up everything the samples created:

```bash
az storage container delete --name agent-workspace --account-name <your-account> --auth-mode login
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Python 3.11+ (required by the `deepagents` extra)
- An Azure Storage account, and permission to create a container in it. The
  [Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles/storage#storage-blob-data-contributor)
  role covers both.
- A chat model to drive the agents (see [Configuring the model](#configuring-the-model))

To run without an Azure Storage account, see
[Running against the Azurite emulator](#running-against-the-azurite-emulator).

## Setup

Create a `.env` file in this directory (`.env` is gitignored):

```env
AZURE_STORAGE_ACCOUNT_URL=https://<your-account>.blob.core.windows.net
MODEL_NAME=<your-model>
```

Credentials come from
[`DefaultAzureCredential`](https://learn.microsoft.com/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview),
so signing in with the Azure CLI is enough:

```bash
az login
```

`DefaultAzureCredential` also picks up managed identity, workload identity, and
environment-variable credentials. To use a different credential type, pass `credential=`
to `AzureBlobBackend` directly.

### Storage environment variables

| Variable | Description |
|---|---|
| `AZURE_STORAGE_ACCOUNT_URL` | Blob endpoint of your storage account, authenticated with `DefaultAzureCredential`. |
| `AZURE_STORAGE_CONNECTION_STRING` | Alternative: authenticate with a connection string. Takes precedence when both are set. |

### Configuring the model

`MODEL_NAME` selects the chat model. To run the samples entirely on Azure, set it to your
model deployment name and set one Azure AI endpoint variable — the samples then build the
model with
[`langchain-azure-ai`](../../libs/azure-ai#microsoft-foundry-models), authenticated with
`DefaultAzureCredential`:

```env
MODEL_NAME=gpt-5.5
AZURE_AI_PROJECT_ENDPOINT=https://<your-resource>.services.ai.azure.com/api/projects/<your-project>
```

`AZURE_AI_OPENAI_ENDPOINT` and `AZURE_OPENAI_ENDPOINT` work too; see
[`AzureAIOpenAIApiChatModel`](../../libs/azure-ai/langchain_azure_ai/chat_models/openai.py)
for how each is resolved.

With no Azure AI endpoint set, `MODEL_NAME` is passed through to
[`init_chat_model`](https://docs.langchain.com/oss/python/langchain/models) as a
`provider:model` identifier, so any supported provider works:

```env
MODEL_NAME=anthropic:claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

## Running the samples

Each sample uses [PEP 723 inline script metadata](https://peps.python.org/pep-0723/), so
uv installs the dependencies automatically — no separate install step. The scripts pin
`langchain-azure-storage` to this repository checkout via `[tool.uv.sources]`, so they
run against your local code; if you copy a sample elsewhere, delete that block to use the
released package instead.

### Basic agent ([basic_agent.py](basic_agent.py))

A minimal Deep Agent whose workspace persists in Azure Blob Storage. After the run it
lists the workspace and prints the blob URL each file landed at.

```bash
cd samples/deepagents-storage-backend
uv run --env-file .env basic_agent.py
```

### Resuming a workspace ([resume_workspace.py](resume_workspace.py))

The demo only a durable backend can run: one agent writes research notes and is torn
down completely; a brand-new backend and agent then attach to the same prefix and
summarize what they find. State survives because it lives in Blob Storage, not in
process memory.

```bash
cd samples/deepagents-storage-backend
uv run --env-file .env resume_workspace.py
```

### Composite backend with memory and subagents ([composite_with_memories.py](composite_with_memories.py))

Routes part of the agent's filesystem to Azure Blob Storage with
[`CompositeBackend`](https://docs.langchain.com/oss/python/deepagents/backends#compositebackend-router):
`/memories/` holds an `AGENTS.md` that survives every run, `/workspace/` holds the shared
working files, and everything else stays thread-scoped in `StateBackend`.

That last part matters: Deep Agents writes its own bookkeeping into the backend (offloaded
large tool results under `/large_tool_results/`, conversation history under
`/conversation_history/`), and with a bare backend those land in your container next to the
agent's real output. Routing only the prefixes you care about keeps the container clean.

A coder and a tester subagent share the durable `/workspace/`, so the coder's files are
immediately visible to the tester. The run is streamed so the output attributes each file
operation to the agent that performed it, then prints what each route persisted.

```bash
cd samples/deepagents-storage-backend
uv run --env-file .env composite_with_memories.py
```

## Browsing the results

In the [Azure portal](https://portal.azure.com), open your storage account and go to
**Data storage > Containers > agent-workspace**. The prefixes in
[Resources these samples create](#resources-these-samples-create) appear as folders; open
one to read the files an agent wrote.

[Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer)
works too, and is the easiest option when running against Azurite: connect to the **Local
storage emulator** with its default settings and browse `agent-workspace` under
**devstoreaccount1 > Blob Containers**.

## Running against the Azurite emulator

To run the samples with no Azure Storage account, use the
[Azurite](https://learn.microsoft.com/azure/storage/common/storage-use-azurite) emulator.
You still need a chat model.

Start it with [Docker](https://docs.docker.com/get-docker/):

```bash
docker run -d --name azurite -p 10000:10000 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck
```

(`--skipApiVersionCheck` keeps the emulator working when the `azure-storage-blob` client
library is newer than the Azurite image.)

Then point the samples at it with Azurite's well-known development connection string,
replacing `AZURE_STORAGE_ACCOUNT_URL` in your `.env`:

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;
```

That account name and key are Azurite's
[published defaults](https://learn.microsoft.com/azure/storage/common/storage-use-azurite#well-known-storage-account-and-key),
identical for every Azurite install — not a secret. The shorthand
`UseDevelopmentStorage=true` is a .NET convention that the Python SDK does not accept, so
the full string is required here.

To tear the emulator down:

```bash
docker rm -f azurite
```
