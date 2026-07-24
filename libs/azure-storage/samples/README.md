# Deep Agents Backend Samples

Runnable examples for using [`AzureBlobBackend`](../langchain_azure_storage/deepagents/backend.py)
as a [Deep Agents](https://github.com/langchain-ai/deepagents) filesystem backend.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- [Docker](https://docs.docker.com/get-docker/) installed (for the Azurite emulator)
- An [Anthropic API key](https://console.anthropic.com/) (the samples use the default
  Anthropic model)
- Python 3.11+ (required by the `deepagents` extra)

## Setup

### 1. Start Azurite (local Azure Storage emulator)

```bash
docker run -d --name azurite -p 10000:10000 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite-blob --blobHost 0.0.0.0 --skipApiVersionCheck
```

(`--skipApiVersionCheck` keeps the emulator working when the `azure-storage-blob`
client library is newer than the Azurite image.)

### 2. Configure environment variables

Create a `.env` file in this directory (it is gitignored):

```env
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
AZURE_STORAGE_CONNECTION_STRING=UseDevelopmentStorage=true
```

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required. Your Anthropic API key. |
| `AZURE_STORAGE_CONNECTION_STRING` | Authenticate with a connection string (e.g. Azurite). Takes precedence. |
| `AZURE_STORAGE_ACCOUNT_URL` | Authenticate against a live account with [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview). |

Set `AZURE_STORAGE_CONNECTION_STRING` **or** `AZURE_STORAGE_ACCOUNT_URL`. With an
account URL, credentials come from `DefaultAzureCredential` (e.g. `az login`, managed
identity, or environment variables); to use a different credential type, pass
`credential=` to `AzureBlobBackend` directly.

## Running the samples

Each sample uses [PEP 723 inline script metadata](https://peps.python.org/pep-0723/), so
uv installs the dependencies automatically — no separate install step. The scripts pin
`langchain-azure-storage` to this repository checkout via `[tool.uv.sources]`, so they
run against your local code; if you copy a sample elsewhere, delete that block to use
the released package instead.

### Basic agent

A minimal Deep Agent whose workspace persists in Azure Blob Storage. After the run it
lists the workspace and prints the blob URL each file landed at.

```bash
cd samples
uv run --env-file .env basic_agent.py
```

### Resuming a workspace ([resume_workspace.py](resume_workspace.py))

The demo only a durable backend can run: one agent writes research notes and is torn
down completely; a brand-new backend and agent then attach to the same prefix and
summarize what they find. State survives because it lives in Blob Storage, not in
process memory.

```bash
cd samples
uv run --env-file .env resume_workspace.py
```

### Composite agent with memory and subagents

Deep Agents features — an `AGENTS.md` memory file and delegation to specialized coder
and tester subagents — running on one shared, durable workspace: the coder's files are
immediately visible to the tester.

```bash
cd samples
uv run --env-file .env composite_with_memories.py
```

## Browsing the results

To inspect the blobs the agents create, use
[Azure Storage Explorer](https://azure.microsoft.com/products/storage/storage-explorer):
connect to the **Local storage emulator** with the default settings and browse the
`agent-workspace` container under **devstoreaccount1 > Blob Containers**.

## Running against a live storage account

Replace the connection string in your `.env` with an account URL:

```env
AZURE_STORAGE_ACCOUNT_URL=https://<your-account>.blob.core.windows.net
```

and make sure `DefaultAzureCredential` can authenticate (e.g. `az login`).
