# Azure Blob Storage backend for Deep Agents

| Proposal    | Metadata                          |
|-------------|-----------------------------------|
| **Author**  | Dariel Dato-on (@darieldatoon)    |
| **Status**  | Proposed                          |
| **Created** | 28-June-2026                      |

## Abstract

This proposal outlines the design and implementation of `AzureBlobBackend`, an
Azure Blob Storage implementation of the [Deep Agents][deepagents] `BackendProtocol`.
It is part of the [`langchain-azure-storage`][langchain-azure-storage-pkg] package and
provides first-party support for using Azure Blob Storage as the virtual filesystem of a
deep agent. The implementation originated as the community package
[`deepagents-azure-blob-backend`][community-pkg] and is being incorporated into
`langchain-azure-storage` so that it is maintained alongside the rest of the Azure Storage
integrations.

## Background and motivation

### What is a Deep Agents backend?

[Deep Agents][deepagents] is a framework for building long-horizon agents. It exposes a
`BackendProtocol`: a pluggable interface for file operations (`read`, `write`, `edit`,
`ls`, `glob`, `grep`, plus batch `upload_files`/`download_files`) that the agent uses as a
virtual filesystem for scratch space, intermediate artifacts, and durable memory. The
default backend stores files in agent state (in memory). Swapping in a persistent backend
lets an agent's workspace survive across runs and be shared or inspected out of band.

### Why Azure Blob Storage?

Teams running agents on Azure need their agent filesystem to live in Azure-native storage
for the same durability, access-control, and compliance reasons that drive other workloads
onto Azure Blob Storage. Before this integration there was no first-party Azure backend for
Deep Agents; the community `deepagents-azure-blob-backend` package filled the gap. Folding
it into `langchain-azure-storage` gives it the same release process, CI, and Azure Storage
team review as the document loaders, and exposes it under a single, discoverable package.

## Design

### Public API

The backend is imported from the `deepagents` subpackage:

```python
from langchain_azure_storage.deepagents import AzureBlobBackend
```

`AzureBlobBackend` implements `BackendProtocol` and takes constructor parameters mirroring
`AzureBlobStorageLoader` (no separate config object):

```python
AzureBlobBackend(
    account_url: str = "",
    container_name: str = "",
    *,
    prefix: str | None = None,
    credential: AzureSasCredential | TokenCredential | AsyncTokenCredential | None = None,
    connection_string: str | None = None,
)
```

It exposes the synchronous `BackendProtocol` methods (`read`/`write`/`edit`/`ls`/`glob`/
`grep`/`upload_files`/`download_files`) and their `a`-prefixed async counterparts.

### Storage model

- File content is stored as UTF-8 text in the blob body (binary uploads are preserved as
  raw bytes).
- `created_at` and `modified_at` timestamps are stored in blob metadata as ISO 8601 strings.
- Directories are **synthesized** from blob key prefixes; no directory marker blobs are
  written. `ls` derives immediate child directories from the keys it lists.
- A configurable `prefix` namespaces all keys within the container, enabling isolation of
  multiple agents/sessions in a single container.
- `glob`/`grep` use [`wcmatch`][wcmatch] for `**` (globstar) and `{a,b}` brace expansion.
  `grep` is a literal substring search across blob contents (async `agrep` bounds
  concurrency with a semaphore).
- `read`/`aread` return raw content; the Deep Agents middleware applies line numbering, the
  empty-file reminder, and base64 multimodal rendering. Blobs that are not valid UTF-8 are
  returned base64-encoded with `encoding="base64"`.

### Sync and async clients

Synchronous methods use the synchronous `azure.storage.blob` client and asynchronous methods
use the `azure.storage.blob.aio` client, mirroring the document loader (rather than driving an
async client from a synchronous wrapper). Clients are context-managed per operation;
credential resolution reuses the loader's `_get_sync_credential` / `_get_async_credential`
helpers. A caller-supplied credential is caller-owned and is never closed by the backend;
a `DefaultAzureCredential` the backend creates itself is closed after use.

### Authentication

Like the document loader, authentication defaults to `DefaultAzureCredential` and accepts a
`credential` override (any Azure SAS, token, or async-token credential). Credential validity
is delegated to the Azure SDK. A `connection_string` may be supplied instead of
`account_url` + `credential`, primarily for local development against Azurite.

## Packaging

`deepagents` pulls in a large dependency tree (`langchain`, `langchain-anthropic`,
`langchain-google-genai`) and requires Python 3.11+, whereas `langchain-azure-storage`
supports Python 3.10+. To avoid forcing that footprint (or a Python-version bump) on
document-loader users, the backend's dependencies live behind an optional extra gated by an
environment marker:

```toml
[project.optional-dependencies]
deepagents = [
    "deepagents>=0.6.12,<1; python_version>='3.11'",
    "wcmatch>=10.1; python_version>='3.11'",
]
```

Install with `pip install "langchain-azure-storage[deepagents]"`. The backend lives in the
`langchain_azure_storage.deepagents` subpackage; importing it without the extra installed (or
on Python 3.10) raises an `ImportError` directing the user to install it. The top-level
`langchain_azure_storage` package stays importable and dependency-free for document-loader
users.

> The `python_version>='3.11'` marker is required: `deepagents` needs Python >= 3.11, and
> without the marker `uv` cannot resolve a universal lockfile that also spans the package's
> 3.10 support. The marker keeps the lock resolvable while the subpackage import surfaces the
> requirement clearly at first use.

## Testing

- **Unit tests** mock the sync and async Azure SDK clients and cover path normalization,
  credential resolution, and each operation (sync + async).
- **Integration and contract tests** run against the [Azurite][azurite] emulator via
  `make integration_tests`, locally for now (as with the document loaders); they are skipped
  on Python 3.10 via `collect_ignore_glob`.

[deepagents]: https://github.com/langchain-ai/deepagents
[community-pkg]: https://github.com/oddrationale/deepagents-azure-blob-backend
[langchain-azure-storage-pkg]: https://pypi.org/project/langchain-azure-storage/
[wcmatch]: https://facelessuser.github.io/wcmatch/
[azurite]: https://learn.microsoft.com/azure/storage/common/storage-use-azurite
