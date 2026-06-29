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

The backend is exposed at the top level of the package:

```python
from langchain_azure_storage import AzureBlobBackend, AzureBlobConfig
```

- `AzureBlobConfig` is a dataclass holding connection details (`account_url`,
  `container_name`, `prefix`, credential fields, `max_concurrency`, `encoding`,
  `api_version`). Its `__post_init__` validates that at most one explicit credential source
  is configured (see [Authentication](#authentication)).
- `AzureBlobBackend(config)` implements `BackendProtocol`. It is async-first
  (`aread`/`awrite`/`aedit`/`als`/`aglob`/`agrep`/`aupload_files`/`adownload_files`); each
  has a synchronous wrapper of the same name without the leading `a`.

### Storage model

- File content is stored as raw text (UTF-8 by default) in the blob body.
- `created_at` and `modified_at` timestamps are stored in blob metadata as ISO 8601 strings.
- Directories are **synthesized** from blob key prefixes; no directory marker blobs are
  written. `ls` derives immediate child directories from the keys it lists.
- A configurable `prefix` namespaces all keys within the container, enabling isolation of
  multiple agents/sessions in a single container.
- `glob`/`grep` use [`wcmatch`][wcmatch] for `**` (globstar) and `{a,b}` brace expansion,
  and bound concurrency with a semaphore (`max_concurrency`). `grep` is a literal substring
  search across blob contents.

### Async/sync bridging

The Azure async SDK clients are loop-affine. The sync wrappers run the coroutine in a fresh
event loop and use a context-local pool of temporary clients (rather than mutating the
instance-level async cache) so that sync calls are safe to invoke from worker threads after
the instance client has already been initialised in another loop. User-supplied credentials
are treated as caller-owned and are never closed by the backend; credentials the backend
creates itself (e.g. `DefaultAzureCredential`) are closed on cleanup.

### Authentication

`AzureBlobConfig` supports five mutually exclusive methods, defaulting to
`DefaultAzureCredential`: Microsoft Entra ID (default), connection string, account key, SAS
token, and an arbitrary Azure credential object. This mirrors the credential model of the
existing document loader.

## Packaging

`deepagents` pulls in a large dependency tree (`langchain`, `langchain-anthropic`,
`langchain-google-genai`) and requires Python 3.11+, whereas `langchain-azure-storage`
supports Python 3.10+. To avoid forcing that footprint (or a Python-version bump) on
document-loader users, the backend's dependencies live behind an optional extra gated by an
environment marker:

```toml
[project.optional-dependencies]
deepagents = [
    "deepagents>=0.6.1; python_version>='3.11'",
    "wcmatch>=8.0; python_version>='3.11'",
]
```

Install with `pip install "langchain-azure-storage[deepagents]"`. The implementation lives in
the private `langchain_azure_storage._deepagents` subpackage and is re-exported lazily from
the top-level `__init__` (PEP 562 `__getattr__`), so importing `langchain_azure_storage`
remains cheap and dependency-free; accessing `AzureBlobBackend`/`AzureBlobConfig` without the
extra installed raises an `ImportError` directing the user to install it.

## Testing

- **Unit tests** mock the Azure SDK and cover path normalization, config validation,
  credential-client creation, the async/sync bridging, and each operation.
- **Integration and contract tests** run against the [Azurite][azurite] emulator. CI starts
  Azurite in a container scoped to the storage test job and runs them on Python 3.12 (where
  the extra is installed); on Python 3.10 they are skipped via `collect_ignore_glob`.

[deepagents]: https://github.com/langchain-ai/deepagents
[community-pkg]: https://github.com/oddrationale/deepagents-azure-blob-backend
[langchain-azure-storage-pkg]: https://pypi.org/project/langchain-azure-storage/
[wcmatch]: https://facelessuser.github.io/wcmatch/
[azurite]: https://learn.microsoft.com/azure/storage/common/storage-use-azurite
