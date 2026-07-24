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

`AzureBlobBackend` implements `BackendProtocol`. Its constructor takes parameters mirroring
`AzureBlobStorageLoader` (no separate config object); a connection string is supplied via a
separate classmethod, matching the shape of the Azure SDK's own `from_connection_string`:

```python
AzureBlobBackend(
    account_url: str,
    container_name: str,
    *,
    prefix: str | None = None,
    credential: AzureSasCredential | TokenCredential | AsyncTokenCredential | None = None,
)

AzureBlobBackend.from_connection_string(
    connection_string: str,
    container_name: str,
    *,
    prefix: str | None = None,
) -> AzureBlobBackend
```

`account_url` and `container_name` have no defaults, so it's an immediate `TypeError` if a
caller forgets one rather than a confusing failure once a request is made.

It exposes the synchronous `BackendProtocol` methods (`read`/`write`/`edit`/`ls`/`glob`/
`grep`/`upload_files`/`download_files`) and their `a`-prefixed async counterparts, plus
`close()`/`aclose()` (and context-manager support) to release the resources described below.

### Storage model

- File content is stored as UTF-8 text in the blob body (binary uploads are preserved as
  raw bytes).
- `FileInfo.modified_at` comes from the blob's native `last_modified` property; no
  backend-managed metadata is written.
- Directories are **synthesized** from blob key prefixes; no directory marker blobs are
  written. `ls` derives immediate child directories from the keys it lists.
- A configurable `prefix` namespaces all keys within the container, enabling isolation of
  multiple agents/sessions in a single container.
- `glob`/`grep` use [`wcmatch`][wcmatch] for `**` (globstar) and `{a,b}` brace expansion.
  `grep` is a literal substring search across blob contents, matching ripgrep's `--glob`
  semantics (a slash-less pattern matches file names at any depth).
- `read`/`aread` return raw content; the Deep Agents middleware applies line numbering, the
  empty-file reminder, and base64 multimodal rendering. Blobs that are not valid UTF-8 are
  returned base64-encoded with `encoding="base64"`.
- `edit`/`aedit` upload conditioned on the blob's ETag (`MatchConditions.IfNotModified`), so
  a concurrent editor gets a recoverable `EditResult` error instead of silently clobbering the
  other edit; `write`/`awrite` are already atomic (`overwrite=False`). Batch overwrites via
  `upload_files` are last-write-wins, matching every other Deep Agents backend.

### Sync and async clients

Synchronous methods use the synchronous `azure.storage.blob` client and asynchronous methods
use the `azure.storage.blob.aio` client, mirroring the document loader (rather than driving an
async client from a synchronous wrapper). Unlike the document loader, the backend **caches**
its container client and any credential it creates, building them lazily on first use and
reusing them across every subsequent call (guarded by a `threading.Lock`/`asyncio.Lock` so
concurrent first calls don't race). This matters here in a way it doesn't for the loader:
`load_documents()` is typically called once per loader instance, while an agent may call a
backend method many times over its lifetime, so paying the client/credential construction cost
(and losing HTTP connection pooling and credential token caching) on every call would be
wasteful. `close()`/`aclose()` release the cached client and any credential the backend
created itself; a caller-supplied credential is never closed by the backend.

### Error handling

Backend methods only catch exceptions they can translate into a meaningful `Result.error`
(e.g. `ResourceNotFoundError`, `ResourceModifiedError`, path validation) — matching
`FilesystemBackend`'s own pattern of catching `OSError` for its per-file operations and only
catching broad `Exception` in its batch upload/download methods. An infrastructure failure a
backend can't meaningfully translate (a network error, an unexpected `AzureError`) is left to
propagate, since the middleware calls backend methods directly with no surrounding
try/except — swallowing it into a generic string would hide the real cause from both logs and
the agent. Where the protocol defines standardized `FileOperationError` codes
(`FILE_NOT_FOUND`, `INVALID_PATH`, `PERMISSION_DENIED`) for the batch upload/download
responses, the backend uses those constants; `upload_files`/`download_files` fall back to the
exception's own message for failures that don't fit one of those codes, rather than a
non-actionable generic string.

### Authentication

Like the document loader, authentication defaults to `DefaultAzureCredential` and accepts a
`credential` override (any Azure SAS, token, or async-token credential). Credential validity
is delegated to the Azure SDK. `from_connection_string` may be used instead of `account_url` +
`credential`, primarily for local development against Azurite.

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
- **Integration tests** run against either a live storage account
  (`AZURE_STORAGE_ACCOUNT_URL`) or the [Azurite][azurite] emulator
  (`AZURE_STORAGE_CONNECTION_STRING`) via `make integration_tests`, locally for now (as with
  the document loaders); they are skipped on Python 3.10 via `pytest.importorskip`.
- **Contract tests** are a fork of `langchain-tests`' `SandboxIntegrationTests` trimmed to
  the `BackendProtocol` surface. Once a shared `BackendProtocol` suite lands
  ([langchain-ai/langchain#37905][backend-tests-issue]), the fork should be deleted in favor
  of subclassing the shared suite.

## Future ideas & deferred work

Consolidated from the [#783][pr-783] review. Items with a tracking issue link to it.

### Follow-up features

- **Runtime-scoped workspaces.** Accept a callable for `prefix` (and later
  `container_name`) that receives the LangGraph runtime, mirroring how `StoreBackend`
  supports [user- and agent-scoped memory][scoped-memory]. A deployment serving many
  users could then isolate each user's (or each agent's) files at runtime — e.g.
  `prefix=lambda runtime: f"{runtime.context['user_id']}/"` — without constructing a
  backend per request. Starting with `prefix` keeps it cheap (pure key namespacing);
  extending to `container_name` would add hard isolation boundaries (per-container
  access policies, lifecycle rules).
- **Azurite-backed CI.** Wire the integration and contract suites into the shared CI
  workflow with an Azurite service container, for both this backend and the document
  loaders. Deferred from #783 because `_test.yml` is shared across the Azure packages
  and the approach should be agreed with the other package owners first. Tracked in
  [langchain-azure#816][azurite-ci-issue].
- **Chunked/paginated reads for large files.** `read`/`aread` currently download the whole
  blob and slice it in-memory to the requested `offset`/`limit` window, since Blob Storage's
  ranged GETs are byte-indexed and our offsets are line-indexed. For the common case (a
  single read within the default 2000-line limit) this costs one round trip either way. It's
  only a real cost for the less common case of an agent paginating through a very large file
  section by section, where each page re-downloads the full blob. A byte-position index
  keyed by `(blob path, etag)` built up incrementally as pages are read (invalidated on any
  write) would let subsequent pages seek via a ranged GET instead of a full re-download.
  Worth revisiting if this shows up as a real cost in practice, e.g. via profiling.

### Version-triggered maintenance

- **Python 3.11 floor.** Bump `requires-python` to `>=3.11` at Python 3.10 EOL
  (October 2026) or when `langchain` core drops 3.10, whichever comes first. Removes the
  `python_version >= "3.11"` environment marker on the `deepagents` extra, so an
  unsupported install fails at install time instead of import time. The `ImportError`
  shim stays (it guards a missing extra on any Python version). Tracked in
  [langchain-azure#815][py310-issue].
- **deepagents 0.7.0 re-sync.** When a stable 0.7.0 lands (and the floor bumps for other
  reasons), switch read-encoding classification to `_get_backend_read_file_type` (which
  adds e.g. `.mkv` handling) and refresh the vendored `_NON_TEXT_EXTENSIONS` set in
  `_utils.py`. The parity test in `tests/unit_tests/deepagents/test_utils.py` fails on
  any drift between the vendored set and the installed deepagents, so this cannot be
  silently missed.
- **Shared `BackendProtocol` contract suite.** Once `langchain-tests` ships a shared
  suite ([langchain-ai/langchain#37905][backend-tests-issue]), delete our fork of
  `SandboxIntegrationTests` and subclass the shared suite instead (see Testing above).
- **Community package hand-off.** The backend was ported from the community
  [`deepagents-azure-blob-backend`][community-pkg] package (PyPI, currently 0.4.1). Once
  the first `langchain-azure-storage` release containing the backend ships: publish a
  final community release whose README and an import-time `DeprecationWarning` point
  here, including a short old-to-new migration note (the constructor API changed during
  review, so a re-export shim would not be compatible); then mark the PyPI project
  archived and archive the GitHub repo. The PyPI project is never deleted — existing
  pins must keep resolving, and deletion would free the name for reuse.

### Upstream (deepagents)

- **Coded error constants in single-object results.** The reference backends use
  `FILE_NOT_FOUND`/`INVALID_PATH`/`PERMISSION_DENIED` only in the batch
  `FileUploadResponse`/`FileDownloadResponse` results and prose messages in
  `ReadResult`/`EditResult`/`GrepResult`; we follow the same pattern. Worth asking
  upstream whether that split is intentional (codes for programmatic batch consumers,
  prose for the LLM-facing tool boundary) and whether the constants should extend to
  single-object results. No issue filed yet.
- **Content-type-aware encoding classification.** Classification is extension-only
  (`_get_file_type`); no backend consults MIME metadata, yet object-store backends have
  it available (e.g. Blob Storage's `content_settings.content_type`), which would cover
  extensionless or mis-named blobs. Worth proposing upstream if demand appears. No issue
  filed yet.
- **`glob()` recursivity inconsistency.** `FilesystemBackend` matches `*.py` recursively
  while `StateBackend`/`StoreBackend` (and this backend) follow the non-recursive
  `BackendProtocol.glob` docstring ([deepagents#4978][upstream-glob-issue]). If upstream
  resolves in favor of the recursive behavior, our `glob()` should follow.

### On request

- **Dedicated `account_key`/SAS parameters and an `encoding` knob.** Deliberately not
  exposed to keep the surface OAuth-first and minimal; add if users ask. SAS already
  works today via `credential=AzureSasCredential(...)`.

[deepagents]: https://github.com/langchain-ai/deepagents
[community-pkg]: https://github.com/oddrationale/deepagents-azure-blob-backend
[langchain-azure-storage-pkg]: https://pypi.org/project/langchain-azure-storage/
[wcmatch]: https://facelessuser.github.io/wcmatch/
[azurite]: https://learn.microsoft.com/azure/storage/common/storage-use-azurite
[backend-tests-issue]: https://github.com/langchain-ai/langchain/issues/37905
[pr-783]: https://github.com/langchain-ai/langchain-azure/pull/783
[azurite-ci-issue]: https://github.com/langchain-ai/langchain-azure/issues/816
[py310-issue]: https://github.com/langchain-ai/langchain-azure/issues/815
[upstream-glob-issue]: https://github.com/langchain-ai/deepagents/issues/4978
[scoped-memory]: https://docs.langchain.com/oss/python/deepagents/memory#scoped-memory
