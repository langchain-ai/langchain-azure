# `langchain-azure-storage` (`libs/azure-storage`)

Azure Blob Storage document loaders and a Deep Agents filesystem backend.
The package is in **public preview**: its README says interfaces may change.
Preview status is not a licence to break users silently — a breaking change
still needs to be called out in the PR description and README.

## Document loaders

- `lazy_load()` is the real implementation; `load()` builds on it. Both must
  stream: materializing every blob in a container into memory defeats the
  loader's purpose and fails on realistic data. A change that collects results
  into a list before yielding is a design finding.
- Blobs are downloaded to temporary files for parsing. Every path out of that
  code — success, parse failure, cancellation — must delete the temporary file.
  Missing cleanup on the exception path is the common defect.
- Prefix, name, and container filtering must not accidentally widen scope, and
  results should stay deterministic in ordering where the service allows it.
- Metadata attached to each `Document` (source URL, blob name, container,
  metadata properties) is a contract that downstream retrieval code depends on.
- Credential handling follows [azure-sdk-contracts.md](azure-sdk-contracts.md):
  `TokenCredential` support, no SAS token or account key in logs, exception
  messages, or `Document.metadata`.

## Deep Agents blob backend

`AzureBlobBackend` presents a flat blob namespace as a filesystem, so the
mapping itself is where bugs live:

- Path-to-blob-name translation must be consistent in both directions and must
  not let a `..` segment or an absolute path escape the configured prefix.
- Directory semantics are simulated. Listing, existence checks, and recursive
  operations must agree on whether a prefix with no blob of its own is a
  directory.
- Concurrency is optimistic: ETag / `If-Match` conditions are how lost updates
  are prevented. A write that drops the ETag condition introduces a silent
  last-writer-wins data-loss window, which is a High-severity finding.
- Batch operations must report **per-item** outcomes. Failing the whole batch
  because one blob failed, or reporting success because most succeeded, both
  lose information the agent needs.
- The protocol rules in [ecosystem-contracts.md](ecosystem-contracts.md) apply:
  errors in the result `error` field rather than raised, the fixed
  `FileOperationError` literals, input-ordered results, and honest `truncated`
  reporting.

## User agent

`_user_agent.py` stamps the partner user agent on constructed clients. Any new
client construction path must go through it.
