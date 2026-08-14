# langchain-azure-storage

This package contains the LangChain integrations for [Azure Storage](https://learn.microsoft.com/en-us/azure/storage/common/storage-introduction). Currently, it includes:
- [Document loader support for Azure Blob Storage](#azure-blob-storage-document-loader-usage)
- [Deep Agents filesystem backend backed by Azure Blob Storage](#deep-agents-azure-blob-storage-backend-usage)

> [!NOTE]
> This package is in Public Preview. For more information, see [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/support/legal/preview-supplemental-terms/).

## Installation

```bash
pip install -U langchain-azure-storage
```

## Configuration
`langchain-azure-storage` should work without any explicit credential configuration.

The `langchain-azure-storage` interface defaults to [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview)
for credentials which automatically retrieves [Microsoft Entra ID tokens](https://learn.microsoft.com/en-us/azure/storage/blobs/authorize-access-azure-active-directory) based on
your current environment. For more information on using credentials with
`langchain-azure-storage`, see the [override default credentials](#override-default-credentials) section.

## Azure Blob Storage Document Loader Usage
[Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/) are used to load data from many sources (e.g., cloud storage, web pages, etc.) and turn them into [LangChain Documents](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html), which can then be used in AI applications (e.g., [RAG](https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain)). This package offers the `AzureBlobStorageLoader` which downloads blob content from Azure Blob Storage and parses it as UTF-8 by default. Additionally, [parsing customization](#customizing-blob-content-parsing) is also available to handle content of various file types and customize document chunking.  

The `AzureBlobStorageLoader` replaces the current `AzureBlobStorageContainerLoader` and `AzureBlobStorageFileLoader` in the [LangChain Community Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/). Refer to the [migration section](#migrating-from-langchain-community-azure-storage-document-loaders) for more details. 

The following examples go over the various use cases for the document loader.

### Load from container
Below shows how to load documents from all blobs in a given container in Azure Blob Storage:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
)

for doc in loader.lazy_load():
    print(doc.page_content)  # Prints content of each blob in UTF-8 encoding.
```

The example below shows how to load documents from blobs in a container with a given prefix:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
    prefix="test",
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

### Load from container by blob name
The example below shows how to load documents from a list of blobs in Azure Blob Storage. This approach does not call list blobs and instead uses only the blobs provided:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
    blob_names=["blob-1", "blob-2", "blob-3"],
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

### Override default credentials
Below shows how to override the default credentials used by the document loader:

```python
from azure.core.credentials import AzureSasCredential
from azure.identity import ManagedIdentityCredential
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

# Override with SAS token
loader = AzureBlobStorageLoader(
    "https://<my-storage-account-name>.blob.core.windows.net",
    "<my-container-name>",
    credential=AzureSasCredential("<sas-token>")
)

# Override with more specific token credential than the entire
# default credential chain (e.g., system-assigned managed identity)
loader = AzureBlobStorageLoader(
    "https://<my-storage-account-name>.blob.core.windows.net",
    "<my-container-name>",
    credential=ManagedIdentityCredential()
)
```

### Customizing blob content parsing
Currently, the default when parsing each blob is to return the content as a single `Document` object with UTF-8 encoding regardless of the file type. For file types that require specific parsing (e.g., PDFs, CSVs, etc.) or when you want to control the document content format, you can provide the `loader_factory` argument to take in an already existing document loader (e.g., PyPDFLoader, CSVLoader, etc.) or a customized loader.

This works by downloading the blob content to a temporary file. The `loader_factory` then gets called with the filepath to use the specified document loader to load/parse the file and return the `Document` object(s).

Below shows how to override the default loader used to parse blobs as PDFs using the using the [PyPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html#pypdfloader):

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_community.document_loaders import PyPDFLoader

loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
    blob_names="<my-pdf-file.pdf>",
    loader_factory=PyPDFLoader,
)

for doc in loader.lazy_load():
    print(doc.page_content)  # Prints content of each page as a separate document
```

To provide additional configuration, you can define a callable that returns an instantiated document loader as shown below:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_community.document_loaders import PyPDFLoader

def loader_factory(file_path: str) -> PyPDFLoader:
    return PyPDFLoader(
        file_path,
        mode="single",  # To return the PDF as a single document instead of extracting documents by page
    )

loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
    blob_names="<my-pdf-file.pdf>",
    loader_factory=loader_factory,
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

### Migrating from LangChain Community Azure Storage Document Loaders
This section goes over the actions required to migrate from the existing community document loaders to the new Azure Blob Storage document loader:

1. Depend on the `langchain-azure-storage` package instead of `langchain-community`.
2. Update import statements from `langchain_community.document_loaders` to
   `langchain_azure_storage.document_loaders`.
3. Change class names from `AzureBlobStorageFileLoader` and `AzureBlobStorageContainerLoader`
   to `AzureBlobStorageLoader`.
4. Update document loader constructor calls to:
    1. Use an account URL instead of a connection string.
    2. Specify `UnstructuredLoader` as the `loader_factory` if they want to continue to use Unstructured for parsing documents.
5. Ensure environment has proper credentials (e.g., running `azure login` command, setting up managed identity, etc.) as the connection string would have previously contained the credentials.

The examples below show the before and after migrating to the `langchain-azure-storage package`:

#### Before migration
```python
from langchain_community.document_loaders import AzureBlobStorageFileLoader, AzureBlobStorageContainerLoader

file_loader = AzureBlobStorageFileLoader(
    conn_str="<my-connection-string>",
    container="<my-container-name>",
    blob_name="<my-blob-name>",
)

container_loader = AzureBlobStorageContainerLoader(
    conn_str="<my-connection-string>",
    container="<my-container-name>",
    prefix="<prefix>",
)
```

#### After migration
```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_unstructured import UnstructuredLoader

file_loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
    blob_names="<my-blob-name>",
)

container_loader = AzureBlobStorageLoader(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="<my-container-name>",
    prefix="<prefix>",
    loader_factory=UnstructuredLoader,
)
```

## Deep Agents Azure Blob Storage Backend Usage

[Deep Agents](https://github.com/langchain-ai/deepagents) exposes a `BackendProtocol` — a pluggable interface for file operations (`read`, `write`, `edit`, `delete`, `ls`, `glob`, `grep`, plus batch upload/download) that an agent uses as its virtual filesystem. This package provides `AzureBlobBackend`, an Azure Blob Storage implementation of that interface, so a deep agent can persist its workspace in a blob container.

The backend requires the optional `deepagents` extra (which itself requires Python 3.11+):

```bash
pip install -U "langchain-azure-storage[deepagents]"
```

`AzureBlobBackend` is imported from the `deepagents` subpackage:

```python
from langchain_azure_storage.deepagents import AzureBlobBackend
```

> [!NOTE]
> Importing it without the `deepagents` extra installed raises an `ImportError` directing you to install the extra. The document loader does not require the extra.

### Quick start

```python
from deepagents import create_deep_agent
from langchain_azure_storage.deepagents import AzureBlobBackend

backend = AzureBlobBackend(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="agent-workspace",
    prefix="session-001/",  # Optional: isolate each agent/session under a prefix.
)

agent = create_deep_agent(backend=backend)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Write a hello world script to hello.py"}]}
)
# The agent's write_file tool call persists the script through the backend. With
# the configuration above, it lands in the "agent-workspace" container at
# https://<my-storage-account-name>.blob.core.windows.net/agent-workspace/session-001/hello.py
```

Runnable examples — including a workspace that persists across agent lifetimes and a
composite agent with memory and subagents — live in
[`samples/deepagents-storage-backend/`](../../samples/deepagents-storage-backend/README.md).

File content is stored as UTF-8 text in blob bodies (binary uploads are preserved as bytes). Directories are synthesized from blob key prefixes (no directory marker blobs are created). The backend exposes both synchronous methods (`read`, `write`, `edit`, `delete`, `ls`, `glob`, `grep`, `upload_files`, `download_files`) and their `a`-prefixed async counterparts (`aread`, `awrite`, …).

### Writes and deletes are destructive

Two backend operations replace or remove data that is already in your container. Both follow the Deep Agents `BackendProtocol` contract, and both are driven by the agent:

- **`write` replaces an existing file in full.** There is no create-only mode — `write_file` on a path that already exists overwrites it rather than erroring. Use `edit` when existing content must be preserved.
- **`delete` is recursive.** Deleting a directory removes it and everything nested under it. Deleting `"/"` removes every blob in the configured `prefix` namespace, or — if no `prefix` is set — **every blob in the container**.

Recommended mitigations:

- Set a `prefix` so the agent can only reach its own key namespace, never the whole container.
- Enable [soft delete](https://learn.microsoft.com/azure/storage/blobs/soft-delete-blob-overview) and/or [blob versioning](https://learn.microsoft.com/azure/storage/blobs/versioning-overview) on the container so destructive tool calls are recoverable.
- Scope the agent's credential to the container it should be able to modify, following the [Azure RBAC best practices](https://learn.microsoft.com/azure/role-based-access-control/best-practices): assign the role at the container scope rather than the subscription or storage account, and pick the least-privileged built-in role for what the agent actually does — `Storage Blob Data Reader` for a read-only agent, `Storage Blob Data Contributor` only when it must write or delete.
- Drop or gate the tools you don't want. Omit `delete` from the filesystem middleware, or add a Deep Agents permission rule that denies it or routes it through a human-approval interrupt:

  ```python
  from deepagents import FilesystemMiddleware, create_deep_agent

  agent = create_deep_agent(
      backend=backend,
      middleware=[
          # Register every filesystem tool except `delete`.
          FilesystemMiddleware(
              backend=backend,
              tools=["ls", "read_file", "write_file", "edit_file", "glob", "grep"],
          )
      ],
  )
  ```

### Authentication

Like the document loader, `AzureBlobBackend` defaults to [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview) and accepts a `credential` override:

```python
from azure.identity import ManagedIdentityCredential

backend = AzureBlobBackend(
    account_url="https://<account>.blob.core.windows.net",
    container_name="agent-workspace",
    credential=ManagedIdentityCredential(),  # or any Azure credential object
)
```

For local development against the [Azurite](https://learn.microsoft.com/azure/storage/common/storage-use-azurite) emulator, use `from_connection_string` instead of `account_url` + `credential`:

```python
backend = AzureBlobBackend.from_connection_string(
    "<connection-string>",
    container_name="agent-workspace",
)
```

### Resource lifecycle

`AzureBlobBackend` creates its underlying Azure SDK client (and, unless you pass a `credential`, a `DefaultAzureCredential`) lazily on first use and reuses it across calls.

When you use the **async** methods (`aread`, `awrite`, …), close the backend when you're done so the underlying `aiohttp` session is released; otherwise you'll see `Unclosed client session` warnings. Use it as an async context manager, or call `aclose()`:

```python
async with AzureBlobBackend(account_url="...", container_name="agent-workspace") as backend:
    agent = create_deep_agent(backend=backend)
    ...
# equivalently: await backend.aclose() when you're done
```

The **sync** client releases its resources on garbage collection, so closing it is optional; you can still use `with` (or call `close()`) to release it promptly.

## Changelog