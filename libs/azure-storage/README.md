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

## Deep Agents Azure Blob Storage Backend Usage

[Deep Agents](https://github.com/langchain-ai/deepagents) exposes a `BackendProtocol` — a pluggable interface for file operations (`read`, `write`, `edit`, `ls`, `glob`, `grep`, plus batch upload/download) that an agent uses as its virtual filesystem. This package provides `AzureBlobBackend`, an Azure Blob Storage implementation of that interface, so a deep agent can persist its workspace in a blob container.

The backend requires the optional `deepagents` extra (which itself requires Python 3.11+):

```bash
pip install -U "langchain-azure-storage[deepagents]"
```

`AzureBlobBackend` and its configuration object `AzureBlobConfig` are exposed at the top level of the package:

```python
from langchain_azure_storage import AzureBlobBackend, AzureBlobConfig
```

> [!NOTE]
> Importing these names without the `deepagents` extra installed raises an `ImportError` directing you to install the extra. The document loader does not require the extra.

### Quick start

```python
from deepagents import create_deep_agent
from langchain_azure_storage import AzureBlobBackend, AzureBlobConfig

config = AzureBlobConfig(
    account_url="https://<my-storage-account-name>.blob.core.windows.net",
    container_name="agent-workspace",
    prefix="session-001/",  # Optional: isolate each agent/session under a prefix.
)
backend = AzureBlobBackend(config)

agent = create_deep_agent(backend=backend)
```

File content is stored as UTF-8 text in blob bodies, with `created_at`/`modified_at` timestamps kept in blob metadata. Directories are synthesized from blob key prefixes (no directory marker blobs are created). The backend is async-first (`aread`, `awrite`, `aedit`, `als`, `aglob`, `agrep`, `aupload_files`, `adownload_files`), and each method has a synchronous wrapper of the same name without the leading `a`.

### Authentication

Like the document loader, `AzureBlobConfig` defaults to [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview). It also supports four other mutually exclusive authentication methods:

```python
# 1. Default (Microsoft Entra ID via DefaultAzureCredential) — account_url only
AzureBlobConfig(account_url="https://<account>.blob.core.windows.net")

# 2. Connection string (e.g., from the Azure portal, or Azurite for local dev)
AzureBlobConfig(connection_string="<connection-string>")

# 3. Account key
AzureBlobConfig(account_url="https://<account>.blob.core.windows.net", account_key="<key>")

# 4. SAS token (a leading "?" is accepted and stripped)
AzureBlobConfig(account_url="https://<account>.blob.core.windows.net", sas_token="<sas-token>")

# 5. Any Azure credential object (e.g., a more specific managed identity)
from azure.identity.aio import ManagedIdentityCredential
AzureBlobConfig(
    account_url="https://<account>.blob.core.windows.net",
    credential=ManagedIdentityCredential(),
)
```

Setting more than one credential source raises `ValueError`. `connection_string` is self-contained and must not be combined with `account_url`.

## Migrating from LangChain Community Azure Storage Document Loaders
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

## Changelog