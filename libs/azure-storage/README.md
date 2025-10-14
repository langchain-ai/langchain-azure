# langchain-azure-storage

This package contains the LangChain integrations for Azure Storage. Currently, it includes:
- `AzureBlobStorageLoader` - Loads LangChain document objects from Azure Blob Storage

## Installation

```bash
pip install -U langchain-azure-storage
```

## Migration Details
To migrate from the existing community document loaders to the new Azure Storage document loader, you will have to make the following changes:
1. Depend on the `langchain-azure-storage` package instead of `langchain-community`.
2. Update import statements from `langchain_community.document_loaders` to
   `langchain_azure_storage.document_loaders`.
3. Change class names from `AzureBlobStorageFileLoader` and `AzureBlobStorageContainerLoader`
   to `AzureBlobStorageLoader`.
4. Update document loader constructor calls to:
    1. Use an account URL instead of a connection string.
    2. Specify `UnstructuredLoader` as the `loader_factory` if they want to continue to use Unstructured for parsing documents.
5. Ensure environment has proper credentials (e.g., running `azure login` command, setting up managed identity, etc.) as the connection string would have previously contained the credentials.


## Document Loader Examples

### Load from container
Below shows how to load documents from all blobs in a given container in Azure Blob Storage:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    account_url=<your_account_url>,
    container_name=<your_container_name>,
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

The example below shows how to load documents from blobs in a container with a given prefix:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    account_url=<your_account_url>,
    container_name=<your_container_name>,
    prefix="test",
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

### Load from a list of blobs
Below shows how to load documents from a list of blobs in Azure Blob Storage:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    account_url=<your_account_url>,
    container_name=<your_container_name>,
    blob_names=["blob-1", "blob-2", "blob-3"],
)

for doc in loader.lazy_load():
    print(doc.page_content)
```

### Load asynchronously
Below shows how to load documents asynchronously which can be done by using either `aload()` or `alazy_load()`:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

async def main():
    loader = AzureBlobStorageLoader(
        account_url=<your_account_url>,
        container_name=<your_container_name>,
    )

    async for doc in loader.alazy_load():
        print(doc.page_content)
```


### Override loader
Below shows how to override the default loader used to parse blobs:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_unstructured import UnstructuredLoader

loader = AzureBlobStorageLoader(
    account_url=<your_account_url>,
    container_name=<your_container_name>,
    loader_factory=UnstructuredLoader,
)
```

To provide additional configuration, you can define a callable that returns an instantiated document loader as shown below:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_unstructured import UnstructuredLoader


def loader_factory(file_path: str) -> UnstructuredLoader:
    return UnstructuredLoader(
        file_path,
        mode="by_title",  # Custom configuration
        strategy="fast",  # Custom configuration
    )

loader = AzureBlobStorageLoader(
    account_url=<your_account_url>,
    container_name=<your_container_name>,
    loader_factory=loader_factory,
)
```

### Override credentials
Below shows how to override the default credentials used by the document loader:

```python
from azure.core.credentials import AzureSasCredential
from azure.identity import ManagedIdentityCredential
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

# Override with SAS token
loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    credential=AzureSasCredential("<sas-token>")
)

# Override with more specific token credential than the entire
# default credential chain (e.g., system-assigned managed identity)
loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    credential=ManagedIdentityCredential()
)
```