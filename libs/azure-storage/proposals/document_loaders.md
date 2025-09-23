# Azure Storage document loaders

| Proposal    | Metadata          |
|-------------|-------------------|
| **Author**  | Kyle Knapp        |
| **Status**  | Proposed          |
| **Created** | 12-September-2025 |

## Abstract
This proposal outlines the design and implementation of Azure Blob Storage document loaders. These
document loaders will be part of the [`langchain-azure-storage`][langchain-azure-storage-pkg] package and
provide first-party support for loading LangChain [`Document`][langchain-document-ref] objects from either
a container or a specific blob in Azure Blob Storage. These document loaders aim to replace the existing
community-sourced Azure Blob Storage document loaders, which can be improved in several areas and are not
directly maintained by Azure Storage.

## Background and motivation

### What are document loaders?
[Document loaders][langchain-document-loader-concept] load data from a variety of sources (e.g., local
filesystem, cloud storage, etc.) and parse the data (e.g., from JSON, PDF, DOCX, etc.) to human readable
text. Typically, the loaded text is then stored along with an [embedding][langchain-embedding-concept]
representation in a [vector store][langchain-vector-store-concept] for use in [RAG][langchain-rag-concept].
Using [semantic search][wiki-semantic-search], this allows an LLM to retrieve information from sources outside its training data to use to
generate responses.


Below shows a basic example of how document loaders fit into a LangChain workflow. Outside the
runtime of an AI application, document loaders are utilized to seed a vector store with
documents:

```python
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_core.vectorstores import InMemoryVectorStore

# Load a local PDF using a PDF document loader. This will return
# the text of each page in the PDF as `Document` objects.
documents = PyPDFLoader('azure-storage-user-guide.pdf').load()

# Split the documents by character chunks to create smaller documents
# to be stored in the vector store. This allows for retrieving smaller,
# more relevant chunks of text to use as context for the LLM.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunked_documents = text_splitter.split_documents(documents)

# Configure an embedding model to use to embed each chunk of text.
embed_model = AzureAIEmbeddingsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    model="text-embedding-3-large",
)
# Create a vector store to store chunks and their embeddings. Using an in-memory
# vector store here for example simplicity, but in production a more robust vector store
# (e.g., Azure AI search, etc.) should be used.
db = InMemoryVectorStore(embed_model)
db.add_documents(chunked_documents)
```
Then within the runtime of the AI application, the vector store can be queried to retrieve
relevant documents to use as context for the LLM:

```python
# At AI application runtime, the vector store will be connected to and accessed in its
# retriever mode to be able to perform semantic search on documents stored in the vector store.
db = get_vector_store()
retriever = db.as_retriever()

# Retrieves original PDF chunks that are relevant to the query as `Document` objects.
docs = retriever.invoke("What are the different types of blobs in Azure Blob Storage?")
```

All document loader implementations follow the [`BaseLoader`][langchain-document-loader-base-ref] interface whose methods
return [`Document`][langchain-document-ref] objects representing the parsed text of a loaded document. `Document` objects
are also widely expected in downstream LangChain RAG components such as [text splitters][langchain-text-splitter-concept] and
[vector stores][langchain-vector-store-concept].

Today, there is a wide variety of [community-maintained document loader][langchain-document-loader-integrations] implementations
available for LangChain. In general, each document loader implementation:

* Typically, loads data from only a single logical source (e.g. a local directory, a cloud storage bucket, social media platform, etc.). For example,
  there is a [`DirectoryLoader`][langchain-directory-loader] that loads files from a local directory and an [`S3DirectoryLoader`][langchain-s3-directory-loader]
  that loads files from an S3 bucket, but there is no superset loader implementation that loads files from both a local directory and an S3 bucket.
* Supports parsing data for 1 to N different file types (e.g. `.pdf`, `.docx`, `.txt`, etc.). For example, the [`JSONLoader`][langchain-json-loader] supports only
  parsing JSON content to text, but the [`UnstructuredLoader`][langchain-unstructured-loader] supports a variety of formats (e.g. `.txt`, `.pdf`, `.docx`, etc.).
* May provide built-in chunking of documents into smaller documents to be stored in a vector store. If supported, chunking configuration is configurable (
  e.g., whether chunking is enabled, how to chunk, etc.).

While technically document loaders are not needed to store text in a vector store (i.e., an individual could load and parse documents themselves and then insert
the text directly into a vector store), document loaders provide value in they:

* Abstract away the details of loading and parsing documents from various sources and formats. Developers just need to provide configuration to the document loader
  (e.g., path to directory, connection details to cloud storage, etc.) and the document loader handles the logic needed to extract a resource's text.
* Provides a consistent interface for loading documents. This makes it easy to load documents across a variety of different sources and formats
  and use across other LangChain components (e.g., text splitters, vector stores, etc.) without needing custom adapters.


### How does document loaders fit with Azure Storage?
Azure Blob Storage is heavily used for storing unstructured data whether it
be documents, CSVs, logs, etc. This sort of data is commonly used in RAG scenarios to
provide context to LLMs (e.g., searching company documents to answer questions about company policies). Having purpose-built Azure Blob Storage document loaders allows
AI application developers to easily retrieve these documents from Azure Blob Storage,
augment them with embeddings, and store them in a vector store of their choice for use in RAG scenarios.

While for larger AI applications a developer would likely opt for a full-fledged embedding ingestion pipeline (e.g., includes scheduled/event-driven re-embeddings,
content enrichment stages, data lineage, etc.), document loaders provide the opportunity to quickly prototype and get up and running with a RAG application.


### Community Azure Storage document loaders
Prior to this proposal, there are two community-built Azure Blob Storage document loaders:

* [`AzureBlobStorageFileLoader`][community-azure-blob-storage-file] - Loads `Document` objects from a single blob from Azure Blob Storage.
* [`AzureBlobStorageContainerLoader`][community-azure-blob-storage-container] - Loads `Document` objects from all blobs from a container in Azure Blob Storage.

These document loaders were contributed to Langchain as part of this [pull request][community-pr] to satisfy a [feature request][community-feature-request] asking
for Azure Blob Storage document loaders to match already offered S3 and GCS document loaders. Today, these document loaders reside in
the [langchain-community repository][community-repository], which is the de facto package for third-party LangChain integrations that do not
have an owner outside the LangChain core team.


#### Areas of improvement with current community document loaders

With the current community document loaders, there are several opportunities for
improvement:

* There is no active owner or maintainer of these document loaders. Recently, Azure
created the [`langchain-azure`][langchain-azure-repo] repository to host and take
first-party ownership of Azure-related LangChain integrations. Pulling these
document loaders into this repository would allow Azure Storage to provide
first-party support and maintenance of these document loaders. This approach also aligns
with the direction from the LangChain core team to push community integrations to dedicated
`langchain-*` packages for individual integrations instead of continuing to maintain them in the
`langchain-community` package.
* The document loaders do not implement [`lazy_load()`][langchain-lazy-load-ref]. This
method allows customers to lazily iterate through all documents without loading them all into memory at once. This is especially
important in the context of Azure Blob Storage where billions+ of documents could be stored totaling to TiB+ of data.
* The document loaders only support connection string for authentication. This
prevents customers from following best practices around using Microsoft Entra ID
and managed identities for authentication and authorization.
* By default, uses the [`UnstructuredLoader`][langchain-unstructured-loader] for parsing documents. Unstructured provides
wide coverage for parsing different file formats. However, this decision forces Azure Storage customers to rely on Unstructured
and any of its underlying dependencies. Furthermore, there are no options to customize how to load and parse blobs that are
downloaded from Azure Blob Storage (e.g., use a different parser).
* Uses the default async implementations of `aload()` and `alazy_load()` which just
runs the synchronous implementations in an executor. While this prevents blocking the
event loop, the Azure Storage SDK offers asynchronous interfaces that could further improve performance when accessing document loaders asynchronously.

See [Appendix A](#appendix-a-reported-customer-issues-with-community-azure-storage-document-loaders) for a list of
reported customer issues with the existing community Azure Storage document loaders that echo the areas of improvement listed above.

### Goals
Based on the background, the goals of this proposal and design are to:
* Port the existing community Azure Blob Storage document loaders into the `langchain-azure` repository so that they can receive first-party support and maintenance.
* Address the areas of improvement listed [above](#areas-of-improvement-with-current-community-document-loaders) as part of implementation port.
* Minimize interfaces changes from existing community document loaders to only those that are necessary. Ideally, changes needed to migrate away from the current
community document loaders should be minimal, requiring at most changes to import statements and  constructor call patterns.
* Prioritize ease of use and sensible defaults. Ideally, customers should only need
to provide minimal configuration (e.g., the resource URL) to get up and running
quickly. It should not require in-depth knowledge of either Azure Blob
Storage or the SDKs in order to use.

### Non-goals
Below are non-goals for this proposal:
* Adding Azure Storage integrations for LangChain components outside of
document loaders (e.g., [blob loaders][langchain-blob-loader-ref] and
[memory stores][langgraph-memory-store]). This is intended to keep the scope of
the proposal narrow and manageable. These additional integration opportunities should be researched and considered in future proposals.
* Maximizing available network throughput to the environment. Implementation should
still effectively use the SDK to download blobs quickly, but this first iteration
should not be targeting additional optimization layers to further boost download
throughput (e.g., eagerly download batches of blobs in parallel) especially when
blobs are expected to be lazily loaded one at a time to be chunked, embedded,
and stored in a vector store. Based on feedback from the initial implementation,
we can explore speed optimizations in future iterations.


## Specification
Below is the proposed specification for the Azure Blob Storage document loaders.

### Public interfaces
All Azure Storage document loaders will live in the [`langchain_azure_storage` package][langchain-azure-storage-pkg]
under a new `document_loaders` module.

There will be a single document loader introduced, `AzureBlobStorageLoader`. This single loader will encompass
functionality from both the community-sourced `AzureBlobStorageFileLoader` and `AzureBlobStorageContainerLoader`
document loaders.

The document loader will subclass from [`BaseLoader`][langchain-document-loader-base-ref] and support both synchronous
and asynchronous loading of documents, as well as lazy loading of documents.

Below shows the proposed constructor signature for the document loader:

```python
from typing import Optional, Union, Callable, Iterable
import azure.core.credentials
import azure.core.credentials_async
from langchain_core.document_loaders import BaseLoader


class AzureBlobStorageLoader(BaseLoader):
    def __init__(self,
        account_url: str,
        container_name: str,
        blob_names: Optional[Union[str, Iterable[str]]] = None,
        *,
        prefix: Optional[str] = None,
        credential: Optional[
            Union[
                azure.core.credentials.AzureSasCredential,
                azure.core.credentials.TokenCredential,
                azure.core.credentials_async.AsyncTokenCredential,
            ]
        ] = None,
        loader_factory: Optional[Callable[str, BaseLoader]] = None,
    ): ...
```

In terms of parameters supported:
* `account_url` - The URL to the storage account (e.g., `https://<account>.blob.core.windows.net`)
* `container_name` - The name of the container within the storage account
* `blob_names` - The name of the blob(s) within the container to load. If provided, only the specified blob(s)
in the container will be loaded. If not provided, the loader will list blobs from the container to load, which
will be all blobs unless `prefix` is specified.
* `credential` - The credential object to use for authentication. If not provided,
the loader will use [Azure default credentials][azure-default-credentials]. The
`credential` field only supports token-based credentials and SAS credentials. It does
not support access key based credentials nor anonymous access.
* `prefix` - An optional prefix to filter blobs when listing from the container. Only blobs whose names start with the
specified prefix will be loaded. This parameter is incompatible with `blob_names` and will raise a `ValueError` if both
are provided.
* `loader_factory` - A callable that returns a custom document loader (e.g., `UnstructuredLoader`) to use
for parsing blobs downloaded. When provided, the Azure Storage document loader will download each blob to
a temporary local file and then call `loader_factory` with the path to the temporary file to get a document
loader to use to load and parse the local file as `Document` objects. If `loader_factory` is not provided,
the loader will return  the content as is in a single `Document` object for each blob. The blob content will be
treated as UTF-8 encoding for this default case.


### Usage examples
Below are some example usage patterns for the Azure Blob Storage document loaders.

#### Load from a blob
Below shows how to load a document from a single blob in Azure Blob Storage:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader("https://<account>.blob.core.windows.net", "<container>", "<blob>")
for doc in loader.lazy_load():
    print(doc.page_content)  # Prints content of blob. There should only be one document loaded.
```

### Load from a list of blobs
Below shows how to load documents from a list of blobs in Azure Blob Storage:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    ["blob1", "blob2", "blob3"]
)
for doc in loader.lazy_load():
    print(doc.page_content)  # Prints content of each blob from list.
```

#### Load from a container

Below shows how to load documents from all blobs in a given container in Azure Blob Storage:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader("https://<account>.blob.core.windows.net", "<container>")
for doc in loader.lazy_load():
    print(doc.page_content)  # Prints content of each blob in the container.
```

Below shows how to load documents from blobs in a container with a given prefix:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net", "<container>", prefix="some/prefix/"
)
for doc in loader.lazy_load():
    print(doc.page_content)  # Prints content of each blob whose name starts with "some/prefix/" .
```

#### Load asynchronously
Below shows how to load documents asynchronously. This is acheived by calling the `aload()` or `alazy_load()` methods on the document loader. For example:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader


async def main():
    loader = AzureBlobStorageLoader("https://<account>.blob.core.windows.net", "<container>")
    async for doc in loader.alazy_load():
        print(doc.page_content)  # Prints content of each blob in the container.
```

#### Override credentials
Below shows how to override the default credentials used by the document loader:

```python
from azure.core.credentials import AzureSasCredential
from azure.idenity import ManagedIdentityCredential
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

#### Override loader
Below shows how to override the default document loader used to parse downloaded blobs.

In the below example, the blobs are downloaded to a temporary local file and uses
the `UnstructuredLoader` to parse the local file and return `Document` objects
on behalf of the Azure Storage document loader:

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_unstructured import UnstructuredLoader

loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    # The UnstructuredLoader class accepts a string to the local file path to its constructor,
    # so the class can be provided directly as the loader_factory.
    loader_factory=UnstructuredLoader
)
for doc in loader.lazy_load():
    # Document returned are generated directly from UnstructuredLoader and
    # are not just the unmodified blob content.
    print(doc.page_content)
```

If a customer wants to provide additional configuration to the document loader, they can
define a callable that returns an instantiated document loader. For example, to provide
custom configuration to the `UnstructuredLoader`:
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
    "https://<account>.blob.core.windows.net",  "<container>",
    loader_factory=loader_factory
)
```


### Migration details

In migrating from the existing community document loaders to the new Azure Storage document loaders,
customers will need to perform the following changes:

1. Depend on the `langchain-azure-storage` package instead of `langchain-community`.
2. Update import statements from `langchain_community.document_loaders` to
   `langchain_azure_storage.document_loaders`.
3. Change class names from `AzureBlobStorageFileLoader` and `AzureBlobStorageContainerLoader`
   to `AzureBlobStorageLoader`.
4. Update document loader constructor calls to:
   1. Use an account URL instead of a connection string.
   2. Specify `UnstructuredLoader` as the `loader_factory` if they continue to want to use
      Unstructured for parsing documents.
5. Ensure environment has proper credentials (e.g., running `azure login` command, setting up
   managed identity, etc.) as the connection string would have previously contained the credentials.

Below shows code snippets of what usage patterns look like before and after the proposed migration:

**Before migration:**

```python
from langchain_community.document_loaders import AzureBlobStorageContainerLoader, AzureBlobStorageFileLoader

container_loader = AzureBlobStorageContainerLoader(
    "DefaultEndpointsProtocol=https;AccountName=<account>;AccountKey=<account-key>;EndpointSuffix=core.windows.net",
    "<container>",
)

file_loader = AzureBlobStorageFileLoader(
    "DefaultEndpointsProtocol=https;AccountName=<account>;AccountKey=<account-key>;EndpointSuffix=core.windows.net",
    "<container>",
    "<blob>"
)
```

**After migration:**

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_unstructured import UnstructuredLoader

container_loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    loader_factory=UnstructuredLoader
)

file_loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    "<blob>",
    loader_factory=UnstructuredLoader
)
```


### Implementation details

This section dives into implementation details stemming from public interface outlined above.

### `BaseLoader` methods to implement
For each of the Azure Storage document loaders, the following methods from `BaseLoader` will be implemented:

* [`lazy_load()`][langchain-lazy-load-ref] - Lazily loads documents one at a time
* [`alazy_load()`][langchain-alazy-load-ref] - Asynchronously and lazily loads documents one at a time. This
should use the asynchronous SDK clients instead of rely on the default `alazy_load()` implementation which just runs
the synchronous implementation in an executor.

For the rest of the methods offered by `BaseLoader` (e.g., `load()`, `aload()`), the default
document loader implementations will be used, which call into the lazy loading methods and do not
require any overrides.

### Credential details

When no `credential` is provided, the document loaders will use [Azure default credentials][azure-default-credentials].
Specifically, the document loaders will instantiate `azure.identity.DefaultAzureCredential` and
`azure.identity.aio.DefaultAzureCredential` credential objects and provide them to the synchronous and
asynchronous SDK clients respectively.

When a `credential` is provided, the credential will be:
* Validated to ensure it is one of the supported credential types. If not, a `ValueError` will be raised.
* Passed directly to the SDK client's `credential` parameter.
* If a synchronous token credential (e.g., `TokenCredential`) is provided and asynchronous methods are called
  (e.g., `aload()`), the method will raise a `ValueError`. The reverse applies as well (i.e., throwing exceptions when
  providing an asynchronous token credential and calling synchronous methods). For example:
  ```python
  import azure.identity
  import azure.identity.aio
  from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

  sync_doc_loader = AzureBlobStorageLoader(
      "https://<account>.blob.core.windows.net",
       "<container>",
       credential=azure.identity.ManagedIdentityCredential()
  )
  sync_doc_loader.aload()  # Raises ValueError because a sync credential was provided

  async_doc_loader = AzureBlobStorageLoader(
    "https://<account>.blob.core.windows.net",
    "<container>",
    credential=azure.identity.aio.ManagedIdentityCredential()
  )
  async_doc_loader.load()  # Raises ValueError because an async credential was provided
  ```
  If a customer wants to access both the synchronous and asynchronous methods, they will need to
  instantiate two separate document loaders: one for sync operations and one for async operations.
  Also note that this restriction does not apply to when no credential is provided and default credentials are used;
  the document loader implementation will automatically handle matching the correct credential to client type.

### Document metadata

By default, the document loaders will populate the `source` metadata field of each `Document`
object with the URL of the blob (e.g., `https://<account>.blob.core.windows.net/<container>/<blob>`). For example:
```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

loader = AzureBlobStorageLoader("https://<account>.blob.core.windows.net", "<container>")
for doc in loader.lazy_load():
    print(doc.metadata["source"])  # Prints URL of each blob in the container.
```
If a custom `loader_factory` is provided, the document loaders will override any `source` metadata field
set by the custom loader to be the URL of the blob but retain all other metadata fields set by the custom loader.


### `loader_factory` mechanics

When no `loader_factory` is provided, the document loaders will download the blob content
and return a single `Document` object per blob with the blob content treated as UTF-8. This
will happen all in-memory without writing the blob to a temporary file.

If a `loader_factory` is provided, the document loaders will download each blob to a temporary local file
and then call `loader_factory` with the path to the temporary file to get a document loader to use to load and parse the local file. Below shows a simplified example of how this would work:
```python
import tempfile
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator


class AzureBlobStorageLoader(BaseLoader):
    ...
    def _lazy_load_from_custom_loader(self, blob_name: str) -> Iterator[Document]:
        with tempfile.NamedTemporaryFile() as temp_file:
            self._download_blob_to_file(blob_name, temp_file.name)
            loader = self._loader_factory(temp_file.name)
            yield from loader.lazy_load()
```
It's important to note that the temporary file should be deleted after it has been loaded by the custom loader
so that disk usage does not continue to grow over the duration of the `lazy_load()` or `load()` invocation.


## Rationale

This section discusses the rationale behind design decisions made in the specification above.

### Alternatives considered

Below discusses alternatives considered for key design decisions made in the specification above.

#### Exposing Azure storage integrations as a blob loader instead of document loader
A [blob loader][langchain-blob-loader-ref] is another type of LangChain loader that just loads
the raw bytes from storage and does not parse the data to text. Instead, a [blob parser][langchain-blob-parser-ref]
is used to parse the raw bytes to text. Together, blob loaders and parsers effectively help decouple how
data is loaded from how data is converted to text for an LLM. Blob loaders are arguably even a cleaner
integration point for Azure Storage since data stored in Azure Blob Storage is generally unstructured and could
be in any format.

However, the decision to support document loaders instead is a reflection of meeting customers where they are
at:

* Document loaders are more widely used and have far more 3rd party integrations than blob loaders and
  blob parsers. For these reasons, customers will expect that the integration is in the form of a document
  loader.
* The current community Azure storage integrations are document loaders. If we were to only supporting blob
  loaders, the migration to `langchain-azure-storage` would be more involved such as needing to learn how
  blob loaders and parsers work and possibly implementing their own blob parser if the LangChain community
  does not offer a parser that meets their needs.

In the future, we should consider exposing blob loaders if requested by customers, especially if we see
customers wanting to customize loading behavior beyond what is offered by the `loader_factory` parameter.


#### Exposing a `blob_parser` parameter instead of `loader_factory`
In order to customize how content was parsed to text, it was considered to expose a `blob_parser` parameter,
which would accept a [blob parser][langchain-blob-parser-ref] to use to parse the raw bytes of a blob to text.
The advantage of this approach is that:
1. The implementation would not need to write to disk since blob parsers accept blobs loaded into memory. With
   the `loader_factory` approach, the blob needs to be written to disk because most document loaders expect a
   path to a local file to load from.
2. There would be more customization available to the customer as we could more naturally proxy metadata of the blob
   stored in Azure Blob Storage (e.g., content type, last modified time, tags, metadata etc.) to the `blob_parser`.

However, similar to why document loaders were chosen over blob loaders, blob parsers do not have as wide of
3rd party support as document loaders, which would require customers to write their own blob parser wrappers
over libraries like Unstructured and takeaway from the batteries-included value proposition that LangChain document
loaders provide.

It's important to note that this decision does not prevent us from exposing a `blob_parser` parameter in the future.
Specifically, this would be useful if we see customers wanting to customize loading behavior more but not necessarily
want to drop down to using a blob loader interface.


#### Exposing document loaders as two classes, `AzureBlobStorageFileLoader` and `AzureBlobStorageContainerLoader`, instead of a single `AzureBlobStorageLoader`
Exposing the document loaders as these two classes would be beneficial in that they would match the existing community
document loaders and lessen the amount of changes needed to migrate. However, combining them into a single class
has the following advantages:

* It simplifies the getting started experience. Customers will no longer have to make a decision on which Azure Storage
document loader class to use as there will be only one document loader class to choose from.
* It simplifies class names by removing the additional `File` and `Container` qualifiers, which could lead to
misinterpretations on what the classes do.
* It is easier to maintain as there is only one class that will need to be maintained and less code will likely need to
be duplicated.

While this will introduce an additional step in migrating (i.e., change class names), the impact is limited
as customers will still be providing the same positional parameters even after changing class names
(i.e., use account + container for the container loader and account + container + blob for the file loader).


#### Alternatives to default parsing to UTF-8 text
The default parsing logic when no `loader_factory` is provided is to treat the blob content as UTF-8 text
and return a single `Document` object per blob. This was chosen to have some default to get up and running
with document loaders with minimal configuration. Alternatives considered included:
* Require a `loader_factory` to be provided. While it is likely customers will want to provide their own loader
factory (unless all of their documents are just plain text), not having a default parser would add friction
to getting started in that they will now need to decide both how to properly setup their connection to Azure Blob Storage and have to decide which document loader to use.
* Default to a more full-featured document loader like [`UnstructuredLoader`][langchain-unstructured-loader].
However, this would require taking a 3rd party dependency outside the Azure Storage and
LangChain ecosystem, which customers would likely not expect as a first-party Azure Storage integration.
Furthermore, LangChain does not offer a recommended default document loader nor blob parser to use for integrations.


#### Alternatives to defaulting to Azure default credentials
Using [Azure default credentials][azure-default-credentials] when no credential is provided was chosen to
minimize configuration needed to get up and running in addition to defaulting to using Oauth2 token-based
credentials, which is the recommended best practice for authenticating to Azure services. Furthermore,
many of the other Azure LangChain integrations packages (e.g., `langchain-azure-dynamic-sessions` and
`langchain-azure-postgresql`) default to Azure default credentials. In terms of alternative defaults
considered, these included:
* Requiring a credential to be provided. While this would force customers to think about authentication
and authorization, it would add friction to getting started as customers would need to now learn about
the various in-code credential types and select the proper one to use. Furthermore, in most cases,
customers will want to use the Azure default credentials anyway, especially for getting started.
* Match SDK default of anonymous access when no credential is provided. Most customers will want to be
making authenticated requests to Azure Blob Storage. So customers would likely need to provide a credential
anyway with the added undesired side effect that to resolve auth issues, customers may also try to just make
the container public.


### FAQs

Below are some additional questions and answers about the design not covered by alternatives considered
above.

#### Q: Why not support access key, anonymous, or connection string based credentials?
It is purposely not supported to encourage customers to follow best practices around using
Microsoft Entra ID and managed identities for authentication and authorization. While the community
document loaders only supported connection string based authentication, it is a relatively small
configuration value change to switch to using account URL, especially since account URL is a heavily
prevalent configuration value in Azure Blob Storage.

#### Q: Why not support synchronous token credentials when calling asynchronous methods and vice versa?
In general, it is not recommended to interchange synchronous credentials with asynchronous clients and
vice versa (e.g., synchronous credentials can block the async event loop). Furthermore, there are no
adapter mechanisms to use them interchangeably.  By enforcing this restriction, it ensures customers
do not accidentally use the wrong credential type for their sync or async usage pattern.

Technically, we could support this use case by being able to accept both sync and async token credentials
at instantiation (e.g., expose an `async_credential` parameter or a tuple of sync and async credentials).
However, this could introduce more complexity to the interface, when we expect for a single document loader
instance, customers will likely be using only sync or only async methods, but not both.

If we get the feedback that customers want to use both sync and async methods with a single
document loader instance, we can revisit this decision in the future.


#### Q: How would the `loader_factory` expand in the future?
In general, the `loader_factory` is intended to be a simple escape hatch to allow customers to
customize how blobs are parsed to text. However, possible requested extension points may include:
* Needing blob properties to make decisions on how to parse the blob.
* Wanting the blob data to be passed using an in-memory representation than file on disk

If we ever plan to extend the interface, we should strongly consider exposing blob loaders
and/or a `blob_parser` parameter instead as discussed in the [alternatives considered](#exposing-a-blob_parser-parameter-instead-of-loader_factory)
section above.

If blob loaders nor a `blob_parser` parameter suffice, we could consider expanding the `loader_factory` to:

* Inspect signature arguments of callable provided to `loader_factory` and call the callable with
  additional parameters if detected (e.g., detect if the a `blob_properties` parameter is present and
  proxy it when calling the callable).
* Introduce additional opt-in parameters to the document loader constructor to control how the
  `loader_factory` is called. For example, we could expose an `include_blob_properties` to include
  blob properties when calling the `loader_factory`.


#### Q: Why is the blob properties not exposed in the `Document` metadata?
It was done to simplify the initial implementation and provide flexibility in how blob properties are
represented as document metadata in the future. The `source` field is a widely adopted metadata field and
is generally expected to be set. However, there is no strong guidance on other metadata fields
beyond `source`.

Based on customer requests, in the future, we could consider exposing these properties by either:
* Adding a `blob_properties` field in the `Document.metadata` dictionary to store all blob properties
* Mapping certain blob properties to commonly expected metadata fields (e.g., map last modified time to
  `Document.metadata["last_modified"]` or content type to `Document.metadata["type"]`)


## Future work
Below are some possible future work ideas that could be considered after the initial implementation based on customer feedback:

* Expose blob loader and/or blob parser integrations (see [alternatives considered](#exposing-a-blob_parser-parameter-instead-of-loader_factory) section).
* Proxy additional blob properties as document metadata (see [FAQs](#q-why-is-the-blob-properties-not-exposed-in-the-document-metadata) section).
* Support `async_credential` parameter to allow using both sync and async token credentials with a single document loader instance
  (see [FAQs](#q-why-not-support-synchronous-token-credentials-when-calling-asynchronous-methods-and-vice-versa) section).
* Support integrations that streamline loading documents directly from Azure Storage as part of the
vector store or memory store interface instead of needing to do it manually with document loaders (see
[non-goals](#non-goals) section).


## Appendix

### Appendix A: Reported customer issues with community Azure Storage document loaders
Below audits some of the issues reported by customers with the existing community Azure Storage document
loaders. Community reported issues were located by searching LangChain discussion board for ["AzureBlobStorageFileLoader"][issue-tracker-langchain-1] and ["AzureBlobStorageContainerLoader"][issue-tracker-langchain-2]. Note there were no reports filed in the LangChain GitHub issue queue.

* [#7883](https://github.com/langchain-ai/langchain/discussions/7883) - Customer wants the entire document
returned from container loader and not something that is already chunked using Unstructured.
* [#8615](https://github.com/langchain-ai/langchain/discussions/8615) - Customer wants to use PDF parser
to load blobs from container.
* [#9743](https://github.com/langchain-ai/langchain/discussions/9743) - Customer wants to be able to load
the markdown as is without using Unstructured to parse the markdown.
* [#9934](https://github.com/langchain-ai/langchain/discussions/9934) - Customer wants to be able to use
token-based authentication instead of forced to use access keys via connection string.
* [#17812](https://github.com/langchain-ai/langchain/discussions/17812) - Customer wants to load blobs
using `PyPDFLoader` instead of use behavior from `UnstructuredLoader`.
* [#19992](https://github.com/langchain-ai/langchain/discussions/19992) - Customer notes that process
to parse blobs using Unstructured is really slow for their environment. Being able to customize the loader
would unblock them.


<!-- Reference Links -->
[langchain-document-loader-concept]: https://python.langchain.com/docs/concepts/document_loaders/
[langchain-text-splitter-concept]: https://python.langchain.com/docs/concepts/text_splitters/
[langchain-embedding-concept]: https://python.langchain.com/docs/concepts/embedding_models/
[langchain-vector-store-concept]: https://python.langchain.com/docs/concepts/vectorstores/
[langchain-rag-concept]: https://python.langchain.com/docs/concepts/rag/
[langchain-document-loader-integrations]: https://python.langchain.com/docs/integrations/document_loaders/
[wiki-semantic-search]: https://en.wikipedia.org/wiki/Semantic_search
[langchain-document-ref]: https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html
[langchain-document-loader-base-ref]: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseLoader.html
[langchain-lazy-load-ref]: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseLoader.html#langchain_core.document_loaders.base.BaseLoader.lazy_load
[langchain-alazy-load-ref]: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseLoader.html#langchain_core.document_loaders.base.BaseLoader.alazy_load
[langchain-blob-loader-ref]: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.blob_loaders.BlobLoader.html
[langchain-blob-parser-ref]: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseBlobParser.html
[langchain-directory-loader]: https://python.langchain.com/docs/how_to/document_loader_directory/
[langchain-s3-directory-loader]: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.s3_directory.S3DirectoryLoader.html
[langchain-json-loader]: https://python.langchain.com/docs/integrations/document_loaders/json/
[langchain-unstructured-loader]: https://python.langchain.com/docs/integrations/document_loaders/unstructured_file/
[langgraph-memory-store]: https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store

[community-repository]: https://github.com/langchain-ai/langchain-community
[community-azure-blob-storage-file]: https://python.langchain.com/docs/integrations/document_loaders/azure_blob_storage_file/
[community-azure-blob-storage-container]: https://python.langchain.com/docs/integrations/document_loaders/azure_blob_storage_container/
[issue-tracker-langchain-1]: https://github.com/langchain-ai/langchain/discussions?discussions_q=is%3Aopen+AzureBlobStorageFileLoader+
[issue-tracker-langchain-2]: https://github.com/langchain-ai/langchain/discussions?discussions_q=is%3Aopen+AzureBlobStorageContainerLoader+
[community-pr]: https://github.com/langchain-ai/langchain/pull/1890
[community-feature-request]: https://github.com/langchain-ai/langchain/issues/1805

[langchain-azure-repo]: https://github.com/langchain-ai/langchain-azure
[langchain-azure-storage-pkg]: https://pypi.org/project/langchain-azure-storage/

[azure-default-credentials]: https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview