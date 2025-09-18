import azure.identity
import azure.core.credentials
import azure.core.credentials_async
from azure.storage.blob import ContainerClient, BlobClient
from typing import Optional, Union, Callable, Iterator, Iterable
from langchain_core.document_loaders import BaseLoader, BaseBlobParser
from langchain_core.documents.base import Document, Blob


SDK_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.TokenCredential,
    ]
]

def _get_client_kwargs(credential=None) -> dict:
    return {"credential": credential, "connection_data_block_size": 256 * 1024}


def _lazy_load_documents_from_blob(
    blob_client: BlobClient, parser: Optional[BaseBlobParser] = None
) -> Iterator[Document]:
    blob_data = blob_client.download_blob(max_concurrency=10)
    blob = Blob.from_data(
        data=blob_data.readall(),
        mime_type=blob_data.properties.content_settings.content_type,
        metadata={
            "source": blob_client.url,
        },
    )
    if parser:
        yield from parser.lazy_parse(blob)
    else:
        yield Document(page_content=blob.as_string(), metadata=blob.metadata)


def _get_sdk_credential(credential: SDK_CREDENTIAL_TYPE) -> SDK_CREDENTIAL_TYPE:
    if credential is None:
        return {
            "sync": azure.identity.DefaultAzureCredential(), 
        }
    if isinstance(credential, (
        azure.core.credentials.TokenCredential, 
        azure.core.credentials.AzureSasCredential, 
    )):
        return credential
    raise TypeError("Invalid credential type provided.")


class AzureBlobStorageLoader(BaseLoader):
    def __init__(
            self, 
            account_url: str,
            container_name: str,
            blob_names: Optional[Union[str, Iterable[str]]] = None,
            prefix: str = "",
            credential: Optional[
                Union[
                    azure.core.credentials.AzureSasCredential,
                    azure.core.credentials.TokenCredential,
                ]
            ] = None,
            loader_factory: Optional[Callable[str, BaseLoader]] = None,
    ):
        self._account_url = account_url
        self._container_name = container_name
        self._blob_names = [blob_names] if isinstance(blob_names, str) else blob_names
        self._prefix = prefix
        self._credential = credential
        self._loader_factory = loader_factory

    def lazy_load(self) -> Iterator[Document]:
        self._credential = _get_sdk_credential(self._credential)
        if isinstance(self._credential, azure.core.credentials_async.AsyncTokenCredential):
            raise ValueError("Async credential provided to sync method.")
        container_url = f"{self._account_url}/{self._container_name}"
        container_client = ContainerClient.from_container_url(
            container_url, **_get_client_kwargs(self._credential["sync"] if isinstance(self._credential, dict) else self._credential)
        )
        blobs_with_prefix = list(container_client.list_blob_names(name_starts_with=self._prefix))
        for blob_name in self._blob_names:
            if blob_name in blobs_with_prefix:
                blob_client = container_client.get_blob_client(blob_name)
                yield from _lazy_load_documents_from_blob(blob_client, self._loader_factory)
