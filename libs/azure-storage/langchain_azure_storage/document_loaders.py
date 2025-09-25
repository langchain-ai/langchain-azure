"""Azure Blob Storage document loader."""

from typing import Callable, Iterable, Iterator, Literal, Optional, Union

import azure.core.credentials
import azure.core.credentials_async
import azure.identity
from azure.storage.blob import BlobClient, ContainerClient
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document

_SDK_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.TokenCredential,
    ]
]


class AzureBlobStorageLoader(BaseLoader):
    """Document loader for loading LangChain Document object from Azure Blob Storage."""

    def __init__(
        self,
        account_url: str,
        container_name: str,
        blob_names: Optional[Union[str, Iterable[str]]] = None,
        *,
        prefix: Optional[str] = None,
        credential: _SDK_CREDENTIAL_TYPE = None,
        loader_factory: Optional[Callable[[str], BaseLoader]] = None,
    ):
        """Initialize AzureBlobStorageLoader.

        Args:
            account_url: URL to the Azure Storage account, e.g. "https://<account_name>.blob.core.windows.net"
            container_name: Name of the container to retrieve blobs from in the
                storage account
            blob_names: List of blob names to load. If None, all blobs will be loaded
            prefix: Prefix to filter blobs in the container
            credential: Credential to authenticate with the Azure Storage account.
                If None, DefaultAzureCredential will be used
            loader_factory: A callable that returns a custom document loader to use for
                parsing blobs downloaded. Ex: PyPDFLoader, CSVLoader, etc.
        """
        self._account_url = account_url
        self._container_name = container_name
        self._blob_names = [blob_names] if isinstance(blob_names, str) else blob_names
        self._prefix = prefix
        self._credential = credential
        self._loader_factory = loader_factory

    def _get_client_kwargs(self, credential: _SDK_CREDENTIAL_TYPE = None) -> dict:
        return {"credential": credential, "connection_data_block_size": 256 * 1024}

    def _lazy_load_documents_from_blob(
        self, blob_client: BlobClient
    ) -> Iterator[Document]:
        blob_data = blob_client.download_blob(max_concurrency=10)
        blob_content = blob_data.readall()
        yield Document(
            blob_content.decode("utf-8"), metadata={"source": blob_client.url}
        )

    def _get_sdk_credential(
        self, credential: _SDK_CREDENTIAL_TYPE, function_type: Literal["sync", "async"]
    ) -> _SDK_CREDENTIAL_TYPE:
        if credential is None:
            return azure.identity.DefaultAzureCredential()
        if isinstance(
            credential,
            (
                azure.core.credentials.TokenCredential,
                azure.core.credentials.AzureSasCredential,
            ),
        ):
            if function_type == "sync" and isinstance(
                credential, azure.core.credentials_async.AsyncTokenCredential
            ):
                raise TypeError("Async credential provided to sync method.")
            return credential
        raise TypeError("Invalid credential type provided.")

    def _yield_blob_names(self, container_client: ContainerClient) -> Iterator[str]:
        if self._blob_names is not None and self._prefix is not None:
            raise ValueError("Cannot specify both blob_names and prefix.")

        if self._blob_names:
            blob_list = list(container_client.list_blob_names())
            for blob_name in blob_list:
                if blob_name in self._blob_names:
                    yield blob_name
        elif self._prefix:
            yield from container_client.list_blob_names(name_starts_with=self._prefix)
        else:
            yield from container_client.list_blob_names()

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the specified blobs in the container.

        Yields:
            the documents.
        """
        self._credential = self._get_sdk_credential(self._credential, "sync")
        container_url = f"{self._account_url}/{self._container_name}"
        container_client = ContainerClient.from_container_url(
            container_url,
            **self._get_client_kwargs(self._credential),
        )

        for blob_name in self._yield_blob_names(container_client):
            blob_client = container_client.get_blob_client(blob_name)
            yield from self._lazy_load_documents_from_blob(blob_client)
