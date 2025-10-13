"""Azure Blob Storage document loader."""

import os
import tempfile
from typing import Callable, Iterable, Iterator, Optional, Union, get_args

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
    """Document loader for LangChain Document objects from Azure Blob Storage."""

    _CONNECTION_DATA_BLOCK_SIZE = 256 * 1024
    _MAX_CONCURRENCY = 10

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
            blob_names: List of blob names to load. If None, all blobs will be loaded.
            prefix: Prefix to filter blobs when listing from the container.
                Cannot be used with blob_names.
            credential: Credential to authenticate with the Azure Storage account.
                If None, DefaultAzureCredential will be used.
            loader_factory: Optional callable that returns a custom document loader
                (e.g. UnstructuredLoader) for parsing downloaded blobs. If provided,
                the blob contents will be downloaded to a temporary file whose name
                gets passed to the callable. If None, content will be returned as a
                single Document with UTF-8 text.
        """
        self._account_url = account_url
        self._container_name = container_name

        if blob_names is not None and prefix is not None:
            raise ValueError("Cannot specify both blob_names and prefix.")
        self._blob_names = [blob_names] if isinstance(blob_names, str) else blob_names
        self._prefix = prefix

        if credential is None or isinstance(credential, get_args(_SDK_CREDENTIAL_TYPE)):
            self._provided_credential = credential
        else:
            raise TypeError("Invalid credential type provided.")

        self._loader_factory = loader_factory

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from Azure Blob Storage.

        Yields:
            the documents.
        """
        credential = self._get_sync_credential(self._provided_credential)
        container_client = ContainerClient(**self._get_client_kwargs(credential))
        for blob_name in self._yield_blob_names(container_client):
            blob_client = container_client.get_blob_client(blob_name)
            yield from self._lazy_load_documents_from_blob(blob_client)

    def _get_client_kwargs(self, credential: _SDK_CREDENTIAL_TYPE = None) -> dict:
        return {
            "account_url": self._account_url,
            "container_name": self._container_name,
            "credential": credential,
            "connection_data_block_size": self._CONNECTION_DATA_BLOCK_SIZE,
        }

    def _lazy_load_documents_from_blob(
        self, blob_client: BlobClient
    ) -> Iterator[Document]:
        blob_data = blob_client.download_blob(max_concurrency=self._MAX_CONCURRENCY)
        blob_content = blob_data.readall()
        if self._loader_factory is None:
            yield Document(
                blob_content.decode("utf-8"), metadata={"source": blob_client.url}
            )
        else:
            yield from self._yield_documents_from_custom_loader(
                blob_content, blob_client
            )

    def _yield_documents_from_custom_loader(
        self, blob_content: bytes, blob_client: BlobClient
    ) -> Iterator[Document]:
        with tempfile.TemporaryDirectory() as temp_dir:
            blob_name = os.path.basename(blob_client.blob_name)
            temp_file_path = os.path.join(temp_dir, blob_name)
            with open(temp_file_path, "wb") as file:
                file.write(blob_content)

            if self._loader_factory is not None:
                loader = self._loader_factory(temp_file_path)
                for doc in loader.lazy_load():
                    doc.metadata["source"] = blob_client.url
                    yield doc

    def _get_sync_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> _SDK_CREDENTIAL_TYPE:
        if provided_credential is None:
            return azure.identity.DefaultAzureCredential()
        if isinstance(
            provided_credential, azure.core.credentials_async.AsyncTokenCredential
        ):
            raise ValueError(
                "Cannot use synchronous load methods when AzureBlobStorageLoader is "
                "instantiated using an AsyncTokenCredential. Use its asynchronous load "
                "method instead or supply a synchronous TokenCredential to its "
                "credential parameter."
            )
        return provided_credential

    def _yield_blob_names(self, container_client: ContainerClient) -> Iterator[str]:
        if self._blob_names is not None:
            yield from self._blob_names
        else:
            yield from container_client.list_blob_names(name_starts_with=self._prefix)
