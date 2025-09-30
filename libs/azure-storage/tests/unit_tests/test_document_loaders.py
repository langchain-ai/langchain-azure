from typing import Any, Callable, Iterator, Tuple
from unittest.mock import MagicMock, patch

import azure.identity
import pytest
from azure.storage.blob import BlobClient, ContainerClient
from azure.storage.blob._download import StorageStreamDownloader

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader


@pytest.fixture
def account_url() -> str:
    return "https://testaccount.blob.core.windows.net"


@pytest.fixture
def container_name() -> str:
    return "test-container"


@pytest.fixture
def blobs() -> list[dict[str, str]]:
    return [
        {"blob_name": "text_file.txt", "blob_content": "test content"},
        {"blob_name": "json_file.json", "blob_content": "{'test': 'test content'}"},
        {"blob_name": "csv_file.csv", "blob_content": "col1,col2\nval1,val2"},
    ]


@pytest.fixture
def create_azure_blob_storage_loader(
    account_url: str, container_name: str
) -> Callable[..., AzureBlobStorageLoader]:
    def _create_azure_blob_storage_loader(**kwargs: Any) -> AzureBlobStorageLoader:
        return AzureBlobStorageLoader(
            account_url,
            container_name,
            **kwargs,
        )

    return _create_azure_blob_storage_loader


@pytest.fixture
def mock_blob_clients(
    account_url: str, container_name: str, blobs: list[dict[str, str]]
) -> Callable[[str], MagicMock]:
    def _get_blob_client(blob_name: str) -> MagicMock:
        mock_blob_client = MagicMock(spec=BlobClient)
        mock_blob_client.url = f"{account_url}/{container_name}/{blob_name}"
        mock_blob_data = MagicMock(spec=StorageStreamDownloader)
        content = next(
            blob["blob_content"] for blob in blobs if blob["blob_name"] == blob_name
        )
        mock_blob_data.readall.return_value = content.encode("utf-8")
        mock_blob_client.download_blob.return_value = mock_blob_data
        return mock_blob_client

    return _get_blob_client


@pytest.fixture(autouse=True)
def mock_container_client(
    blobs: list[dict[str, str]], mock_blob_clients: Callable[[str], MagicMock]
) -> Iterator[Tuple[MagicMock, MagicMock]]:
    with patch(
        "langchain_azure_storage.document_loaders.ContainerClient"
    ) as mock_container_client_cls:
        mock_client = MagicMock(spec=ContainerClient)
        mock_client.list_blob_names.return_value = [blob["blob_name"] for blob in blobs]
        mock_client.get_blob_client.side_effect = mock_blob_clients
        mock_container_client_cls.return_value = mock_client
        yield mock_container_client_cls, mock_client


def test_lazy_load(
    account_url: str,
    container_name: str,
    blobs: list[dict[str, str]],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader()
    docs = loader.lazy_load()
    assert len(list(docs)) == len(blobs)
    for index, doc in enumerate(docs):
        assert doc.page_content == blobs[index]["blob_content"]
        assert (
            doc.metadata["source"]
            == f"{account_url}/{container_name}/{blobs[index]['blob_name']}"
        )


def test_get_blob_client(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    mock_container_client: Tuple[MagicMock, MagicMock],
) -> None:
    _, mock_client = mock_container_client
    mock_client.list_blob_names.return_value = ["text_file.txt"]

    loader = create_azure_blob_storage_loader(prefix="text")
    list(loader.lazy_load())
    mock_client.get_blob_client.assert_called_once_with("text_file.txt")


def test_default_credential(
    mock_container_client: Tuple[MagicMock, MagicMock],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    mock_container_client_cls, _ = mock_container_client
    loader = create_azure_blob_storage_loader(blob_names="text_file.txt")
    list(loader.lazy_load())
    cred = mock_container_client_cls.call_args[1]["credential"]
    assert isinstance(cred, azure.identity.DefaultAzureCredential)


def test_override_credential(
    mock_container_client: Tuple[MagicMock, MagicMock],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.core.credentials import AzureSasCredential

    mock_container_client_cls, _ = mock_container_client
    mock_credential = AzureSasCredential("test_sas_token")
    loader = create_azure_blob_storage_loader(
        blob_names="text_file.txt", credential=mock_credential
    )
    list(loader.lazy_load())
    assert mock_container_client_cls.call_args[1]["credential"] is mock_credential


def test_async_credential_provided_to_sync(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.core.credentials_async import AsyncTokenCredential

    mock_credential = MagicMock(spec=AsyncTokenCredential)
    loader = create_azure_blob_storage_loader(
        blob_names="text_file.txt", credential=mock_credential
    )
    with pytest.raises(ValueError, match="Cannot use synchronous load"):
        list(loader.lazy_load())


def test_invalid_credential_type(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    mock_credential = MagicMock(spec=str)
    with pytest.raises(TypeError, match="Invalid credential type provided."):
        create_azure_blob_storage_loader(
            blob_names="text_file.txt", credential=mock_credential
        )


def test_both_blob_names_and_prefix_set(
    blobs: list[dict[str, str]],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    with pytest.raises(ValueError, match="Cannot specify both blob_names and prefix."):
        create_azure_blob_storage_loader(
            blob_names=[blob["blob_name"] for blob in blobs], prefix="text"
        )
