from typing import Callable, Iterable, Iterator, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import azure.identity
import pytest
from azure.storage.blob import ContainerClient

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

_SDK_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.TokenCredential,
    ]
]


@pytest.fixture
def account_url() -> str:
    return "https://testaccount.blob.core.windows.net"


@pytest.fixture
def container_name() -> str:
    return "test-container"


@pytest.fixture
def blob_names() -> list[str]:
    return ["text_file.txt", "json_file.json", "csv_file.csv"]


@pytest.fixture
def blob_contents() -> list[str]:
    return [
        "test content",
        "{'test': 'test content'}",
        "col1,col2\nval1,val2",
    ]


@pytest.fixture
def create_azure_blob_storage_loader(
    account_url: str, container_name: str
) -> Callable[..., AzureBlobStorageLoader]:
    def _create_azure_blob_storage_loader(
        account_url: str = account_url,
        container_name: str = container_name,
        blob_names: Optional[Union[str, Iterable[str]]] = None,
        prefix: Optional[str] = None,
        credential: _SDK_CREDENTIAL_TYPE = None,
    ) -> AzureBlobStorageLoader:
        return AzureBlobStorageLoader(
            account_url,
            container_name,
            blob_names,
            prefix=prefix,
            credential=credential,
        )

    return _create_azure_blob_storage_loader


@pytest.fixture(autouse=True)
def mock_container_client(
    blob_names: list[str],
    blob_contents: list[str],
    account_url: str,
    container_name: str,
) -> Iterator[Tuple[MagicMock, MagicMock]]:
    with patch(
        "langchain_azure_storage.document_loaders.ContainerClient"
    ) as mock_container_client_cls:
        mock_client = MagicMock(spec=ContainerClient)
        mock_client.list_blob_names.return_value = blob_names

        def get_blob_client(blob_name: str) -> MagicMock:
            mock_blob_client = MagicMock()
            mock_blob_client.url = f"{account_url}/{container_name}/{blob_name}"
            mock_blob_data = MagicMock()
            blob_index = blob_names.index(blob_name)
            mock_blob_data.readall.return_value = blob_contents[blob_index].encode(
                "utf-8"
            )
            mock_blob_client.download_blob.return_value = mock_blob_data
            return mock_blob_client

        mock_client.get_blob_client.side_effect = get_blob_client
        mock_container_client_cls.return_value = mock_client
        yield mock_container_client_cls, mock_client


def test_lazy_load(
    account_url: str,
    container_name: str,
    blob_names: list[str],
    blob_contents: list[str],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader()

    for doc, expected_content, blob_name in zip(
        loader.lazy_load(), blob_contents, blob_names
    ):
        assert doc.page_content == expected_content
        assert doc.metadata["source"] == f"{account_url}/{container_name}/{blob_name}"


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
    with patch.object(
        azure.identity, "DefaultAzureCredential"
    ) as mock_default_credential:
        mock_container_client_cls, _ = mock_container_client
        mock_default_credential.return_value = MagicMock()
        loader = create_azure_blob_storage_loader(blob_names="text_file.txt")
        list(loader.lazy_load())
        cred = mock_container_client_cls.call_args[1]["credential"]
        assert cred == mock_default_credential.return_value


def test_override_credential(
    mock_container_client: Tuple[MagicMock, MagicMock],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.core.credentials import AzureSasCredential

    mock_container_client_cls, _ = mock_container_client
    mock_credential = MagicMock(spec=AzureSasCredential)
    loader = create_azure_blob_storage_loader(
        blob_names="text_file.txt", credential=mock_credential
    )

    list(loader.lazy_load())
    assert mock_container_client_cls.call_args[1]["credential"] == mock_credential


def test_async_credential_provided_to_sync(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.core.credentials_async import AsyncTokenCredential

    mock_credential = MagicMock(spec=AsyncTokenCredential)
    with pytest.raises(TypeError):
        loader = create_azure_blob_storage_loader(
            blob_names="text_file.txt", credential=mock_credential
        )
        list(loader.lazy_load())


def test_invalid_credential_type(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    mock_credential = MagicMock(spec=str)
    with pytest.raises(TypeError):
        loader = create_azure_blob_storage_loader(
            blob_names="text_file.txt", credential=mock_credential
        )
        list(loader.lazy_load())


def test_both_blob_names_and_prefix_set(
    blob_names: list[str],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    with pytest.raises(ValueError):
        loader = create_azure_blob_storage_loader(blob_names=blob_names, prefix="text")
        list(loader.lazy_load())
