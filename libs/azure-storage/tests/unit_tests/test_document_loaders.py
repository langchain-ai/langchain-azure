from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch

import azure.identity
import azure.identity.aio
import pytest
from azure.storage.blob import BlobClient, ContainerClient
from azure.storage.blob._download import StorageStreamDownloader
from azure.storage.blob.aio import BlobClient as AsyncBlobClient
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from azure.storage.blob.aio import (
    StorageStreamDownloader as AsyncStorageStreamDownloader,
)
from langchain_core.documents.base import Document

from langchain_azure_storage import __version__
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from tests.utils import (
    CustomCSVLoader,
    get_datalake_test_blobs,
    get_expected_documents,
    get_first_column_csv_loader,
    get_test_blobs,
    get_test_mock_blobs,
)


@pytest.fixture
def account_url() -> str:
    return "https://testaccount.blob.core.windows.net"


@pytest.fixture
def container_name() -> str:
    return "test-container"


@pytest.fixture
def get_mock_blob_client(
    account_url: str, container_name: str
) -> Callable[[str], MagicMock]:
    def _get_blob_client(blob_name: str) -> MagicMock:
        mock_blob_client = MagicMock(spec=BlobClient)
        mock_blob_client.url = f"{account_url}/{container_name}/{blob_name}"
        mock_blob_client.blob_name = blob_name
        mock_blob_data = MagicMock(spec=StorageStreamDownloader)
        content = next(
            blob["blob_content"]
            for blob in get_test_blobs()
            if blob["blob_name"] == blob_name
        )
        mock_blob_data.readall.return_value = content.encode("utf-8")
        mock_blob_client.download_blob.return_value = mock_blob_data
        return mock_blob_client

    return _get_blob_client


@pytest.fixture(autouse=True)
def mock_container_client(
    get_mock_blob_client: Callable[[str], MagicMock],
) -> Iterator[Tuple[MagicMock, MagicMock]]:
    with patch(
        "langchain_azure_storage.document_loaders.ContainerClient"
    ) as mock_container_client_cls:
        mock_client = MagicMock(spec=ContainerClient)
        mock_client.list_blobs.return_value = get_test_mock_blobs(get_test_blobs())
        mock_client.get_blob_client.side_effect = get_mock_blob_client
        mock_container_client_cls.return_value = mock_client
        yield mock_container_client_cls, mock_client


@pytest.fixture
def get_mock_datalake_blob_client(
    account_url: str, container_name: str
) -> Callable[[str], MagicMock]:
    def _get_blob_client(blob_name: str) -> MagicMock:
        mock_blob_client = MagicMock(spec=BlobClient)
        mock_blob_client.url = f"{account_url}/{container_name}/{blob_name}"
        mock_blob_client.blob_name = blob_name
        mock_blob_data = MagicMock(spec=StorageStreamDownloader)
        content = next(
            blob["blob_content"]
            for blob in get_datalake_test_blobs(include_directories=True)
            if blob["blob_name"] == blob_name
        )
        mock_blob_data.readall.return_value = content.encode("utf-8")
        mock_blob_client.download_blob.return_value = mock_blob_data
        return mock_blob_client

    return _get_blob_client


@pytest.fixture
def mock_datalake_container_client(
    get_mock_datalake_blob_client: Callable[[str], MagicMock],
) -> Iterator[Tuple[MagicMock, MagicMock]]:
    with patch(
        "langchain_azure_storage.document_loaders.ContainerClient"
    ) as mock_container_client_cls:
        mock_client = MagicMock(spec=ContainerClient)
        mock_client.list_blobs.return_value = get_test_mock_blobs(
            get_datalake_test_blobs(include_directories=True)
        )
        mock_client.get_blob_client.side_effect = get_mock_datalake_blob_client
        mock_container_client_cls.return_value = mock_client
        yield mock_container_client_cls, mock_client


@pytest.fixture()
def get_async_mock_blob_client(
    account_url: str, container_name: str
) -> Callable[[str], AsyncMock]:
    def _get_async_blob_client(blob_name: str) -> AsyncMock:
        async_mock_blob_client = AsyncMock(spec=AsyncBlobClient)
        async_mock_blob_client.url = f"{account_url}/{container_name}/{blob_name}"
        async_mock_blob_client.blob_name = blob_name
        mock_blob_data = AsyncMock(spec=AsyncStorageStreamDownloader)
        content = next(
            blob["blob_content"]
            for blob in get_test_blobs()
            if blob["blob_name"] == blob_name
        )
        mock_blob_data.readall.return_value = content.encode("utf-8")
        async_mock_blob_client.download_blob.return_value = mock_blob_data
        return async_mock_blob_client

    return _get_async_blob_client


@pytest.fixture(autouse=True)
def async_mock_container_client(
    get_async_mock_blob_client: Callable[[str], AsyncMock],
) -> Iterator[Tuple[AsyncMock, AsyncMock]]:
    with patch(
        "langchain_azure_storage.document_loaders.AsyncContainerClient"
    ) as async_mock_container_client_cls:

        async def get_async_blobs(**kwargs: Any) -> AsyncIterator[MagicMock]:
            prefix = kwargs.get("name_starts_with")
            for mock_blob in get_test_mock_blobs(get_test_blobs(prefix=prefix)):
                yield mock_blob

        async_mock_client = AsyncMock(spec=AsyncContainerClient)
        async_mock_client.list_blobs.side_effect = get_async_blobs
        async_mock_client.get_blob_client.side_effect = get_async_mock_blob_client
        async_mock_container_client_cls.return_value = async_mock_client
        yield async_mock_container_client_cls, async_mock_client


@pytest.fixture
def get_async_mock_datalake_blob_client(
    account_url: str, container_name: str
) -> Callable[[str], AsyncMock]:
    def _get_async_blob_client(blob_name: str) -> AsyncMock:
        async_mock_blob_client = AsyncMock(spec=AsyncBlobClient)
        async_mock_blob_client.url = f"{account_url}/{container_name}/{blob_name}"
        async_mock_blob_client.blob_name = blob_name
        mock_blob_data = AsyncMock(spec=AsyncStorageStreamDownloader)
        content = next(
            blob["blob_content"]
            for blob in get_datalake_test_blobs(include_directories=True)
            if blob["blob_name"] == blob_name
        )
        mock_blob_data.readall.return_value = content.encode("utf-8")
        async_mock_blob_client.download_blob.return_value = mock_blob_data
        return async_mock_blob_client

    return _get_async_blob_client


@pytest.fixture
def async_mock_datalake_container_client(
    get_async_mock_datalake_blob_client: Callable[[str], AsyncMock],
) -> Iterator[Tuple[AsyncMock, AsyncMock]]:
    with patch(
        "langchain_azure_storage.document_loaders.AsyncContainerClient"
    ) as async_mock_container_client_cls:

        async def get_async_blobs(**kwargs: Any) -> AsyncIterator[MagicMock]:
            prefix = kwargs.get("name_starts_with")
            for mock_blob in get_test_mock_blobs(
                get_datalake_test_blobs(prefix=prefix, include_directories=True)
            ):
                yield mock_blob

        async_mock_client = AsyncMock(spec=AsyncContainerClient)
        async_mock_client.list_blobs.side_effect = get_async_blobs
        async_mock_client.get_blob_client.side_effect = (
            get_async_mock_datalake_blob_client
        )
        async_mock_container_client_cls.return_value = async_mock_client
        yield async_mock_container_client_cls, async_mock_client


def test_lazy_load(
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader()
    expected_document_list = get_expected_documents(
        get_test_blobs(), account_url, container_name
    )
    assert list(loader.lazy_load()) == expected_document_list


@pytest.mark.parametrize(
    "blob_names",
    [
        "directory/test_file.txt",
        ["directory/test_file.txt", "json_file.json"],
    ],
)
def test_lazy_load_with_blob_names(
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    mock_container_client: Tuple[MagicMock, MagicMock],
    blob_names: Union[str, Iterable[str]],
) -> None:
    _, mock_client = mock_container_client
    loader = create_azure_blob_storage_loader(blob_names=blob_names)
    expected_documents_list = get_expected_documents(
        get_test_blobs(blob_names), account_url, container_name
    )
    assert list(loader.lazy_load()) == expected_documents_list
    assert mock_client.list_blobs.call_count == 0


def test_get_blob_client(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    mock_container_client: Tuple[MagicMock, MagicMock],
) -> None:
    _, mock_client = mock_container_client
    mock_client.list_blobs.return_value = get_test_mock_blobs(
        get_test_blobs(blob_names=["json_file.json"])
    )
    loader = create_azure_blob_storage_loader(prefix="json")
    list(loader.lazy_load())
    mock_client.get_blob_client.assert_called_once_with("json_file.json")
    mock_client.list_blobs.assert_called_once_with(
        name_starts_with="json", include="metadata"
    )


def test_default_credential(
    mock_container_client: Tuple[MagicMock, MagicMock],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    mock_container_client_cls, _ = mock_container_client
    loader = create_azure_blob_storage_loader(blob_names="directory/test_file.txt")
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
        blob_names="directory/test_file.txt", credential=mock_credential
    )
    list(loader.lazy_load())
    assert mock_container_client_cls.call_args[1]["credential"] is mock_credential


def test_async_credential_provided_to_sync(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.identity.aio import DefaultAzureCredential

    mock_credential = DefaultAzureCredential()
    loader = create_azure_blob_storage_loader(
        blob_names="directory/test_file.txt", credential=mock_credential
    )
    with pytest.raises(ValueError, match="Cannot use synchronous load"):
        list(loader.lazy_load())


def test_invalid_credential_type(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    mock_credential = "account-key"
    with pytest.raises(TypeError, match="Invalid credential type provided."):
        create_azure_blob_storage_loader(
            blob_names="directory/test_file.txt", credential=mock_credential
        )


def test_both_blob_names_and_prefix_set(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    with pytest.raises(ValueError, match="Cannot specify both blob_names and prefix."):
        create_azure_blob_storage_loader(
            blob_names=[blob["blob_name"] for blob in get_test_blobs()], prefix="json"
        )


def test_custom_loader_factory(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv", loader_factory=CustomCSVLoader
    )
    assert list(loader.lazy_load()) == expected_custom_csv_documents


def test_custom_loader_factory_with_configurations(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents_with_columns: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv", loader_factory=get_first_column_csv_loader
    )
    assert list(loader.lazy_load()) == expected_custom_csv_documents_with_columns


async def test_alazy_load(
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader()
    expected_document_list = get_expected_documents(
        get_test_blobs(), account_url, container_name
    )
    assert [doc async for doc in loader.alazy_load()] == expected_document_list


@pytest.mark.parametrize(
    "blob_names",
    [
        "directory/test_file.txt",
        ["directory/test_file.txt", "json_file.json"],
    ],
)
async def test_alazy_load_with_blob_names(
    blob_names: Union[str, Iterable[str]],
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    async_mock_container_client: Tuple[AsyncMock, AsyncMock],
) -> None:
    _, async_mock_client = async_mock_container_client
    loader = create_azure_blob_storage_loader(blob_names=blob_names)
    expected_documents_list = get_expected_documents(
        get_test_blobs(blob_names), account_url, container_name
    )
    assert [doc async for doc in loader.alazy_load()] == expected_documents_list
    assert async_mock_client.list_blobs.call_count == 0


async def test_get_async_blob_client(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    async_mock_container_client: Tuple[AsyncMock, AsyncMock],
) -> None:
    _, async_mock_client = async_mock_container_client
    loader = create_azure_blob_storage_loader(prefix="json")
    [doc async for doc in loader.alazy_load()]
    async_mock_client.get_blob_client.assert_called_once_with("json_file.json")
    async_mock_client.list_blobs.assert_called_once_with(
        name_starts_with="json", include="metadata"
    )


async def test_async_token_credential(
    async_mock_container_client: Tuple[AsyncMock, AsyncMock],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.core.credentials_async import AsyncTokenCredential

    async_mock_container_client_cls, _ = async_mock_container_client
    mock_credential = AsyncMock(spec=AsyncTokenCredential)
    loader = create_azure_blob_storage_loader(
        blob_names="json_file.json", credential=mock_credential
    )
    [doc async for doc in loader.alazy_load()]
    assert async_mock_container_client_cls.call_args[1]["credential"] is mock_credential


async def test_default_async_credential(
    async_mock_container_client: Tuple[AsyncMock, AsyncMock],
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    async_mock_container_client_cls, _ = async_mock_container_client
    loader = create_azure_blob_storage_loader(blob_names="json_file.json")
    [doc async for doc in loader.alazy_load()]
    cred = async_mock_container_client_cls.call_args[1]["credential"]
    assert isinstance(cred, azure.identity.aio.DefaultAzureCredential)


async def test_sync_credential_provided_to_async(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.identity import DefaultAzureCredential

    loader = create_azure_blob_storage_loader(
        blob_names="json_file.json", credential=DefaultAzureCredential()
    )
    with pytest.raises(ValueError, match="Cannot use asynchronous load"):
        [doc async for doc in loader.alazy_load()]


async def test_async_custom_loader_factory(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv", loader_factory=CustomCSVLoader
    )
    assert [doc async for doc in loader.alazy_load()] == expected_custom_csv_documents


async def test_async_custom_loader_factory_with_configurations(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents_with_columns: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv", loader_factory=get_first_column_csv_loader
    )
    assert [
        doc async for doc in loader.alazy_load()
    ] == expected_custom_csv_documents_with_columns


def test_user_agent(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    mock_container_client: Tuple[MagicMock, MagicMock],
) -> None:
    mock_container_client_cls, _ = mock_container_client
    user_agent = f"azpartner-langchain/{__version__}"
    loader = create_azure_blob_storage_loader(blob_names="json_file.json")
    list(loader.lazy_load())
    client_kwargs = mock_container_client_cls.call_args[1]
    assert client_kwargs["user_agent"] == user_agent


async def test_async_user_agent(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    async_mock_container_client: Tuple[AsyncMock, AsyncMock],
) -> None:
    async_mock_container_client_cls, _ = async_mock_container_client
    user_agent = f"azpartner-langchain/{__version__}"
    loader = create_azure_blob_storage_loader(blob_names="json_file.json")
    [doc async for doc in loader.alazy_load()]
    client_kwargs = async_mock_container_client_cls.call_args[1]
    assert client_kwargs["user_agent"] == user_agent


def test_datalake_excludes_directories(
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    mock_datalake_container_client: Tuple[MagicMock, MagicMock],
) -> None:
    loader = create_azure_blob_storage_loader()
    expected_documents = get_expected_documents(
        get_datalake_test_blobs(), account_url, container_name
    )
    assert list(loader.lazy_load()) == expected_documents


async def test_async_datalake_excludes_directories(
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    async_mock_datalake_container_client: Tuple[AsyncMock, AsyncMock],
) -> None:
    loader = create_azure_blob_storage_loader()
    expected_documents = get_expected_documents(
        get_datalake_test_blobs(), account_url, container_name
    )
    assert [doc async for doc in loader.alazy_load()] == expected_documents
