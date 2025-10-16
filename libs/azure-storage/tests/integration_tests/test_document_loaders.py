import os
from typing import Callable, Iterator, Optional, Union

import pytest
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from langchain_core.documents.base import Document

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from tests.utils import (
    CustomCSVLoader,
    get_expected_documents,
    get_first_column_csv_loader,
    get_test_blobs,
)


@pytest.fixture(scope="session")
def account_url() -> str:
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    if account_url is None:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_URL environment variable must be set for "
            "integration tests."
        )
    return account_url


@pytest.fixture(scope="session")
def container_name() -> str:
    return "document-loader-tests"


@pytest.fixture(scope="session")
def blob_service_client(account_url: str) -> BlobServiceClient:
    return BlobServiceClient(
        account_url=account_url, credential=DefaultAzureCredential()
    )


@pytest.fixture(scope="session", autouse=True)
def container_client(
    blob_service_client: BlobServiceClient, container_name: str
) -> Iterator[ContainerClient]:
    container_client = blob_service_client.get_container_client(container_name)
    container_client.create_container()
    yield container_client
    container_client.delete_container()


@pytest.fixture(scope="session", autouse=True)
def upload_blobs_to_container(
    container_client: ContainerClient,
) -> None:
    for blob in get_test_blobs():
        blob_client = container_client.get_blob_client(blob["blob_name"])
        blob_client.upload_blob(blob["blob_content"], overwrite=True)


@pytest.mark.parametrize(
    "blob_names,prefix",
    [
        ([], None),
        (None, None),
        (None, "text"),
        ("text_file.txt", None),
        (["text_file.txt", "json_file.json", "csv_file.csv"], None),
    ],
)
def test_lazy_load(
    blob_names: Optional[Union[str, list[str]]],
    prefix: Optional[str],
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader(blob_names=blob_names, prefix=prefix)
    expected_documents_list = get_expected_documents(
        get_test_blobs(blob_names, prefix), account_url, container_name
    )
    assert list(loader.lazy_load()) == expected_documents_list


def test_lazy_load_with_loader_factory(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv",
        loader_factory=CustomCSVLoader,
    )
    assert list(loader.lazy_load()) == expected_custom_csv_documents


def test_lazy_load_with_loader_factory_configurations(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents_with_columns: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv",
        loader_factory=get_first_column_csv_loader,
    )
    assert list(loader.lazy_load()) == expected_custom_csv_documents_with_columns


@pytest.mark.parametrize(
    "blob_names,prefix",
    [
        ([], None),
        (None, None),
        (None, "text"),
        ("text_file.txt", None),
        (["text_file.txt", "json_file.json", "csv_file.csv"], None),
    ],
)
async def test_alazy_load(
    blob_names: Optional[Union[str, list[str]]],
    prefix: Optional[str],
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader(blob_names=blob_names, prefix=prefix)
    expected_documents_list = get_expected_documents(
        get_test_blobs(blob_names, prefix), account_url, container_name
    )
    assert [doc async for doc in loader.alazy_load()] == expected_documents_list


async def test_alazy_load_with_loader_factory(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv",
        loader_factory=CustomCSVLoader,
    )
    assert [doc async for doc in loader.alazy_load()] == expected_custom_csv_documents


async def test_alazy_load_with_loader_factory_configurations(
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
    expected_custom_csv_documents_with_columns: list[Document],
) -> None:
    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv",
        loader_factory=get_first_column_csv_loader,
    )
    assert [
        doc async for doc in loader.alazy_load()
    ] == expected_custom_csv_documents_with_columns
