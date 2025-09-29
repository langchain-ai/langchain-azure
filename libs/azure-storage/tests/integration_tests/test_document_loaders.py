import csv
import os
from typing import Any, Callable, Iterator, Optional, Union

import azure.identity
import pytest
from azure.storage.blob import BlobClient
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document

from langchain_azure_storage import AzureBlobStorageLoader

_CREDENTIAL = azure.identity.DefaultAzureCredential()


# This custom CSV loader follows the langchain_community.document_loaders.CSVLoader
# interface. We are not directly using it to avoid adding langchain_community as a
# dependency for this package.
class CustomCSVLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        content_columns: Optional[list[str]] = None,
    ):
        self.file_path = file_path
        self.content_columns = content_columns

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if self.content_columns is not None:
                    content = "\n".join(
                        f"{key}: {row[key]}"
                        for key in self.content_columns
                        if key in row
                    )
                else:
                    content = "\n".join(f"{key}: {value}" for key, value in row.items())
                yield Document(
                    page_content=content, metadata={"source": self.file_path}
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
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    if container_name is None:
        raise ValueError(
            "AZURE_STORAGE_CONTAINER_NAME environment variable must be set for "
            "integration tests."
        )
    return container_name


@pytest.fixture(scope="session")
def blobs() -> list[dict[str, str]]:
    return [
        {
            "blob_name": "csv_file.csv",
            "blob_content": "col1,col2\nval1,val2\nval3,val4",
        },
        {"blob_name": "json_file.json", "blob_content": "{'test': 'test content'}"},
        {"blob_name": "text_file.txt", "blob_content": "test content"},
        {"blob_name": "text_file2.txt", "blob_content": "test content 2"},
    ]


@pytest.fixture(scope="session", autouse=True)
def upload_blob_to_container(
    account_url: str, container_name: str, blobs: list[dict[str, str]]
) -> None:
    for blob in blobs:
        blob_client = BlobClient.from_blob_url(
            f"{account_url}/{container_name}/{blob["blob_name"]}",
            credential=_CREDENTIAL,
        )
        blob_client.upload_blob(blob["blob_content"], overwrite=True)


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
def expected_custom_csv_documents(
    account_url: str,
    container_name: str,
) -> list[Document]:
    return [
        Document(
            page_content="col1: val1\ncol2: val2",
            metadata={"source": f"{account_url}/{container_name}/csv_file.csv"},
        ),
        Document(
            page_content="col1: val3\ncol2: val4",
            metadata={"source": f"{account_url}/{container_name}/csv_file.csv"},
        ),
    ]


@pytest.fixture
def expected_custom_csv_documents_with_columns(
    account_url: str,
    container_name: str,
) -> list[Document]:
    return [
        Document(
            page_content="col1: val1",
            metadata={"source": f"{account_url}/{container_name}/csv_file.csv"},
        ),
        Document(
            page_content="col1: val3",
            metadata={"source": f"{account_url}/{container_name}/csv_file.csv"},
        ),
    ]


def get_expected_documents(
    blobs: list[dict[str, str]], account_url: str, container_name: str
) -> list[Document]:
    expected_documents_list = []
    for blob in blobs:
        expected_documents_list.append(
            Document(
                page_content=blob["blob_content"],
                metadata={
                    "source": f"{account_url}/{container_name}/{blob['blob_name']}"
                },
            )
        )
    return expected_documents_list


@pytest.mark.parametrize(
    "blob_names,expected_contents,prefix",
    [
        (
            [],
            [],
            None,
        ),
        (
            None,
            [
                {
                    "blob_name": "csv_file.csv",
                    "blob_content": "col1,col2\nval1,val2\nval3,val4",
                },
                {
                    "blob_name": "json_file.json",
                    "blob_content": "{'test': 'test content'}",
                },
                {"blob_name": "text_file.txt", "blob_content": "test content"},
                {"blob_name": "text_file2.txt", "blob_content": "test content 2"},
            ],
            None,
        ),
        (
            None,
            [
                {"blob_name": "text_file.txt", "blob_content": "test content"},
                {"blob_name": "text_file2.txt", "blob_content": "test content 2"},
            ],
            "text",
        ),
        (
            "text_file.txt",
            [{"blob_name": "text_file.txt", "blob_content": "test content"}],
            None,
        ),
        (
            ["text_file.txt", "json_file.json", "csv_file.csv"],
            [
                {"blob_name": "text_file.txt", "blob_content": "test content"},
                {
                    "blob_name": "json_file.json",
                    "blob_content": "{'test': 'test content'}",
                },
                {
                    "blob_name": "csv_file.csv",
                    "blob_content": "col1,col2\nval1,val2\nval3,val4",
                },
            ],
            None,
        ),
    ],
)
def test_lazy_load(
    blob_names: Optional[Union[str, list[str]]],
    expected_contents: list[dict[str, str]],
    prefix: Optional[str],
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    loader = create_azure_blob_storage_loader(blob_names=blob_names, prefix=prefix)
    expected_documents_list = get_expected_documents(
        expected_contents, account_url, container_name
    )
    assert list(loader.lazy_load()) == expected_documents_list


def test_lazy_load_with_credential(
    blobs: list[dict[str, str]],
    account_url: str,
    container_name: str,
    create_azure_blob_storage_loader: Callable[..., AzureBlobStorageLoader],
) -> None:
    from azure.core.credentials import AzureSasCredential

    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    if sas_token is None:
        raise ValueError(
            "AZURE_STORAGE_SAS_TOKEN environment variable must be set for "
            "integration tests."
        )
    loader = create_azure_blob_storage_loader(credential=AzureSasCredential(sas_token))
    expected_documents_list = get_expected_documents(blobs, account_url, container_name)
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
    def custom_loader_factory(file_path: str) -> CustomCSVLoader:
        return CustomCSVLoader(file_path=file_path, content_columns=["col1"])

    loader = create_azure_blob_storage_loader(
        blob_names="csv_file.csv",
        loader_factory=custom_loader_factory,
    )
    assert list(loader.lazy_load()) == expected_custom_csv_documents_with_columns
