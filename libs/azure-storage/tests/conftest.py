from typing import Any, Callable

import pytest
from langchain_core.documents.base import Document

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader


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


# For the following expected csv document fixtures, the page content comes from
# the tests.utils._TEST_BLOBS list.
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
