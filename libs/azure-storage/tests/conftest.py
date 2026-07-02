import importlib.util
from typing import Any, Callable

import pytest
from langchain_core.documents.base import Document

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

# The Deep Agents backend lives behind the optional ``[deepagents]`` extra,
# which is only installed on Python >= 3.11. When ``deepagents`` is absent
# (e.g. the Python 3.10 CI job), skip collecting its test modules so importing
# them does not fail.
if importlib.util.find_spec("deepagents") is None:
    collect_ignore_glob = ["**/test_deepagents_backend*.py"]


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
