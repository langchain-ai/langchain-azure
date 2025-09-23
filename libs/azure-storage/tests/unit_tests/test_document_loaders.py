import azure.identity
import pytest
from azure.storage.blob import ContainerClient
from pytest_mock import MockerFixture

from langchain_azure_storage import AzureBlobStorageLoader


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
def mock_lazy_load_documents_from_blob(
    mocker: MockerFixture, blob_names: list[str], blob_contents: list[str]
) -> None:
    from langchain_core.documents.base import Document

    docs = []
    for name, content in zip(blob_names, blob_contents):
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": f"https://testaccount.blob.core.windows.net/test-container/{name}"
                },
            )
        )

    mocker.patch(
        "langchain_azure_storage.document_loaders._lazy_load_documents_from_blob",
        return_value=iter(docs),
    )


def test_lazy_load(
    mocker: MockerFixture,
    blob_names: list[str],
    blob_contents: list[str],
    mock_lazy_load_documents_from_blob: None,
) -> None:
    mock_container_client = mocker.patch.object(ContainerClient, "from_container_url")
    mock_client = mocker.MagicMock()
    mock_client.list_blob_names.return_value = blob_names
    mock_container_client.return_value = mock_client

    loader = AzureBlobStorageLoader(
        "https://testaccount.blob.core.windows.net",
        "test-container",
        blob_names,
        "",
    )
    for doc, expected_content, blob_name in zip(
        loader.lazy_load(), blob_contents, blob_names
    ):
        assert doc.page_content == expected_content
        assert (
            doc.metadata["source"]
            == f"https://testaccount.blob.core.windows.net/test-container/{blob_name}"
        )


def test_get_blob_client(
    mocker: MockerFixture,
    blob_names: list[str],
    mock_lazy_load_documents_from_blob: None,
) -> None:
    mock_container_client = mocker.patch.object(ContainerClient, "from_container_url")
    mock_client = mocker.MagicMock()
    mock_client.list_blob_names.return_value = ["text_file.txt", "text_file2.txt"]
    mock_container_client.return_value = mock_client

    mock_blob_client = mock_client.get_blob_client

    loader = AzureBlobStorageLoader(
        "https://testaccount.blob.core.windows.net",
        "test-container",
        blob_names,
        "text",
    )
    list(loader.lazy_load())
    mock_blob_client.assert_called_once_with("text_file.txt")


def test_default_credential(mocker: MockerFixture) -> None:
    mock_default_credential = mocker.patch.object(
        azure.identity, "DefaultAzureCredential"
    )
    mock_default_credential.return_value = mocker.MagicMock()
    mock_container_client = mocker.patch.object(ContainerClient, "from_container_url")

    loader = AzureBlobStorageLoader(
        "https://testaccount.blob.core.windows.net",
        "test-container",
        "text_file.txt",
        "",
    )
    list(loader.lazy_load())
    cred = mock_container_client.call_args[1]["credential"]
    assert cred == mock_default_credential.return_value


def test_override_credential(mocker: MockerFixture) -> None:
    from azure.core.credentials import AzureSasCredential

    mock_credential = mocker.MagicMock(spec=AzureSasCredential)
    mock_container_client = mocker.patch.object(ContainerClient, "from_container_url")
    loader = AzureBlobStorageLoader(
        "https://testaccount.blob.core.windows.net",
        "test-container",
        "text_file.txt",
        "",
        credential=mock_credential,
    )
    list(loader.lazy_load())
    assert mock_container_client.call_args[1]["credential"] == mock_credential


def test_async_credential_provided_to_sync(mocker: MockerFixture) -> None:
    from azure.core.credentials_async import AsyncTokenCredential

    mock_credential = mocker.MagicMock(spec=AsyncTokenCredential)
    with pytest.raises(ValueError):
        loader = AzureBlobStorageLoader(
            "https://testaccount.blob.core.windows.net",
            "test-container",
            "text_file.txt",
            "",
            credential=mock_credential,
        )
        list(loader.lazy_load())


def test_invalid_credential_type(mocker: MockerFixture) -> None:
    mock_credential = mocker.MagicMock(spec=str)
    with pytest.raises(TypeError):
        loader = AzureBlobStorageLoader(
            "https://testaccount.blob.core.windows.net",
            "test-container",
            "text_file.txt",
            "",
            credential=mock_credential,
        )
        list(loader.lazy_load())
