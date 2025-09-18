import os
import pytest
import azure.identity

from langchain_azure_storage import AzureBlobStorageLoader
from azure.storage.blob import BlobClient
from azure.core.credentials import AzureSasCredential
from azure.storage.blob import ContainerClient


_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
_CREDENTIAL = azure.identity.DefaultAzureCredential()
_BLOBS_LIST = [
    ("text_file.txt", b"test content"),
    ("text_file2.txt", b"test content 2"),
    ("json_file.json", b"{'test': 'test content'}"), 
    ("csv_file.csv", b"col1,col2\nval1,val2"), 
]


@pytest.fixture(scope="session", autouse=True)
def upload_blob_to_container():
    for blob_name, data in _BLOBS_LIST:
        blob_client = BlobClient.from_blob_url(
           f"{_ACCOUNT_URL}/{_CONTAINER_NAME}/{blob_name}", credential=_CREDENTIAL
        )
        blob_client.upload_blob(data, overwrite=True)

@pytest.mark.parametrize(
        "blob_names,expected_blobs,prefix",
        [
            ([], [], ""),
            ([], [], "text"),
            ("text_file.txt", ["test content"], ""),
            (
                ["text_file.txt", "json_file.json", "csv_file.csv"], 
                ["test content", "{'test': 'test content'}", "col1,col2\nval1,val2"],
                ""
            ),
            (
                ["text_file.txt", "text_file2.txt", "json_file.json", "csv_file.csv",],
                ["test content", "test content 2"],
                "text"
            ),
            (
                ["text_file_nonexistent.txt", "text_file.txt", "text_file2.txt",], 
                ["test content", "test content 2"], 
                "text"
            ),
        ]
)
def test_lazy_load(blob_names, expected_blobs, prefix):
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        blob_names,
        prefix,
    )
    for doc, expected_content in zip(loader.lazy_load(), expected_blobs):
        assert doc.page_content == expected_content


def test_default_credential(mocker):
    mock_default_credential = mocker.patch.object(azure.identity, "DefaultAzureCredential")
    mock_default_credential.return_value = mocker.MagicMock()
    mock_container_client = mocker.patch.object(ContainerClient, "from_container_url")

    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "text_file.txt",
        "",
    )
    list(loader.lazy_load())
    assert mock_container_client.call_args[1]["credential"] == mock_default_credential.return_value
    

def test_override_credential(mocker):
    mock_credential = mocker.MagicMock(spec=AzureSasCredential)
    mock_container_client = mocker.patch.object(ContainerClient, "from_container_url")
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "text_file.txt",
        "",
        credential=mock_credential
    )
    list(loader.lazy_load())
    assert mock_container_client.call_args[1]["credential"] == mock_credential


def test_invalid_credential_type(mocker):
    from azure.core.credentials_async import AsyncTokenCredential

    mock_credential = mocker.MagicMock(spec=AsyncTokenCredential)
    with pytest.raises(ValueError):
        loader = AzureBlobStorageLoader(
            _ACCOUNT_URL,
            _CONTAINER_NAME,
            "text_file.txt",
            "",
            credential=mock_credential
        )
        list(loader.lazy_load())


def test_document_metadata():
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "text_file.txt",
        "",
    )
    for doc in loader.lazy_load():
        assert doc.metadata["source"] == f"{_ACCOUNT_URL}/{_CONTAINER_NAME}/text_file.txt"
