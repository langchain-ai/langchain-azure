import csv
from typing import Iterable, Iterator, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document

_TEST_BLOBS = [
    {
        "blob_name": "csv_file.csv",
        "blob_content": "col1,col2\nval1,val2\nval3,val4",
    },
    {"blob_name": "json_file.json", "blob_content": "{'test': 'test content'}"},
    {"blob_name": "text_file.txt", "blob_content": "test content"},
]


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


def get_test_blobs(
    blob_names: Optional[Union[str, Iterable[str]]] = None, prefix: Optional[str] = None
) -> list[dict[str, str]]:
    if blob_names is not None:
        if isinstance(blob_names, str):
            blob_names = [blob_names]
        updated_list = []
        for name in blob_names:
            for blob in _TEST_BLOBS:
                if blob["blob_name"] == name:
                    updated_list.append(blob)
                    break
        return updated_list

    if prefix is not None:
        return [blob for blob in _TEST_BLOBS if blob["blob_name"].startswith(prefix)]

    return _TEST_BLOBS
