from langchain_sqlserver import __all__

EXPECTED_ALL = [
    "SQLServerChatMessageHistory",
    "SQLServerSaver",
    "SQLServerVectorStore",
    "SQLServer_VectorStore",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_deprecated_alias_is_subclass() -> None:
    """The deprecated alias should subclass the renamed class."""
    from langchain_sqlserver import SQLServer_VectorStore, SQLServerVectorStore

    assert issubclass(SQLServer_VectorStore, SQLServerVectorStore)
    assert SQLServer_VectorStore is not SQLServerVectorStore
