from langchain_azure_dynamic_sessions import __all__

EXPECTED_ALL = [
    "SessionsBashTool",
    "SessionsPythonREPLTool",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_sessions_bash_backend_importable() -> None:
    """SessionsBashBackend should be importable without deepagents installed."""
    from langchain_azure_dynamic_sessions.backends import SessionsBashBackend

    assert SessionsBashBackend is not None
