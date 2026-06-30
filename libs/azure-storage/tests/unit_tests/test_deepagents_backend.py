"""Unit tests for AzureBlobBackend (mocked, no I/O)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from deepagents.backends.protocol import (
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from langchain_azure_storage.deepagents import AzureBlobBackend, AzureBlobConfig
from langchain_azure_storage.deepagents._path import (
    from_blob_key,
    get_prefix_for_path,
    normalize_path,
    to_blob_key,
)
from langchain_azure_storage.deepagents._utils import build_file_info

# Module path used as the target for patching imported Azure SDK symbols.
_BACKEND = "langchain_azure_storage.deepagents.backend"

# ------------------------------------------------------------------
# Path utility tests
# ------------------------------------------------------------------


class TestNormalizePath:
    def test_strips_leading_slash(self) -> None:
        assert normalize_path("/src/main.py") == "src/main.py"

    def test_no_leading_slash(self) -> None:
        assert normalize_path("src/main.py") == "src/main.py"

    def test_root(self) -> None:
        assert normalize_path("/") == ""

    def test_double_slashes(self) -> None:
        assert normalize_path("//src//main.py") == "src/main.py"

    def test_trailing_slash(self) -> None:
        assert normalize_path("/src/") == "src"

    def test_empty_string(self) -> None:
        assert normalize_path("") == ""

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="Path traversal"):
            normalize_path("/src/../secrets.txt")

    def test_rejects_windows_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="Windows absolute paths"):
            normalize_path("C:/temp/file.txt")


class TestToBlobKey:
    def test_with_prefix(self) -> None:
        assert to_blob_key("workspace/", "/src/main.py") == "workspace/src/main.py"

    def test_without_prefix(self) -> None:
        assert to_blob_key("", "/src/main.py") == "src/main.py"

    def test_prefix_no_trailing_slash(self) -> None:
        assert to_blob_key("workspace", "/src/main.py") == "workspace/src/main.py"


class TestFromBlobKey:
    def test_with_prefix(self) -> None:
        assert from_blob_key("workspace/", "workspace/src/main.py") == "/src/main.py"

    def test_without_prefix(self) -> None:
        assert from_blob_key("", "src/main.py") == "/src/main.py"

    def test_prefix_no_trailing_slash(self) -> None:
        assert from_blob_key("workspace", "workspace/src/main.py") == "/src/main.py"


class TestGetPrefixForPath:
    def test_root_with_prefix(self) -> None:
        assert get_prefix_for_path("workspace/", "/") == "workspace/"

    def test_subdir_with_prefix(self) -> None:
        assert get_prefix_for_path("workspace/", "/src") == "workspace/src/"

    def test_root_no_prefix(self) -> None:
        assert get_prefix_for_path("", "/") == ""

    def test_subdir_no_prefix(self) -> None:
        assert get_prefix_for_path("", "/src") == "src/"


# ------------------------------------------------------------------
# Config tests
# ------------------------------------------------------------------


class TestAzureBlobConfig:
    def test_defaults_with_account_url(self) -> None:
        config = AzureBlobConfig(account_url="https://x.blob.core.windows.net")
        assert config.account_url == "https://x.blob.core.windows.net"
        assert config.container_name == ""
        assert config.prefix == ""
        assert config.credential is None
        assert config.account_key is None
        assert config.sas_token is None
        assert config.max_concurrency == 8
        assert config.encoding == "utf-8"
        assert config.connection_string is None

    def test_defaults_with_connection_string(self) -> None:
        conn_str = (
            "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"
        )
        config = AzureBlobConfig(connection_string=conn_str)
        assert config.connection_string is not None
        assert config.account_url == ""

    def test_no_account_url_and_no_connection_string_raises(self) -> None:
        with pytest.raises(ValueError, match="account_url is required"):
            AzureBlobConfig()

    def test_custom_values(self) -> None:
        config = AzureBlobConfig(
            account_url="https://myaccount.blob.core.windows.net",
            container_name="mycontainer",
            prefix="agent-1/",
            max_concurrency=4,
        )
        assert config.account_url == "https://myaccount.blob.core.windows.net"
        assert config.container_name == "mycontainer"
        assert config.prefix == "agent-1/"
        assert config.max_concurrency == 4

    def test_account_key_field(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            account_key="my-key",
        )
        assert config.account_key == "my-key"

    def test_sas_token_field(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            sas_token="sv=2021-06-08&ss=b&srt=co&sp=rwdlacitfx&se=2030-01-01",
        )
        assert config.sas_token is not None

    def test_conflict_account_key_and_sas_token(self) -> None:
        with pytest.raises(ValueError, match="Only one authentication method"):
            AzureBlobConfig(
                account_url="https://x.blob.core.windows.net",
                account_key="my-key",
                sas_token="my-sas",
            )

    def test_conflict_connection_string_and_account_key(self) -> None:
        with pytest.raises(ValueError, match="Only one authentication method"):
            AzureBlobConfig(
                connection_string=(
                    "DefaultEndpointsProtocol=https;AccountName=fake;"
                    "AccountKey=ZmFrZQ==;"
                ),
                account_key="my-key",
            )

    def test_conflict_credential_and_sas_token(self) -> None:
        with pytest.raises(ValueError, match="Only one authentication method"):
            AzureBlobConfig(
                account_url="https://x.blob.core.windows.net",
                credential=object(),
                sas_token="my-sas",
            )

    def test_account_key_requires_account_url(self) -> None:
        with pytest.raises(ValueError, match="account_url is required"):
            AzureBlobConfig(account_key="my-key")

    def test_sas_token_requires_account_url(self) -> None:
        with pytest.raises(ValueError, match="account_url is required"):
            AzureBlobConfig(sas_token="my-sas")

    def test_connection_string_with_account_url_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            AzureBlobConfig(
                account_url="https://x.blob.core.windows.net",
                container_name="test",
                connection_string=(
                    "DefaultEndpointsProtocol=https;AccountName=fake;"
                    "AccountKey=ZmFrZQ==;"
                ),
            )

    def test_empty_string_account_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            AzureBlobConfig(
                account_url="https://x.blob.core.windows.net",
                container_name="test",
                account_key="",
            )

    def test_empty_string_sas_token_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            AzureBlobConfig(
                account_url="https://x.blob.core.windows.net",
                container_name="test",
                sas_token="",
            )

    def test_sas_token_only_question_marks_raises(self) -> None:
        with pytest.raises(ValueError, match="only '\\?' characters"):
            AzureBlobConfig(
                account_url="https://x.blob.core.windows.net",
                container_name="test",
                sas_token="???",
            )

    def test_sas_token_whitespace_stripped(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
            sas_token=" ?sv=2021-06-08&ss=b ",
        )
        assert config.sas_token == "sv=2021-06-08&ss=b"

    def test_empty_string_connection_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            AzureBlobConfig(
                container_name="test",
                connection_string="",
            )


class TestAuthClientCreation:
    """Test that _get_container creates the right client per auth method."""

    async def test_account_key_passed_as_credential(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
            account_key="my-account-key",
        )
        backend = AzureBlobBackend(config)

        with patch(f"{_BACKEND}.BlobServiceClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_container_client.return_value = AsyncMock()
            mock_cls.return_value = mock_client

            await backend._get_container()

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs["credential"] == "my-account-key"

        await backend.close()

    async def test_partner_user_agent_passed_to_client(self) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        config = AzureBlobConfig(
            container_name="test",
            connection_string=(
                "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"
            ),
        )
        backend = AzureBlobBackend(config)

        with patch(f"{_BACKEND}.BlobServiceClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_container_client.return_value = AsyncMock()
            mock_cls.from_connection_string.return_value = mock_client

            await backend._get_container()

            call_kwargs = mock_cls.from_connection_string.call_args
            assert call_kwargs.kwargs["user_agent"] == USER_AGENT

        await backend.close()

    async def test_sas_token_wrapped_in_azure_sas_credential(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
            sas_token="sv=2021-06-08&ss=b",
        )
        backend = AzureBlobBackend(config)

        with (
            patch(f"{_BACKEND}.BlobServiceClient") as mock_cls,
            patch(f"{_BACKEND}.AzureSasCredential") as mock_sas_cls,
        ):
            mock_sas_instance = MagicMock()
            mock_sas_cls.return_value = mock_sas_instance
            mock_client = AsyncMock()
            mock_client.get_container_client.return_value = AsyncMock()
            mock_cls.return_value = mock_client

            await backend._get_container()

            mock_sas_cls.assert_called_once_with("sv=2021-06-08&ss=b")
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs["credential"] is mock_sas_instance

        await backend.close()

    async def test_sas_token_leading_question_mark_stripped(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
            sas_token="?sv=2021-06-08&ss=b",
        )
        backend = AzureBlobBackend(config)

        with (
            patch(f"{_BACKEND}.BlobServiceClient") as mock_cls,
            patch(f"{_BACKEND}.AzureSasCredential") as mock_sas_cls,
        ):
            mock_sas_instance = MagicMock()
            mock_sas_cls.return_value = mock_sas_instance
            mock_client = AsyncMock()
            mock_client.get_container_client.return_value = AsyncMock()
            mock_cls.return_value = mock_client

            await backend._get_container()

            # Leading '?' should be stripped before AzureSasCredential.
            mock_sas_cls.assert_called_once_with("sv=2021-06-08&ss=b")

        await backend.close()

    async def test_connection_string_uses_from_connection_string(self) -> None:
        conn_str = (
            "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"
        )
        config = AzureBlobConfig(
            container_name="test",
            connection_string=conn_str,
        )
        backend = AzureBlobBackend(config)

        with patch(f"{_BACKEND}.BlobServiceClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_container_client.return_value = AsyncMock()
            mock_cls.from_connection_string.return_value = mock_client

            await backend._get_container()

            mock_cls.from_connection_string.assert_called_once()
            # Regular constructor should NOT be called.
            mock_cls.assert_not_called()

        await backend.close()

    async def test_default_credential_used_when_no_auth(self) -> None:
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)

        with (
            patch(f"{_BACKEND}.BlobServiceClient") as mock_cls,
            patch("azure.identity.aio.DefaultAzureCredential") as mock_cred,
        ):
            mock_cred_instance = AsyncMock()
            mock_cred.return_value = mock_cred_instance
            mock_client = AsyncMock()
            mock_client.get_container_client.return_value = AsyncMock()
            mock_cls.return_value = mock_client

            await backend._get_container()

            mock_cred.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs["credential"] is mock_cred_instance

        await backend.close()


# ------------------------------------------------------------------
# Backend class tests (no I/O)
# ------------------------------------------------------------------


class TestAzureBlobBackendInit:
    def test_is_backend_protocol(self) -> None:
        from deepagents.backends.protocol import BackendProtocol

        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)
        assert isinstance(backend, BackendProtocol)

    def test_credential_is_none_on_init(self) -> None:
        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)
        assert backend._credential is None


class TestAzureBlobBackendClose:
    """Tests for the close() lifecycle of AzureBlobBackend."""

    async def test_close_calls_credential_close(self) -> None:
        """close() must await the stored credential's close() coroutine."""
        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)

        mock_credential = AsyncMock()
        mock_credential.close = AsyncMock()
        backend._credential = mock_credential

        await backend.close()

        mock_credential.close.assert_awaited_once()
        assert backend._credential is None

    async def test_close_is_safe_without_credential(self) -> None:
        """close() must not raise when _credential is None."""
        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)
        # Should not raise even with no client or credential initialised.
        await backend.close()

    async def test_get_container_stores_default_credential(self) -> None:
        """_get_container() stores the auto-created DefaultAzureCredential."""
        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)

        mock_credential = AsyncMock()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.get_container_client.return_value = mock_container

        with (
            patch(
                f"{_BACKEND}.BlobServiceClient",
                return_value=mock_client,
            ),
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=mock_credential,
            ),
        ):
            await backend._get_container()

        assert backend._credential is mock_credential

    async def test_get_container_does_not_store_user_credential(self) -> None:
        """A caller-supplied credential is caller-owned and never stored/closed."""
        mock_credential = AsyncMock()

        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
            credential=mock_credential,
        )
        backend = AzureBlobBackend(config)

        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client.get_container_client.return_value = mock_container

        with patch(
            f"{_BACKEND}.BlobServiceClient",
            return_value=mock_client,
        ):
            await backend._get_container()

        # close() must not tear down a caller-owned credential.
        assert backend._credential is None
        await backend.close()
        mock_credential.close.assert_not_awaited()

    async def test_get_container_does_not_store_sync_credential(self) -> None:
        """Sync (non-async) credentials are not stored on self._credential."""
        sync_credential = MagicMock()
        # Ensure close is a plain (non-coroutine) callable.
        sync_credential.close = MagicMock()

        config = AzureBlobConfig(
            account_url="https://test.blob.core.windows.net",
            container_name="test",
            credential=sync_credential,
        )
        backend = AzureBlobBackend(config)

        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.get_container_client.return_value = mock_container

        with patch(
            f"{_BACKEND}.BlobServiceClient",
            return_value=mock_client,
        ):
            await backend._get_container()

        assert backend._credential is None


# ------------------------------------------------------------------
# build_file_info tests
# ------------------------------------------------------------------


class TestBuildFileInfo:
    def test_defaults(self) -> None:
        info = build_file_info("/src/main.py")
        assert info == {
            "path": "/src/main.py",
            "is_dir": False,
            "size": 0,
            "modified_at": "",
        }

    def test_directory(self) -> None:
        info = build_file_info("/src/", is_dir=True, size=0)
        assert info["is_dir"] is True

    def test_custom_values(self) -> None:
        info = build_file_info("/f.txt", size=100, modified_at="2026-01-01T00:00:00Z")
        assert info["size"] == 100
        assert info["modified_at"] == "2026-01-01T00:00:00Z"


# ------------------------------------------------------------------
# Helpers for mocked backend tests
# ------------------------------------------------------------------


def _make_backend(prefix: str = "pfx/") -> AzureBlobBackend:
    config = AzureBlobConfig(
        container_name="test",
        prefix=prefix,
        connection_string=(
            "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"
        ),
    )
    return AzureBlobBackend(config)


def _make_blob(
    name: str,
    content: str = "",
    size: int = 0,
    metadata: dict | None = None,
) -> MagicMock:
    """Create a mock blob object."""
    blob = MagicMock()
    blob.name = name
    blob.size = size
    blob.metadata = metadata
    blob._content = content
    return blob


async def _setup_backend_with_container(
    prefix: str = "pfx/",
) -> tuple[AzureBlobBackend, MagicMock]:
    """Create a backend with a pre-injected mock container.

    Uses MagicMock for the container because get_blob_client() is synchronous
    in the real Azure SDK. Async methods (exists, download_blob, etc.) are
    configured on individual blob client mocks in each test.
    """
    backend = _make_backend(prefix)
    container = MagicMock()
    default_blob_client = AsyncMock()
    default_blob_client.get_blob_properties.side_effect = ResourceNotFoundError(
        "not found"
    )
    container.get_blob_client.return_value = default_blob_client
    backend._container = container
    backend._client = AsyncMock()
    return backend, container


# ------------------------------------------------------------------
# _get_container tests
# ------------------------------------------------------------------


class TestGetContainer:
    async def test_returns_cached_container(self) -> None:
        backend, container = await _setup_backend_with_container()
        result = await backend._get_container()
        assert result is container

    async def test_connection_string_path(self) -> None:
        backend = _make_backend()
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.get_container_client.return_value = mock_container

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = mock_client
            result = await backend._get_container()

        assert result is mock_container
        MockBSC.from_connection_string.assert_called_once()

    async def test_double_checked_locking(self) -> None:
        """Second call inside the lock returns cached container."""
        backend = _make_backend()
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.get_container_client.return_value = mock_container

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = mock_client
            c1 = await backend._get_container()
            c2 = await backend._get_container()

        assert c1 is c2
        MockBSC.from_connection_string.assert_called_once()

    async def test_concurrent_init_double_check(self) -> None:
        """Second task finds container set inside the lock."""
        import asyncio

        backend = _make_backend()
        mock_container = MagicMock()

        # Hold the lock, let a task queue behind us, then set container.
        async with backend._init_lock:
            # Task starts: passes outer check (None), blocks on lock.
            task = asyncio.create_task(backend._get_container())
            await asyncio.sleep(0)  # Let task run until it blocks on the lock.
            # Set container while we hold the lock.
            backend._container = mock_container
        # Lock released: task acquires it, inner check finds container set.
        result = await task
        assert result is mock_container


# ------------------------------------------------------------------
# close tests (with client)
# ------------------------------------------------------------------


class TestCloseWithClient:
    async def test_close_clears_client_and_container(self) -> None:
        backend, _ = await _setup_backend_with_container()
        await backend.close()
        assert backend._client is None
        assert backend._container is None


# ------------------------------------------------------------------
# Helper method tests
# ------------------------------------------------------------------


class TestHelperMethods:
    def test_blob_key(self) -> None:
        backend = _make_backend("pfx/")
        assert backend._blob_key("/src/main.py") == "pfx/src/main.py"

    def test_virtual_path(self) -> None:
        backend = _make_backend("pfx/")
        assert backend._virtual_path("pfx/src/main.py") == "/src/main.py"

    async def test_blob_exists_true(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.exists.return_value = True
        container.get_blob_client.return_value = mock_blob
        assert await backend._blob_exists(container, "pfx/file.txt") is True

    async def test_blob_exists_false(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.exists.return_value = False
        container.get_blob_client.return_value = mock_blob
        assert await backend._blob_exists(container, "pfx/file.txt") is False

    async def test_read_blob(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "hello world"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {"created_at": "t1", "modified_at": "t2"}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        content, metadata = await backend._read_blob(container, "pfx/file.txt")
        assert content == "hello world"
        assert metadata == {"created_at": "t1", "modified_at": "t2"}

    async def test_read_blob_no_metadata(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "data"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = None
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        content, metadata = await backend._read_blob(container, "pfx/file.txt")
        assert content == "data"
        assert metadata == {}

    async def test_write_blob(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        await backend._write_blob(container, "pfx/file.txt", "content")
        mock_blob.upload_blob.assert_awaited_once()
        call_kwargs = mock_blob.upload_blob.call_args
        assert call_kwargs.kwargs["overwrite"] is True
        assert "created_at" in call_kwargs.kwargs["metadata"]

    async def test_write_blob_preserves_created_at(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        await backend._write_blob(
            container, "pfx/file.txt", "content", created_at="original"
        )
        call_kwargs = mock_blob.upload_blob.call_args
        assert call_kwargs.kwargs["metadata"]["created_at"] == "original"

    async def test_list_blobs(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob1 = _make_blob("pfx/a.txt")
        blob2 = _make_blob("pfx/b.txt")

        async def fake_list_blobs(**kwargs: Any) -> AsyncIterator[Any]:
            for b in [blob1, blob2]:
                yield b

        container.list_blobs = fake_list_blobs
        result = await backend._list_blobs(container, "pfx/")
        assert len(result) == 2

    async def test_list_blobs_empty_prefix(self) -> None:
        backend, container = await _setup_backend_with_container()

        async def fake_list_blobs(**kwargs: Any) -> AsyncIterator[Any]:
            assert kwargs.get("name_starts_with") is None
            return
            yield  # make it an async generator

        container.list_blobs = fake_list_blobs
        result = await backend._list_blobs(container, "")
        assert result == []


# ------------------------------------------------------------------
# _run_async tests
# ------------------------------------------------------------------


class TestRunAsync:
    def test_run_async_no_loop(self) -> None:
        backend = _make_backend()

        async def coro() -> Any:
            return 42

        assert backend._run_async(coro) == 42

    def test_run_async_nested_loop(self) -> None:
        """_run_async uses a thread when already inside an event loop."""
        import asyncio

        backend = _make_backend()

        async def inner() -> Any:
            return 99

        async def outer() -> Any:
            return backend._run_async(inner)

        result = asyncio.run(outer())
        assert result == 99

    def test_run_async_closes_temporary_client(self) -> None:
        """A client created inside the temporary loop is closed on cleanup."""
        backend = _make_backend()
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_client.get_container_client.return_value = MagicMock()

        async def coro() -> Any:
            await backend._get_container()
            return "ok"

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = fake_client
            assert backend._run_async(coro) == "ok"

        fake_client.close.assert_awaited_once()
        assert backend._client is None
        assert backend._container is None

    def test_run_async_swallows_client_close_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An error closing the temporary client is logged and swallowed."""
        backend = _make_backend()
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_client.close.side_effect = RuntimeError("boom")
        fake_client.get_container_client.return_value = MagicMock()

        async def coro() -> Any:
            await backend._get_container()
            return "ok"

        # Must not raise even though close() blew up.
        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = fake_client
            with caplog.at_level(logging.WARNING, logger=_BACKEND):
                assert backend._run_async(coro) == "ok"

        fake_client.close.assert_awaited_once()
        assert "Error closing temporary client" in caplog.text

    def test_run_async_closes_temporary_credential(self) -> None:
        """A credential created inside the temporary loop is closed."""
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)
        fake_credential = AsyncMock()
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_client.get_container_client.return_value = MagicMock()

        async def coro() -> Any:
            await backend._get_container()
            return "ok"

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=fake_credential,
            ),
            patch(f"{_BACKEND}.BlobServiceClient", return_value=fake_client),
        ):
            assert backend._run_async(coro) == "ok"

        fake_credential.close.assert_awaited_once()
        fake_client.close.assert_awaited_once()

    def test_run_async_swallows_credential_close_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An error closing the temporary credential is logged and swallowed."""
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
        )
        backend = AzureBlobBackend(config)
        fake_credential = AsyncMock()
        fake_credential.close.side_effect = RuntimeError("boom")
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_client.get_container_client.return_value = MagicMock()

        async def coro() -> Any:
            await backend._get_container()
            return "ok"

        with (
            patch(
                "azure.identity.aio.DefaultAzureCredential",
                return_value=fake_credential,
            ),
            patch(f"{_BACKEND}.BlobServiceClient", return_value=fake_client),
        ):
            with caplog.at_level(logging.WARNING, logger=_BACKEND):
                assert backend._run_async(coro) == "ok"

        fake_credential.close.assert_awaited_once()
        assert "Error closing temporary credential" in caplog.text

    def test_run_async_skips_user_supplied_credential(self) -> None:
        """User-supplied credentials are caller-owned and never closed."""
        user_cred = AsyncMock()
        config = AzureBlobConfig(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
            credential=user_cred,
        )
        backend = AzureBlobBackend(config)
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_client.get_container_client.return_value = MagicMock()

        async def coro() -> Any:
            await backend._get_container()
            return "ok"

        with patch(f"{_BACKEND}.BlobServiceClient", return_value=fake_client):
            assert backend._run_async(coro) == "ok"

        user_cred.close.assert_not_awaited()
        fake_client.close.assert_awaited_once()

    def test_run_async_leaves_instance_cache_untouched(self) -> None:
        """Temporary sync-wrapper clients do not replace the async cache."""
        backend = _make_backend()
        original_client = MagicMock()
        original_container = MagicMock()
        original_credential = MagicMock()
        backend._client = original_client
        backend._container = original_container
        backend._credential = original_credential
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_client.get_container_client.return_value = MagicMock()

        async def coro() -> Any:
            container = await backend._get_container()
            assert container is not original_container
            return "ok"

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = fake_client
            assert backend._run_async(coro) == "ok"

        assert backend._client is original_client
        assert backend._container is original_container
        assert backend._credential is original_credential
        fake_client.close.assert_awaited_once()

    def test_run_async_reuses_temporary_container(self) -> None:
        """Repeated _get_container calls in one wrapper reuse temporary state."""
        backend = _make_backend()
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_container = MagicMock()
        fake_client.get_container_client.return_value = fake_container

        async def coro() -> Any:
            first = await backend._get_container()
            second = await backend._get_container()
            assert first is fake_container
            assert second is fake_container
            return "ok"

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = fake_client
            assert backend._run_async(coro) == "ok"

        MockBSC.from_connection_string.assert_called_once()
        fake_client.close.assert_awaited_once()

    def test_run_async_temporary_container_double_checked_lock(self) -> None:
        """A queued temporary _get_container sees state set inside the lock."""
        import asyncio

        from langchain_azure_storage.deepagents import backend as backend_module

        backend = _make_backend()
        fake_container = MagicMock()

        async def coro() -> Any:
            states = backend_module._temporary_client_states.get()
            assert states is not None
            state = states.setdefault(id(backend), backend_module._ClientState())
            async with state.init_lock:
                task = asyncio.create_task(backend._get_container())
                await asyncio.sleep(0)
                state.container = fake_container
            return await task

        assert backend._run_async(coro) is fake_container

    def test_run_async_uses_separate_temporary_state_per_backend(self) -> None:
        """Backends in the same wrapper do not share temporary clients."""
        backend_one = _make_backend("one/")
        backend_two = _make_backend("two/")
        client_one = MagicMock()
        client_one.close = AsyncMock()
        container_one = MagicMock()
        client_one.get_container_client.return_value = container_one
        client_two = MagicMock()
        client_two.close = AsyncMock()
        container_two = MagicMock()
        client_two.get_container_client.return_value = container_two

        async def coro() -> Any:
            assert await backend_one._get_container() is container_one
            assert await backend_two._get_container() is container_two
            return "ok"

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.side_effect = [client_one, client_two]
            assert backend_one._run_async(coro) == "ok"

        assert MockBSC.from_connection_string.call_count == 2
        client_one.close.assert_awaited_once()
        client_two.close.assert_awaited_once()

    def test_run_async_temporary_state_does_not_hide_instance_cache(self) -> None:
        """Overlapping async work sees the instance cache, not temporary state."""
        import asyncio
        import threading

        backend = _make_backend()
        original_container = MagicMock()
        backend._client = MagicMock()
        backend._container = original_container

        started = threading.Event()
        release = threading.Event()
        result = []
        errors = []
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        fake_container = MagicMock()
        fake_client.get_container_client.return_value = fake_container

        async def coro() -> Any:
            container = await backend._get_container()
            assert container is fake_container
            started.set()
            for _ in range(200):
                if release.is_set():
                    break
                await asyncio.sleep(0.01)
            assert release.is_set()
            return "ok"

        def run_sync_wrapper() -> None:
            try:
                result.append(backend._run_async(coro))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with patch(f"{_BACKEND}.BlobServiceClient") as MockBSC:
            MockBSC.from_connection_string.return_value = fake_client
            thread = threading.Thread(target=run_sync_wrapper)
            thread.start()
            assert started.wait(timeout=2)
            assert asyncio.run(backend._get_container()) is original_container
            release.set()
            thread.join(timeout=2)

        assert not thread.is_alive()
        if errors:
            raise errors[0]
        assert result == ["ok"]

        fake_client.close.assert_awaited_once()


# ------------------------------------------------------------------
# aread tests
# ------------------------------------------------------------------


class TestARead:
    async def test_read_success(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "line1\nline2\nline3\n"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert "line1" in result.file_data["content"]
        assert "line2" in result.file_data["content"]
        assert "line3" in result.file_data["content"]

    async def test_read_not_found(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.download_blob.side_effect = ResourceNotFoundError("not found")
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/missing.txt")
        assert result.error is not None
        assert "Error" in result.error
        assert "not found" in result.error.lower()

    async def test_read_empty_file(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = ""
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/empty.txt")
        # Raw empty content is returned; the empty-file reminder is added by
        # the Deep Agents middleware, not the backend.
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""
        assert result.file_data["encoding"] == "utf-8"

    async def test_read_with_offset_and_limit(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "line1\nline2\nline3\nline4\nline5\n"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/file.txt", offset=1, limit=2)
        # Raw (unformatted) content for the requested window; no line numbers.
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "line2\nline3\n"

    async def test_read_offset_out_of_range(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "line1\n"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/file.txt", offset=100)
        assert result.error is not None
        assert "offset" in result.error.lower()

    async def test_read_binary_returns_base64(self) -> None:
        backend, container = await _setup_backend_with_container()
        raw = b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x01"  # not valid UTF-8
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = raw
        mock_blob.download_blob.return_value = mock_stream
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/image.png")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        import base64

        assert base64.b64decode(result.file_data["content"]) == raw

    async def test_read_success_returns_raw_unformatted(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "alpha\nbeta\ngamma"
        mock_blob.download_blob.return_value = mock_stream
        container.get_blob_client.return_value = mock_blob

        result = await backend.aread("/file.txt")
        # No line-number prefixes (those are added by the middleware).
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "alpha\nbeta\ngamma"
        assert "1\t" not in result.file_data["content"]

    async def test_read_invalid_path(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.aread("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


# ------------------------------------------------------------------
# awrite tests
# ------------------------------------------------------------------


class TestAWrite:
    async def test_write_new_file(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.upload_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        result = await backend.awrite("/new.txt", "hello")
        assert result.path == "/new.txt"
        assert result.error is None

    async def test_write_existing_file_fails(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = mock_blob

        result = await backend.awrite("/existing.txt", "hello")
        assert result.error is not None
        assert "already exists" in result.error

    async def test_write_invalid_path_fails(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.awrite("/src/../bad.txt", "hello")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


# ------------------------------------------------------------------
# aedit tests
# ------------------------------------------------------------------


class TestAEdit:
    async def test_edit_success(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "hello world"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {"created_at": "t1", "modified_at": "t2"}
        mock_blob.get_blob_properties.return_value = mock_props
        mock_blob.upload_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        result = await backend.aedit("/file.txt", "hello", "goodbye")
        assert result.path == "/file.txt"
        assert result.error is None

    async def test_edit_not_found(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.download_blob.side_effect = ResourceNotFoundError("not found")
        container.get_blob_client.return_value = mock_blob

        result = await backend.aedit("/missing.txt", "a", "b")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_string_not_found(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "hello world"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        result = await backend.aedit("/file.txt", "nonexistent", "replacement")
        assert result.error is not None

    async def test_edit_multiple_occurrences_fails(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "aaa"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {}
        mock_blob.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob

        result = await backend.aedit("/file.txt", "a", "b", replace_all=False)
        assert result.error is not None

    async def test_edit_replace_all(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "aaa"
        mock_blob.download_blob.return_value = mock_stream
        mock_props = MagicMock()
        mock_props.metadata = {"created_at": "t1"}
        mock_blob.get_blob_properties.return_value = mock_props
        mock_blob.upload_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        result = await backend.aedit("/file.txt", "a", "b", replace_all=True)
        assert result.path == "/file.txt"
        assert result.occurrences == 3

    async def test_edit_invalid_path(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.aedit("/src/../bad.txt", "a", "b")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


# ------------------------------------------------------------------
# als tests
# ------------------------------------------------------------------


class TestALs:
    async def test_ls_files(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob1 = _make_blob("pfx/src/a.py", size=100, metadata={"modified_at": "t1"})
        blob2 = _make_blob("pfx/src/b.py", size=200, metadata={"modified_at": "t2"})

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            for b in [blob1, blob2]:
                yield b

        container.list_blobs = fake_list
        result = await backend.als("/src")
        assert result.entries is not None
        assert len(result.entries) == 2
        paths = [r["path"] for r in result.entries]
        assert "/src/a.py" in paths
        assert "/src/b.py" in paths

    async def test_ls_synthesizes_directories(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob1 = _make_blob("pfx/src/sub/a.py", size=50)
        blob2 = _make_blob("pfx/src/b.py", size=100, metadata={"modified_at": "t1"})

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            for b in [blob1, blob2]:
                yield b

        container.list_blobs = fake_list
        result = await backend.als("/src")
        assert result.entries is not None
        dirs = [r for r in result.entries if r["is_dir"]]
        files = [r for r in result.entries if not r["is_dir"]]
        assert len(dirs) == 1
        assert dirs[0]["path"] == "/src/sub/"
        assert len(files) == 1

    async def test_ls_empty(self) -> None:
        backend, container = await _setup_backend_with_container()

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            return
            yield

        container.list_blobs = fake_list
        result = await backend.als("/empty")
        assert result.entries is not None
        assert result.entries == []

    async def test_ls_path_without_leading_slash(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/src/a.py", size=100, metadata={"modified_at": "t1"})

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.als("src")
        assert result.entries is not None
        assert len(result.entries) == 1
        assert result.entries[0]["path"] == "/src/a.py"

    async def test_ls_skips_non_matching_blobs(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/other/a.py", size=100)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.als("/src")
        assert result.entries is not None
        assert result.entries == []

    async def test_ls_skips_empty_relative(self) -> None:
        backend, container = await _setup_backend_with_container()
        # Blob key "pfx/src/" -> virtual "/src/" which equals normalized_path
        # "/src/" so relative becomes "" and is skipped.
        blob = _make_blob("pfx/src/", size=0)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.als("/src")
        assert result.entries is not None
        assert result.entries == []

    async def test_ls_no_metadata(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/src/a.py", size=50, metadata=None)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.als("/src")
        assert result.entries is not None
        assert len(result.entries) == 1
        assert result.entries[0]["modified_at"] == ""

    async def test_ls_invalid_path_returns_empty(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.als("/src/../bad")
        assert result.entries is not None
        assert result.entries == []


# ------------------------------------------------------------------
# aglob tests
# ------------------------------------------------------------------


class TestAGlob:
    async def test_glob_pattern(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob1 = _make_blob("pfx/src/a.py", size=100, metadata={"modified_at": "t1"})
        blob2 = _make_blob("pfx/src/b.txt", size=50)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            for b in [blob1, blob2]:
                yield b

        container.list_blobs = fake_list
        result = await backend.aglob("*.py", path="/src")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/src/a.py"

    async def test_glob_recursive(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob1 = _make_blob("pfx/a.py", size=100, metadata={"modified_at": "t1"})
        blob2 = _make_blob("pfx/sub/b.py", size=50, metadata={"modified_at": "t2"})
        blob3 = _make_blob("pfx/sub/c.txt", size=25)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            for b in [blob1, blob2, blob3]:
                yield b

        container.list_blobs = fake_list
        result = await backend.aglob("**/*.py", path="/")
        assert result.matches is not None
        paths = [r["path"] for r in result.matches]
        assert "/a.py" in paths
        assert "/sub/b.py" in paths
        assert len(result.matches) == 2

    async def test_glob_empty(self) -> None:
        backend, container = await _setup_backend_with_container()

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            return
            yield

        container.list_blobs = fake_list
        result = await backend.aglob("*.py")
        assert result.matches is not None
        assert result.matches == []

    async def test_glob_no_metadata(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/a.py", size=100, metadata=None)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.aglob("*.py", path="/")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["modified_at"] == ""

    async def test_glob_exact_path_match(self) -> None:
        """Exact file searches should return only the matching blob."""
        backend, container = await _setup_backend_with_container()
        mock_blob_client = AsyncMock()
        mock_props = MagicMock()
        mock_props.size = 100
        mock_props.metadata = {"modified_at": "t1"}
        mock_blob_client.get_blob_properties.return_value = mock_props
        container.get_blob_client.return_value = mock_blob_client
        container.list_blobs = AsyncMock()

        result = await backend.aglob("main.py", path="/src/main.py")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/src/main.py"
        container.list_blobs.assert_not_called()

    async def test_glob_skips_non_matching_prefix(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/other/a.py", size=100)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.aglob("*.py", path="/src")
        assert result.matches is not None
        assert result.matches == []

    async def test_glob_directory_listing_uses_trailing_slash_prefix(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob_client = AsyncMock()
        mock_blob_client.get_blob_properties.side_effect = ResourceNotFoundError(
            "missing"
        )
        container.get_blob_client.return_value = mock_blob_client

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            assert kwargs["name_starts_with"] == "pfx/src/"
            return
            yield

        container.list_blobs = fake_list
        result = await backend.aglob("*.py", path="/src")
        assert result.matches is not None
        assert result.matches == []

    async def test_glob_no_match(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/a.txt", size=100)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.aglob("*.py", path="/")
        assert result.matches is not None
        assert result.matches == []

    async def test_glob_invalid_path_returns_empty(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.aglob("*.py", path="/src/../bad")
        assert result.matches is not None
        assert result.matches == []


# ------------------------------------------------------------------
# agrep tests
# ------------------------------------------------------------------


class TestAGrep:
    async def test_grep_finds_matches(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/file.py", size=50)
        mock_blob_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "hello world\ngoodbye world\nhello again\n"
        mock_blob_client.download_blob.return_value = mock_stream

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        container.get_blob_client.return_value = mock_blob_client

        result = await backend.agrep("hello")
        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.matches[0]["line"] == 1
        assert result.matches[1]["line"] == 3

    async def test_grep_with_glob_filter(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob_py = _make_blob("pfx/file.py", size=50)
        blob_txt = _make_blob("pfx/file.txt", size=50)
        mock_blob_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "match here\n"
        mock_blob_client.download_blob.return_value = mock_stream

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            for b in [blob_py, blob_txt]:
                yield b

        container.list_blobs = fake_list
        container.get_blob_client.return_value = mock_blob_client

        result = await backend.agrep("match", glob="*.py")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/file.py"

    async def test_grep_with_path_aware_glob_filter(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob_nested = _make_blob("pfx/src/lib/file.py", size=50)
        blob_top = _make_blob("pfx/src/file.py", size=50)
        mock_blob_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "match here\n"
        mock_blob_client.download_blob.return_value = mock_stream

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            for blob in [blob_nested, blob_top]:
                yield blob

        container.list_blobs = fake_list
        container.get_blob_client.return_value = mock_blob_client

        result = await backend.agrep("match", path="/", glob="src/*/*.py")
        assert result.matches is not None
        assert [match["path"] for match in result.matches] == ["/src/lib/file.py"]

    async def test_grep_no_matches(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/file.py", size=50)
        mock_blob_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "nothing here\n"
        mock_blob_client.download_blob.return_value = mock_stream

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        container.get_blob_client.return_value = mock_blob_client

        result = await backend.agrep("missing")
        assert result.matches == []

    async def test_grep_empty_listing(self) -> None:
        backend, container = await _setup_backend_with_container()

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            return
            yield

        container.list_blobs = fake_list
        result = await backend.agrep("pattern")
        assert result.matches == []

    async def test_grep_blob_read_failure(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/file.py", size=50)
        mock_blob_client = AsyncMock()
        mock_blob_client.download_blob.side_effect = ResourceNotFoundError("read error")

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        container.get_blob_client.return_value = mock_blob_client

        result = await backend.agrep("pattern")
        assert result.error is not None
        assert "could not read 1 file" in result.error.lower()
        assert "/file.py" in result.error

    async def test_grep_with_path(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/src/file.py", size=50)
        mock_blob_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "match\n"
        mock_blob_client.get_blob_properties.side_effect = ResourceNotFoundError(
            "not found"
        )
        mock_blob_client.download_blob.return_value = mock_stream

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        container.get_blob_client.return_value = mock_blob_client

        result = await backend.agrep("match", path="/src")
        assert result.matches is not None
        assert len(result.matches) == 1

    async def test_grep_exact_path_uses_exact_blob_lookup(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob_client = AsyncMock()
        mock_props = MagicMock()
        mock_props.size = 10
        mock_props.metadata = {}
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "match\n"
        mock_blob_client.get_blob_properties.return_value = mock_props
        mock_blob_client.download_blob.return_value = mock_stream
        container.get_blob_client.return_value = mock_blob_client
        container.list_blobs = AsyncMock()

        result = await backend.agrep("match", path="/src/file.py")
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "/src/file.py"
        container.list_blobs.assert_not_called()

    async def test_grep_invalid_path(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.agrep("match", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    async def test_grep_skips_blobs_outside_requested_path(self) -> None:
        backend, container = await _setup_backend_with_container()
        blob = _make_blob("pfx/other/file.py", size=50)

        async def fake_list(**kwargs: Any) -> AsyncIterator[Any]:
            yield blob

        container.list_blobs = fake_list
        result = await backend.agrep("match", path="/src")
        assert result.matches == []


# ------------------------------------------------------------------
# aupload_files tests
# ------------------------------------------------------------------


class TestAUploadFiles:
    async def test_upload_success(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        result = await backend.aupload_files([("/file.bin", b"binary data")])
        assert len(result) == 1
        assert result[0].path == "/file.bin"
        assert result[0].error is None

    async def test_upload_failure(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.upload_blob.side_effect = Exception("upload failed")
        container.get_blob_client.return_value = mock_blob

        result = await backend.aupload_files([("/file.bin", b"data")])
        assert len(result) == 1
        assert result[0].error == "permission_denied"

    async def test_upload_multiple_files(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        container.get_blob_client.return_value = mock_blob

        files = [("/a.bin", b"aaa"), ("/b.bin", b"bbb")]
        result = await backend.aupload_files(files)
        assert len(result) == 2
        assert all(r.error is None for r in result)

    async def test_upload_invalid_path(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.aupload_files([("/src/../bad.bin", b"data")])
        assert result[0].error == "invalid_path"


# ------------------------------------------------------------------
# adownload_files tests
# ------------------------------------------------------------------


class TestADownloadFiles:
    async def test_download_success(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = b"file content"
        mock_blob.download_blob.return_value = mock_stream
        container.get_blob_client.return_value = mock_blob

        result = await backend.adownload_files(["/file.txt"])
        assert len(result) == 1
        assert result[0].content == b"file content"
        assert result[0].error is None

    async def test_download_not_found(self) -> None:
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_blob.download_blob.side_effect = ResourceNotFoundError("not found")
        container.get_blob_client.return_value = mock_blob

        result = await backend.adownload_files(["/missing.txt"])
        assert len(result) == 1
        assert result[0].error == "file_not_found"
        assert result[0].content is None

    async def test_download_string_content_encoded(self) -> None:
        """When readall returns a string, it should be encoded to bytes."""
        backend, container = await _setup_backend_with_container()
        mock_blob = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.readall.return_value = "string content"
        mock_blob.download_blob.return_value = mock_stream
        container.get_blob_client.return_value = mock_blob

        result = await backend.adownload_files(["/file.txt"])
        assert result[0].content == b"string content"

    async def test_download_invalid_path(self) -> None:
        backend, _ = await _setup_backend_with_container()

        result = await backend.adownload_files(["/src/../bad.txt"])
        assert result[0].error == "invalid_path"


# ------------------------------------------------------------------
# Sync wrapper tests
# ------------------------------------------------------------------


class TestSyncWrappers:
    def test_read_sync(self) -> None:
        backend = _make_backend()
        expected = ReadResult(file_data={"content": "x", "encoding": "utf-8"})
        backend.aread = AsyncMock(return_value=expected)  # type: ignore[method-assign]
        result = backend.read("/file.txt")
        assert result == expected

    def test_write_sync(self) -> None:
        backend = _make_backend()
        backend.awrite = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteResult(path="/f.txt")
        )
        result = backend.write("/f.txt", "content")
        assert result.path == "/f.txt"

    def test_edit_sync(self) -> None:
        backend = _make_backend()
        backend.aedit = AsyncMock(  # type: ignore[method-assign]
            return_value=EditResult(path="/f.txt", occurrences=1)
        )
        result = backend.edit("/f.txt", "a", "b")
        assert result.path == "/f.txt"

    def test_ls_sync(self) -> None:
        backend = _make_backend()
        backend.als = AsyncMock(return_value=LsResult(entries=[]))  # type: ignore[method-assign]
        result = backend.ls("/")
        assert result == LsResult(entries=[])

    def test_glob_sync(self) -> None:
        backend = _make_backend()
        backend.aglob = AsyncMock(return_value=GlobResult(matches=[]))  # type: ignore[method-assign]
        result = backend.glob("*.py")
        assert result == GlobResult(matches=[])

    def test_grep_sync(self) -> None:
        backend = _make_backend()
        backend.agrep = AsyncMock(return_value=GrepResult(matches=[]))  # type: ignore[method-assign]
        result = backend.grep("pattern")
        assert result == GrepResult(matches=[])

    def test_upload_files_sync(self) -> None:
        backend = _make_backend()
        backend.aupload_files = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = backend.upload_files([])
        assert result == []

    def test_download_files_sync(self) -> None:
        backend = _make_backend()
        backend.adownload_files = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = backend.download_files([])
        assert result == []
