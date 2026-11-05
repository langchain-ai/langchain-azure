"""Unit tests for ``AzureBlobBackend`` construction and resource lifecycle.

Covers the constructor, ``from_connection_string``, user-agent wiring,
credential resolution, container-client caching, and ``close()`` /
``aclose()``. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"

# Nearly every test constructs a backend, so silence the beta warning at the
# module level; TestFromConnectionString::test_emits_beta_warning still
# asserts it (pytest.warns overrides this filter).
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestConstructor:
    def test_is_backend_protocol(self) -> None:
        from deepagents.backends.protocol import BackendProtocol

        backend = AzureBlobBackend(
            account_url="https://x.blob.core.windows.net", container_name="c"
        )
        assert isinstance(backend, BackendProtocol)

    def test_prefix_defaults_to_empty(self) -> None:
        backend = AzureBlobBackend(
            account_url="https://x.blob.core.windows.net", container_name="c"
        )
        assert backend._prefix == ""

    def test_prefix_none_normalized(self) -> None:
        backend = AzureBlobBackend(
            account_url="https://x.blob.core.windows.net",
            container_name="c",
            prefix=None,
        )
        assert backend._prefix == ""

    def test_missing_account_url_raises(self) -> None:
        with pytest.raises(TypeError):
            AzureBlobBackend(container_name="c")  # type: ignore[call-arg]

    def test_missing_container_name_raises(self) -> None:
        with pytest.raises(TypeError):
            AzureBlobBackend("https://x.blob.core.windows.net")  # type: ignore[call-arg]


class TestFromConnectionString:
    def test_sets_connection_string_container_and_prefix(self) -> None:
        backend = AzureBlobBackend.from_connection_string(
            _CONN_STR, "test", prefix="pfx/"
        )
        assert backend._connection_string == _CONN_STR
        assert backend._container_name == "test"
        assert backend._prefix == "pfx/"

    def test_is_backend_protocol(self) -> None:
        from deepagents.backends.protocol import BackendProtocol

        backend = AzureBlobBackend.from_connection_string(_CONN_STR, "test")
        assert isinstance(backend, BackendProtocol)

    def test_emits_beta_warning(self) -> None:
        # The @beta wrapper on __init__ suppresses its warning for callers
        # inside langchain* packages, which includes this classmethod, so the
        # classmethod emits it explicitly.
        from langchain_core._api import LangChainBetaWarning

        with pytest.warns(LangChainBetaWarning, match="public preview"):
            AzureBlobBackend.from_connection_string(_CONN_STR, "test")


class TestUserAgent:
    async def test_user_agent_passed_to_async_client(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        mock_cls, container = patched_async
        container.walk_blobs = async_list([])
        await backend.als("/")
        assert (
            mock_cls.from_connection_string.call_args.kwargs["user_agent"] == USER_AGENT
        )

    def test_user_agent_passed_to_sync_client(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        mock_cls, container = patched_sync
        container.walk_blobs.return_value = []
        backend.ls("/")
        assert (
            mock_cls.from_connection_string.call_args.kwargs["user_agent"] == USER_AGENT
        )


class TestCredentialHelpers:
    def test_sync_default_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity import DefaultAzureCredential

        cred = make_acct_backend()._resolve_sync_credential(None)
        assert isinstance(cred, DefaultAzureCredential)

    def test_sync_rejects_async_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity.aio import DefaultAzureCredential as AioCred

        with pytest.raises(ValueError, match="synchronous"):
            make_acct_backend()._resolve_sync_credential(AioCred())

    def test_sync_passthrough(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        assert make_acct_backend()._resolve_sync_credential(cred) is cred

    async def test_async_default_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity.aio import DefaultAzureCredential as AioCred

        cred = await make_acct_backend()._resolve_async_credential(None)
        assert isinstance(cred, AioCred)

    async def test_async_rejects_sync_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity import DefaultAzureCredential

        with pytest.raises(ValueError, match="asynchronous"):
            await make_acct_backend()._resolve_async_credential(
                DefaultAzureCredential()
            )

    async def test_async_passthrough_sas(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        got = await make_acct_backend()._resolve_async_credential(cred)
        assert got is cred


class TestContainerConstruction:
    """Container construction and caching (account_url path, not connection_string)."""

    def test_sync_creates_default_credential(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        mock_cc, container = patched_sync
        container.list_blobs.return_value = []
        fake_cred = MagicMock()
        with patch("azure.identity.DefaultAzureCredential", return_value=fake_cred):
            make_acct_backend().ls("/")
        assert mock_cc.call_args.kwargs["credential"] is fake_cred

    def test_sync_uses_provided_credential(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        mock_cc, container = patched_sync
        container.list_blobs.return_value = []
        make_acct_backend(cred).ls("/")
        assert mock_cc.call_args.kwargs["credential"] is cred

    def test_sync_container_reused_across_calls(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        mock_cc, container = patched_sync
        container.list_blobs.return_value = []
        backend = make_acct_backend()
        backend.ls("/")
        backend.ls("/")
        assert mock_cc.call_count == 1

    async def test_async_creates_default_credential(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        mock_cc, container = patched_async
        container.list_blobs = async_list([])
        fake_cred = MagicMock()
        fake_cred.close = AsyncMock()
        with patch("azure.identity.aio.DefaultAzureCredential", return_value=fake_cred):
            await make_acct_backend().als("/")
        assert mock_cc.call_args.kwargs["credential"] is fake_cred

    async def test_async_uses_provided_credential(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        mock_cc, container = patched_async
        container.list_blobs = async_list([])
        await make_acct_backend(cred).als("/")
        assert mock_cc.call_args.kwargs["credential"] is cred

    async def test_async_container_reused_across_calls(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        mock_cc, container = patched_async
        container.list_blobs = async_list([])
        backend = make_acct_backend()
        await backend.als("/")
        await backend.als("/")
        assert mock_cc.call_count == 1


class TestCloseAndAclose:
    def test_close_closes_owned_credential_and_container(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        fake_cred = MagicMock()
        with patch("azure.identity.DefaultAzureCredential", return_value=fake_cred):
            backend = make_acct_backend()
            backend.ls("/")
            fake_cred.close.assert_not_called()
            container.close.assert_not_called()
            backend.close()
        fake_cred.close.assert_called_once()
        container.close.assert_called_once()

    def test_close_does_not_close_caller_supplied_credential(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        _, container = patched_sync
        container.list_blobs.return_value = []
        backend = make_acct_backend(cred)
        backend.ls("/")
        backend.close()
        container.close.assert_called_once()

    def test_close_is_idempotent(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        backend = make_acct_backend()
        backend.ls("/")
        backend.close()
        backend.close()
        container.close.assert_called_once()

    def test_close_without_prior_use_is_a_noop(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        make_acct_backend().close()

    def test_context_manager_calls_close(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        with make_acct_backend() as backend:
            backend.ls("/")
        container.close.assert_called_once()

    async def test_aclose_closes_owned_credential_and_container(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([])
        fake_cred = MagicMock()
        fake_cred.close = AsyncMock()
        with patch("azure.identity.aio.DefaultAzureCredential", return_value=fake_cred):
            backend = make_acct_backend()
            await backend.als("/")
            fake_cred.close.assert_not_called()
            await backend.aclose()
        fake_cred.close.assert_awaited_once()
        container.close.assert_awaited_once()

    async def test_aclose_without_prior_use_is_a_noop(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        await make_acct_backend().aclose()

    async def test_async_context_manager_calls_aclose(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([])
        async with make_acct_backend() as backend:
            await backend.als("/")
        container.close.assert_awaited_once()
