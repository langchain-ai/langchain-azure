"""Async variants must delegate to this backend's sync overrides.

`BaseSandbox`'s own `als`/`aread`/`awrite`/`aedit`/`aglob`/`agrep` rebuild its
`python3 -c` commands and send them through `aexecute` -- inheriting them
would silently bypass every shell-native override, the `/mnt/data` transfer
rules, and the output-loss compensation. These pin the delegation so a future
refactor cannot reintroduce that.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

pytest.importorskip("deepagents")

from langchain_azure_compute.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestAsyncDelegation:
    def test_als_delegates_to_ls(self, backend: SessionsBashBackend) -> None:
        sentinel = object()
        backend.ls = mock.Mock(return_value=sentinel)  # type: ignore[method-assign]
        assert asyncio.run(backend.als("/mnt/data")) is sentinel
        backend.ls.assert_called_once_with("/mnt/data")

    def test_aread_delegates_to_read(self, backend: SessionsBashBackend) -> None:
        sentinel = object()
        backend.read = mock.Mock(return_value=sentinel)  # type: ignore[method-assign]
        assert asyncio.run(backend.aread("/mnt/data/f.txt", 3, 7)) is sentinel
        backend.read.assert_called_once_with("/mnt/data/f.txt", 3, 7)

    def test_awrite_delegates_to_write(self, backend: SessionsBashBackend) -> None:
        sentinel = object()
        backend.write = mock.Mock(return_value=sentinel)  # type: ignore[method-assign]
        assert asyncio.run(backend.awrite("/mnt/data/f.txt", "x")) is sentinel
        backend.write.assert_called_once_with("/mnt/data/f.txt", "x")

    def test_aedit_delegates_to_edit(self, backend: SessionsBashBackend) -> None:
        sentinel = object()
        backend.edit = mock.Mock(return_value=sentinel)  # type: ignore[method-assign]
        assert (
            asyncio.run(backend.aedit("/mnt/data/f.txt", "a", "b", replace_all=True))
            is sentinel
        )
        backend.edit.assert_called_once_with("/mnt/data/f.txt", "a", "b", True)

    def test_aglob_delegates_to_glob(self, backend: SessionsBashBackend) -> None:
        sentinel = object()
        backend.glob = mock.Mock(return_value=sentinel)  # type: ignore[method-assign]
        assert asyncio.run(backend.aglob("*.py", "/mnt/data")) is sentinel
        backend.glob.assert_called_once_with("*.py", "/mnt/data")

    def test_agrep_delegates_to_grep(self, backend: SessionsBashBackend) -> None:
        sentinel = object()
        backend.grep = mock.Mock(return_value=sentinel)  # type: ignore[method-assign]
        assert (
            asyncio.run(backend.agrep("needle", "/mnt/data", "*.py", max_count=5))
            is sentinel
        )
        backend.grep.assert_called_once_with("needle", "/mnt/data", "*.py", max_count=5)

    def test_agrep_timeout_is_reported_not_raised(
        self, backend: SessionsBashBackend
    ) -> None:
        backend.grep = mock.Mock()  # type: ignore[method-assign]

        class _FakeAsyncio:
            """A stand-in for the module's `asyncio` reference.

            Patching `sessions.asyncio.wait_for` would mutate the *global*
            asyncio module for the test's duration -- `sessions.asyncio` is
            that module. Replacing the module attribute itself keeps the patch
            scoped to `sessions`; everything but `wait_for` delegates.
            """

            def __getattr__(self, name: str) -> object:
                return getattr(asyncio, name)

            @staticmethod
            async def wait_for(awaitable: object, timeout: float) -> None:
                # Close the `to_thread` coroutine before raising. A bare
                # `side_effect=TimeoutError` never consumes it, and the
                # abandoned coroutine surfaces later as `RuntimeWarning:
                # coroutine 'to_thread' was never awaited`.
                if hasattr(awaitable, "close"):
                    awaitable.close()  # type: ignore[attr-defined]
                raise TimeoutError

        with mock.patch(
            "langchain_azure_compute.dynamic_sessions.backends.sessions.asyncio",
            _FakeAsyncio(),
        ):
            result = asyncio.run(backend.agrep("needle"))
        assert result.matches is None
        assert result.error is not None
        assert "timed out" in result.error
