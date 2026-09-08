"""Both Deep Agents backends must advertise that their interface is unstable.

deepagents is pre-1.0 and ACA sandboxes is an Early Access service, so the
`@beta` marker is load-bearing rather than decorative -- it matches the marker
on `langchain_azure_storage`'s backend and on the Azure document loaders.

`@beta` warns at most once per class per process and stays silent for callers
inside `langchain*` modules, so a `pytest.warns` assertion here would depend on
which test happened to construct the class first. The injected docstring
directive is asserted instead: it is what the marker guarantees regardless of
call order, and what users and the API reference actually see.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")


def _beta_notice(obj: type) -> str:
    doc = obj.__doc__
    assert doc is not None
    return doc


class TestSessionsBashBackend:
    def test_is_marked_beta(self) -> None:
        from langchain_azure_compute.dynamic_sessions.backends import (
            SessionsBashBackend,
        )

        doc = _beta_notice(SessionsBashBackend)
        assert doc.startswith(".. beta::")
        assert "`SessionsBashBackend` is in public preview" in doc

    def test_keeps_its_own_docstring(self) -> None:
        from langchain_azure_compute.dynamic_sessions.backends import (
            SessionsBashBackend,
        )

        assert "Azure Dynamic Sessions backend" in _beta_notice(SessionsBashBackend)


class TestACASandbox:
    def test_is_marked_beta(self) -> None:
        pytest.importorskip("azure.containerapps.sandbox")
        from langchain_azure_compute.sandboxes import ACASandbox

        doc = _beta_notice(ACASandbox)
        assert doc.startswith(".. beta::")
        assert "`ACASandbox` is in public preview" in doc

    def test_keeps_its_own_docstring(self) -> None:
        pytest.importorskip("azure.containerapps.sandbox")
        from langchain_azure_compute.sandboxes import ACASandbox

        assert "Azure Container Apps sandbox" in _beta_notice(ACASandbox)
