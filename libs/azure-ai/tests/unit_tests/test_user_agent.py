"""Unit tests for :mod:`langchain_azure_ai._user_agent`."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from langchain_azure_ai import _user_agent


@pytest.fixture(autouse=True)
def _isolate_ua_state() -> Iterator[None]:
    """Snapshot and restore the module-level prefix registry per test.

    The hosting subpackage registers its prefix at import time. Since
    ``sys.modules`` caches the import across tests, simply clearing the
    registry would orphan that registration. We snapshot first, then
    let each test mutate the registry, then fully restore.
    """
    saved_prefixes = dict(_user_agent._user_agent_prefixes)
    saved_detected = _user_agent._hosted_env_detected
    try:
        _user_agent._user_agent_prefixes.clear()
        _user_agent._hosted_env_detected = False
        yield
    finally:
        _user_agent._user_agent_prefixes.clear()
        _user_agent._user_agent_prefixes.update(saved_prefixes)
        _user_agent._hosted_env_detected = saved_detected


# ---------------------------------------------------------------------------
# Token shape
# ---------------------------------------------------------------------------


class TestGetUserAgent:
    def test_base_token_format(self) -> None:
        assert _user_agent.BASE_USER_AGENT.startswith("langchain-azure-ai/")

    def test_no_prefixes_returns_base(self) -> None:
        assert _user_agent.get_user_agent() == _user_agent.BASE_USER_AGENT

    def test_single_prefix(self) -> None:
        _user_agent.add_user_agent_prefix("foo/1.0")
        assert _user_agent.get_user_agent() == f"foo/1.0 {_user_agent.BASE_USER_AGENT}"

    def test_multiple_prefixes_in_insertion_order(self) -> None:
        _user_agent.add_user_agent_prefix("alpha/1.0")
        _user_agent.add_user_agent_prefix("beta/2.0")
        assert (
            _user_agent.get_user_agent()
            == f"alpha/1.0 beta/2.0 {_user_agent.BASE_USER_AGENT}"
        )

    def test_add_prefix_is_idempotent(self) -> None:
        _user_agent.add_user_agent_prefix("dup/1.0")
        _user_agent.add_user_agent_prefix("dup/1.0")
        assert _user_agent.get_user_agent().count("dup/1.0") == 1

    def test_empty_prefix_ignored(self) -> None:
        _user_agent.add_user_agent_prefix("")
        assert _user_agent.get_user_agent() == _user_agent.BASE_USER_AGENT


class TestVersionResolution:
    def test_unknown_package_falls_back_to_zero(self) -> None:
        """Reloading with a failing version lookup must not raise."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("langchain-azure-ai"),
        ):
            reloaded = importlib.reload(_user_agent)
            try:
                assert reloaded.BASE_USER_AGENT == "langchain-azure-ai/0.0.0"
            finally:
                # Restore the live module so subsequent tests see the real
                # version and the shared prefix registry.
                importlib.reload(_user_agent)


# ---------------------------------------------------------------------------
# Opt-out
# ---------------------------------------------------------------------------


class TestOptOut:
    def test_disabled_returns_empty_string(self) -> None:
        with patch.dict(
            os.environ,
            {_user_agent.USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: "1"},
        ):
            assert _user_agent.get_user_agent() == ""

    def test_disabled_accepts_true(self) -> None:
        with patch.dict(
            os.environ,
            {_user_agent.USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: "TRUE"},
        ):
            assert _user_agent.get_user_agent() == ""

    def test_false_value_keeps_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {_user_agent.USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: "false"},
        ):
            assert _user_agent.get_user_agent() == _user_agent.BASE_USER_AGENT


# ---------------------------------------------------------------------------
# with_user_agent
# ---------------------------------------------------------------------------


class TestWithUserAgent:
    def test_none_input_returns_ua_only(self) -> None:
        result = _user_agent.with_user_agent(None)
        assert result == {_user_agent.USER_AGENT_KEY: _user_agent.BASE_USER_AGENT}

    def test_empty_dict_input_returns_ua_only(self) -> None:
        result = _user_agent.with_user_agent({})
        assert result == {_user_agent.USER_AGENT_KEY: _user_agent.BASE_USER_AGENT}

    def test_preserves_other_headers(self) -> None:
        result = _user_agent.with_user_agent({"X-Trace-Id": "abc"})
        assert result["X-Trace-Id"] == "abc"
        assert result[_user_agent.USER_AGENT_KEY] == _user_agent.BASE_USER_AGENT

    def test_prepends_to_existing_user_agent(self) -> None:
        result = _user_agent.with_user_agent({_user_agent.USER_AGENT_KEY: "my-app/1.0"})
        assert (
            result[_user_agent.USER_AGENT_KEY]
            == f"{_user_agent.BASE_USER_AGENT} my-app/1.0"
        )

    def test_does_not_mutate_input(self) -> None:
        original = {_user_agent.USER_AGENT_KEY: "my-app/1.0"}
        _user_agent.with_user_agent(original)
        assert original == {_user_agent.USER_AGENT_KEY: "my-app/1.0"}

    def test_opt_out_returns_input_copy(self) -> None:
        original = {_user_agent.USER_AGENT_KEY: "my-app/1.0", "X-Trace-Id": "abc"}
        with patch.dict(
            os.environ,
            {_user_agent.USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: "1"},
        ):
            result = _user_agent.with_user_agent(original)
        assert result == original
        assert result is not original  # shallow copy

    def test_opt_out_with_none_returns_empty_dict(self) -> None:
        with patch.dict(
            os.environ,
            {_user_agent.USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: "1"},
        ):
            assert _user_agent.with_user_agent(None) == {}


# ---------------------------------------------------------------------------
# Hosting subpackage self-registration
# ---------------------------------------------------------------------------


class TestHostingPrefixRegistration:
    def test_hosting_prefix_registered_on_import(self) -> None:
        # Importing the subpackage executes its registration code path.
        # The fixture clears the registry before each test, so we explicitly
        # re-trigger registration here (mirrors what would happen on a
        # fresh import in a real process).
        import langchain_azure_ai.agents.hosting as hosting

        _user_agent.add_user_agent_prefix(hosting.HOSTING_USER_AGENT)
        assert hosting.HOSTING_USER_AGENT.startswith(
            "langchain_azure_ai.agents.hosting/"
        )
        assert hosting.HOSTING_USER_AGENT in _user_agent._user_agent_prefixes

    def test_get_user_agent_includes_hosting_prefix(self) -> None:
        import langchain_azure_ai.agents.hosting as hosting

        _user_agent.add_user_agent_prefix(hosting.HOSTING_USER_AGENT)
        ua = _user_agent.get_user_agent()
        assert ua.startswith(hosting.HOSTING_USER_AGENT + " ")
        assert ua.endswith(_user_agent.BASE_USER_AGENT)

    def test_re_register_is_noop(self) -> None:
        import langchain_azure_ai.agents.hosting as hosting

        _user_agent.add_user_agent_prefix(hosting.HOSTING_USER_AGENT)
        _user_agent.add_user_agent_prefix(hosting.HOSTING_USER_AGENT)
        ua = _user_agent.get_user_agent()
        assert ua.count(hosting.HOSTING_USER_AGENT) == 1


# ---------------------------------------------------------------------------
# Foundry hosted-env detection
# ---------------------------------------------------------------------------


class TestHostedEnvDetection:
    def test_env_var_sets_detected_flag(self) -> None:
        with patch.dict(os.environ, {"FOUNDRY_HOSTING_ENVIRONMENT": "1"}):
            _user_agent._detect_hosted_environment()
            assert _user_agent._hosted_env_detected is True
            assert _user_agent.is_hosted_environment() is True

    def test_unset_env_var_without_agentserver_leaves_flag_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOUNDRY_HOSTING_ENVIRONMENT", None)
            with patch("importlib.util.find_spec", return_value=None):
                _user_agent._detect_hosted_environment()
                assert _user_agent._hosted_env_detected is False

    def test_agentserver_spec_probe_failure_leaves_flag_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOUNDRY_HOSTING_ENVIRONMENT", None)
            with patch("importlib.util.find_spec", side_effect=ValueError("boom")):
                _user_agent._detect_hosted_environment()
                assert _user_agent._hosted_env_detected is False

    def test_detection_is_cached(self) -> None:
        with patch.dict(os.environ, {"FOUNDRY_HOSTING_ENVIRONMENT": "1"}):
            _user_agent._detect_hosted_environment()
        # Even after env var is cleared, the cached flag remains.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOUNDRY_HOSTING_ENVIRONMENT", None)
            with patch("importlib.util.find_spec", return_value=None) as find_spec:
                _user_agent._detect_hosted_environment()
                find_spec.assert_not_called()
            assert _user_agent._hosted_env_detected is True


# ---------------------------------------------------------------------------
# Hosting subpackage: AZURE_HTTP_USER_AGENT env var
# ---------------------------------------------------------------------------


class TestHostingAzureHttpUserAgent:
    @staticmethod
    @contextmanager
    def _reload_hosting_with_env(
        env_value: str | None,
    ) -> Iterator[Any]:
        """Reload the hosting module under a controlled ``os.environ``.

        Snapshots and restores both ``AZURE_HTTP_USER_AGENT`` and the
        four ``openai`` SDK client class ``__init__``s, since reloading
        the hosting module also re-runs the openai SDK monkey-patch
        installer.
        """
        import langchain_azure_ai.agents.hosting as hosting

        saved_inits: dict[type, Any] = {}
        try:
            openai_mod = importlib.import_module("openai")
        except ImportError:
            openai_mod = None
        if openai_mod is not None:
            for cls_name in (
                "OpenAI",
                "AsyncOpenAI",
                "AzureOpenAI",
                "AsyncAzureOpenAI",
            ):
                cls = getattr(openai_mod, cls_name, None)
                if cls is not None:
                    saved_inits[cls] = cls.__init__

        saved_env = os.environ.get("AZURE_HTTP_USER_AGENT")
        try:
            if env_value is None:
                os.environ.pop("AZURE_HTTP_USER_AGENT", None)
            else:
                os.environ["AZURE_HTTP_USER_AGENT"] = env_value
            importlib.reload(hosting)
            yield hosting
        finally:
            if saved_env is None:
                os.environ.pop("AZURE_HTTP_USER_AGENT", None)
            else:
                os.environ["AZURE_HTTP_USER_AGENT"] = saved_env
            for cls, init in saved_inits.items():
                cls.__init__ = init  # type: ignore[method-assign]

    def test_env_var_is_set_on_import(self) -> None:
        """With no pre-existing env var, hosting injects its prefix."""
        with self._reload_hosting_with_env(None) as hosting:
            assert (
                os.environ.get("AZURE_HTTP_USER_AGENT")
                == hosting.HOSTING_USER_AGENT
            )

    def test_caller_value_wins(self) -> None:
        """Hosting's ``setdefault`` preserves a caller-supplied value."""
        with self._reload_hosting_with_env("caller/1.0") as hosting:
            assert os.environ["AZURE_HTTP_USER_AGENT"] == "caller/1.0"
            # Sanity: the hosting prefix exists but was not used here.
            assert hosting.HOSTING_USER_AGENT.startswith(
                "langchain_azure_ai.agents.hosting/"
            )


# ---------------------------------------------------------------------------
# Hosting subpackage: openai SDK UA stamping
# ---------------------------------------------------------------------------


class TestHostingOpenAIUserAgentStamp:
    """End-to-end behavior of the ``openai``-SDK ``__init__`` monkey-patch.

    The hosting subpackage installs the patch at import time; the
    autouse fixture clears the prefix registry per test, so we re-add
    the hosting prefix in each case before constructing a client.
    """

    def _hosting_prefix(self) -> str:
        import langchain_azure_ai.agents.hosting as hosting

        _user_agent.add_user_agent_prefix(hosting.HOSTING_USER_AGENT)
        return hosting.HOSTING_USER_AGENT

    def test_async_openai_client_carries_hosting_prefix(self) -> None:
        openai = pytest.importorskip("openai")

        prefix = self._hosting_prefix()
        client = openai.AsyncOpenAI(api_key="test-key")
        ua = client._custom_headers["User-Agent"]
        assert ua.startswith(prefix + " ")
        # SDK's own UA token preserved as suffix.
        assert "AsyncOpenAI/Python" in ua

    def test_sync_openai_client_carries_hosting_prefix(self) -> None:
        openai = pytest.importorskip("openai")

        prefix = self._hosting_prefix()
        client = openai.OpenAI(api_key="test-key")
        ua = client._custom_headers["User-Agent"]
        assert ua.startswith(prefix + " ")
        assert "OpenAI/Python" in ua

    def test_caller_default_headers_user_agent_preserved(self) -> None:
        openai = pytest.importorskip("openai")

        prefix = self._hosting_prefix()
        client = openai.AsyncOpenAI(
            api_key="test-key",
            default_headers={"User-Agent": "my-app/1.0"},
        )
        ua = client._custom_headers["User-Agent"]
        assert ua.startswith(prefix + " ")
        assert "my-app/1.0" in ua

    def test_idempotent_no_double_stamp(self) -> None:
        openai = pytest.importorskip("openai")

        prefix = self._hosting_prefix()
        # Re-running the installer must be a no-op.
        import langchain_azure_ai.agents.hosting as hosting

        hosting._install_openai_user_agent_stamp()
        hosting._install_openai_user_agent_stamp()

        client = openai.AsyncOpenAI(api_key="test-key")
        ua = client._custom_headers["User-Agent"]
        assert ua.count(prefix) == 1

    def test_opt_out_skips_stamping(self) -> None:
        openai = pytest.importorskip("openai")

        self._hosting_prefix()
        with patch.dict(
            os.environ,
            {_user_agent.USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: "1"},
        ):
            client = openai.AsyncOpenAI(api_key="test-key")
            ua = client._custom_headers.get("User-Agent", "")
            # Hosting prefix must not be injected when telemetry is off.
            assert "langchain_azure_ai.agents.hosting/" not in ua

    def test_install_is_noop_when_openai_missing(self) -> None:
        """When ``openai`` is not installed, the installer returns cleanly."""
        import langchain_azure_ai.agents.hosting as hosting

        original_flag = hosting._OPENAI_INIT_PATCHED
        try:
            hosting._OPENAI_INIT_PATCHED = False
            with patch(
                "importlib.import_module", side_effect=ImportError("no openai")
            ):
                # Must not raise.
                hosting._install_openai_user_agent_stamp()
            # Flag stayed False since we never patched anything.
            assert hosting._OPENAI_INIT_PATCHED is False
        finally:
            hosting._OPENAI_INIT_PATCHED = original_flag

    def test_install_isolates_per_class_failure(self) -> None:
        """A failure on one client class must not break the others or import."""
        openai = pytest.importorskip("openai")
        import langchain_azure_ai.agents.hosting as hosting

        # Simulate a future SDK where one of the four client classes
        # raises while being patched (e.g. ``__init__`` non-writable,
        # module ``__getattr__`` raising, etc.). The installer must
        # swallow the per-class failure and continue patching the rest.
        class _BadMeta(type):
            def __setattr__(cls, name: str, value: Any) -> None:
                if name == "__init__":
                    raise TypeError("simulated immutable __init__")
                super().__setattr__(name, value)

        class _BadOpenAI(metaclass=_BadMeta):
            pass

        original_flag = hosting._OPENAI_INIT_PATCHED
        original_async_init = openai.AsyncOpenAI.__init__
        try:
            hosting._OPENAI_INIT_PATCHED = False

            with patch.object(openai, "OpenAI", _BadOpenAI):
                # Must not raise even though _BadOpenAI patching fails.
                hosting._install_openai_user_agent_stamp()

            prefix = self._hosting_prefix()
            # The other (real) classes were still patched successfully.
            client = openai.AsyncOpenAI(api_key="test-key")
            ua = client._custom_headers["User-Agent"]
            assert ua.startswith(prefix + " ")
        finally:
            # Restore real __init__ on the class we wrapped.
            openai.AsyncOpenAI.__init__ = original_async_init  # type: ignore[method-assign]
            hosting._OPENAI_INIT_PATCHED = original_flag
