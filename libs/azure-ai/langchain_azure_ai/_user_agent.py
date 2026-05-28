# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared User-Agent helpers for ``langchain-azure-ai``.

Stamps outbound HTTP traffic with a canonical token of the form::

    langchain-azure-ai/<version>

Subpackages (currently :mod:`langchain_azure_ai.agents.hosting`) may
register additional prefixes via :func:`add_user_agent_prefix` to
identify themselves in Azure-side telemetry. Registered prefixes are
prepended to the base token by :func:`get_user_agent`, yielding e.g.::

    langchain_azure_ai.agents.hosting/<host-ver> langchain-azure-ai/<pkg-ver>

The :envvar:`LANGCHAIN_AZURE_AI_USER_AGENT_DISABLED` environment
variable opts the process out — :func:`get_user_agent` then returns the
empty string and :func:`with_user_agent` becomes a no-op.

Modelled after ``agent_framework._telemetry`` from the Microsoft Agent
Framework so the two layers compose cleanly when a user mixes both.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import logging
import os
from typing import Any, Final

logger = logging.getLogger(__name__)

_PACKAGE_NAME: Final[str] = "langchain-azure-ai"
try:
    _PACKAGE_VERSION: str = importlib.metadata.version(_PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    _PACKAGE_VERSION = "0.0.0"

USER_AGENT_KEY: Final[str] = "User-Agent"
"""Canonical HTTP header name for the User-Agent value."""

BASE_USER_AGENT: Final[str] = f"{_PACKAGE_NAME}/{_PACKAGE_VERSION}"
"""Base UA token for this package, e.g. ``langchain-azure-ai/1.2.3``."""

USER_AGENT_TELEMETRY_DISABLED_ENV_VAR: Final[str] = (
    "LANGCHAIN_AZURE_AI_USER_AGENT_DISABLED"
)
"""Env var that, when truthy, suppresses the langchain-azure-ai UA prefix."""

_FOUNDRY_HOSTING_ENV_VAR: Final[str] = "FOUNDRY_HOSTING_ENVIRONMENT"

# Insertion-ordered prefix registry (dict-as-ordered-set).
_user_agent_prefixes: "dict[str, None]" = {}
_hosted_env_detected: bool = False


def _telemetry_enabled() -> bool:
    """Return ``True`` unless the opt-out env var is truthy."""
    return os.environ.get(
        USER_AGENT_TELEMETRY_DISABLED_ENV_VAR, "false"
    ).lower() not in ("true", "1")


def add_user_agent_prefix(prefix: str) -> None:
    """Register an extra UA prefix (idempotent).

    Prefixes are emitted by :func:`get_user_agent` in insertion order,
    space-separated, before :data:`BASE_USER_AGENT`. Subpackages should
    use this from their ``__init__`` to identify themselves, e.g.
    ``"langchain_azure_ai.agents.hosting/1.2.3"``.

    Args:
        prefix: The token to register. Empty strings are ignored.
    """
    if prefix:
        _user_agent_prefixes.setdefault(prefix, None)


def _detect_hosted_environment() -> None:
    """Detect if running inside Azure AI Foundry hosting.

    Sets a flag on first call and is a no-op thereafter. Hosting layers
    auto-register their own prefix via :func:`add_user_agent_prefix`;
    this function only flips ``_hosted_env_detected`` so callers know
    detection ran. Detection logic mirrors
    ``agent_framework._telemetry._detect_hosted_environment``: env var
    first, then a lazy ``find_spec`` fallback against
    ``azure.ai.agentserver.core``.
    """
    global _hosted_env_detected
    if _hosted_env_detected:
        return

    if os.environ.get(_FOUNDRY_HOSTING_ENV_VAR):
        _hosted_env_detected = True
        return

    if importlib.util.find_spec("azure.ai.agentserver.core") is None:
        return

    with contextlib.suppress(ImportError, AttributeError):
        from azure.ai.agentserver.core import (  # type: ignore[import-not-found]
            AgentConfig,
        )

        if AgentConfig.from_env().is_hosted:
            _hosted_env_detected = True


def is_hosted_environment() -> bool:
    """Return whether the current process is running inside Foundry hosting."""
    _detect_hosted_environment()
    return _hosted_env_detected


def get_user_agent() -> str:
    """Return the combined User-Agent string.

    Format: ``"<prefix1> <prefix2> ... langchain-azure-ai/<ver>"`` when
    prefixes have been registered, otherwise just
    ``"langchain-azure-ai/<ver>"``.

    Returns an empty string when telemetry is disabled via
    :envvar:`LANGCHAIN_AZURE_AI_USER_AGENT_DISABLED`.
    """
    if not _telemetry_enabled():
        return ""
    _detect_hosted_environment()
    if not _user_agent_prefixes:
        return BASE_USER_AGENT
    return " ".join((*_user_agent_prefixes.keys(), BASE_USER_AGENT))


def with_user_agent(headers: "dict[str, Any] | None" = None) -> "dict[str, Any]":
    """Return ``headers`` with our UA prepended to any existing ``User-Agent``.

    When ``headers`` is ``None`` and telemetry is enabled, a new dict is
    returned containing only the ``User-Agent`` entry. When telemetry is
    disabled the input is returned unchanged (or an empty dict for
    ``None`` input).

    Args:
        headers: Existing header dict. Not mutated; a shallow copy is
            returned.

    Returns:
        A new headers dict with the UA stamped in.
    """
    if not _telemetry_enabled():
        return dict(headers) if headers else {}
    ua = get_user_agent()
    if not headers:
        return {USER_AGENT_KEY: ua}
    headers = dict(headers)
    headers[USER_AGENT_KEY] = (
        f"{ua} {headers[USER_AGENT_KEY]}" if USER_AGENT_KEY in headers else ua
    )
    return headers


def reset_for_testing() -> None:
    """Clear the prefix registry and detection cache.

    Tests that toggle env vars or simulate hosting detection should call
    this to keep state isolated.
    """
    global _hosted_env_detected
    _user_agent_prefixes.clear()
    _hosted_env_detected = False
