"""Utilities for environment variables."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

# ---------------------------------------------------------------------------
# Project-endpoint resolution
# ---------------------------------------------------------------------------

# Ordered list of environment variable names that hold the Azure AI Foundry
# project endpoint.  The first variable that is set wins, so
# AZURE_AI_PROJECT_ENDPOINT always takes precedence over the alias.
PROJECT_ENDPOINT_ENV_VARS: list[str] = [
    "AZURE_AI_PROJECT_ENDPOINT",
    "FOUNDRY_PROJECT_ENDPOINT",
]


def get_project_endpoint(
    data: Optional[dict[str, Any]] = None,
    *,
    nullable: bool = False,
) -> Optional[str]:
    """Resolve the Azure AI Foundry project endpoint.

    Resolution order:

    1. ``data["project_endpoint"]`` when *data* is provided and the key is set.
    2. ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
    3. ``FOUNDRY_PROJECT_ENDPOINT`` environment variable.

    Args:
        data: Optional mapping that may contain a ``project_endpoint`` key
            (e.g. Pydantic ``values`` dict inside a validator).
        nullable: When ``True``, return ``None`` instead of raising
            ``ValueError`` if the endpoint cannot be resolved.

    Returns:
        The resolved project endpoint string, or ``None`` when *nullable* is
        ``True`` and no value is found.

    Raises:
        ValueError: When the endpoint cannot be resolved and *nullable* is
            ``False``.
    """
    return get_from_dict_or_env(
        data or {},
        "project_endpoint",
        PROJECT_ENDPOINT_ENV_VARS,
        nullable=nullable,
    )


def get_from_dict_or_env(
    data: dict[str, Any],
    key: Union[str, list[str]],
    env_key: Union[str, list[str]],
    default: Optional[str] = None,
    nullable: bool = False,
) -> Optional[str]:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary. This can be a list of keys to try
            in order.
        env_key: The environment variable (or ordered list of environment variables)
            to look up if the key is not in the dictionary.  When a list is given the
            variables are tried in order and the first match wins.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.
        nullable: Whether to allow None values. Defaults to False.

    Returns:
        The dict value or the environment variable value.
    """
    if isinstance(key, (list, tuple)):
        for k in key:
            if value := data.get(k):
                return value

    if isinstance(key, str) and key in data and data[key]:
        return data[key]

    key_for_err = key[0] if isinstance(key, (list, tuple)) else key

    return get_from_env(key_for_err, env_key, default=default, nullable=nullable)


def get_from_env(
    key: str,
    env_key: Union[str, list[str]],
    default: Optional[str] = None,
    nullable: bool = False,
) -> Optional[str]:
    """Get a value from a dictionary or an environment variable.

    Args:
        key: The key to look up in the dictionary.
        env_key: The environment variable (or ordered list of environment variables)
            to look up if the key is not in the dictionary.  When a list is given the
            variables are tried in order and the first match wins.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.
        nullable: Whether to allow None values. Defaults to False.

    Returns:
        str: The value of the key.

    Raises:
        ValueError: If the key is not in the dictionary and no default value is
            provided or if the environment variable is not set.
    """
    env_keys: list[str] = [env_key] if isinstance(env_key, str) else list(env_key)
    for k in env_keys:
        if env_value := os.getenv(k):
            return env_value
    if default is not None or nullable:
        return default
    primary_key = env_keys[0] if env_keys else "(unknown)"
    msg = (
        f"Did not find {key}, please add an environment variable"
        f" `{primary_key}` which contains it, or pass"
        f" `{key}` as a named parameter."
    )
    raise ValueError(msg)
