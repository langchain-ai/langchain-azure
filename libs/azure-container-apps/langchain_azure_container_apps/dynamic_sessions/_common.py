"""Shared plumbing for the dynamic-sessions tools and Deep Agents backend.

Lives below both so the backend does not have to import a tool module to
authenticate or build a URL.
"""

from __future__ import annotations

import importlib.metadata
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, Optional

from azure.core.credentials import AccessToken
from azure.identity import DefaultAzureCredential

_DISTRIBUTION = "langchain-azure-container-apps"

try:
    _package_version = importlib.metadata.version(_DISTRIBUTION)
except importlib.metadata.PackageNotFoundError:
    _package_version = "0.0.0"

USER_AGENT = f"{_DISTRIBUTION}/{_package_version} (Language=Python)"

_TOKEN_SCOPE = "https://dynamicsessions.io/.default"

# Every `requests` call against the data plane carries this timeout: without
# one a wedged connection blocks the calling tool forever. Generous because
# executions are synchronous server-side -- the response arrives only when the
# command finishes.
REQUEST_TIMEOUT = 120

# Refresh this far before expiry so a token cannot lapse mid-request.
_TOKEN_REFRESH_MARGIN = timedelta(minutes=5)


def build_session_url(
    pool_management_endpoint: str,
    path: str,
    *,
    session_id: str,
    api_version: str,
    params: Optional[Mapping[str, str]] = None,
) -> str:
    """Build a session-scoped data-plane URL.

    Args:
        pool_management_endpoint: Session pool management endpoint.
        path: Resource path to append, e.g. ``"executions"``.
        session_id: Session identifier, sent as the ``identifier`` parameter.
        api_version: Service API version. Deliberately not defaulted -- the
            tools and the backend are pinned to different versions, and
            silently moving a caller to another one changes the contract it
            talks to.
        params: Extra query parameters, encoded alongside the standard ones.

    Returns:
        The absolute URL.

    Raises:
        ValueError: If the endpoint is empty, or carries a query or fragment.
    """
    if not pool_management_endpoint:
        raise ValueError("pool_management_endpoint is not set")

    parts = urllib.parse.urlsplit(pool_management_endpoint)
    # Reject rather than accommodate. The previous implementation appended the
    # resource path *after* an existing query, yielding
    # `https://host/pool?x=1/executions&identifier=...`, and no real pool
    # management endpoint carries either component.
    if parts.query or parts.fragment:
        raise ValueError(
            "pool_management_endpoint must not contain a query string or "
            f"fragment; got {pool_management_endpoint!r}"
        )

    # Extras first, standard last: a caller-supplied key can never silently
    # replace the session identifier or the API version.
    query = urllib.parse.urlencode(
        {**(params or {}), "identifier": session_id, "api-version": api_version},
        # `identifier` is caller-supplied: percent-encode it fully, so a value
        # containing "/" cannot extend the URL path.
        quote_via=urllib.parse.quote,
        safe="",
    )
    return f"{parts.geturl().rstrip('/')}/{path.lstrip('/')}?{query}"


def access_token_provider_factory() -> Callable[[], Optional[str]]:
    """Build a token provider backed by `DefaultAzureCredential`.

    The credential is created lazily on first use (the factory runs at import
    time as a class-level default) and then reused: it holds its own token
    cache and HTTP pipeline, so building a fresh one per refresh would discard
    both. The closure is not locked -- two threads refreshing at once fetch a
    duplicate token, which is wasteful but harmless.

    Returns:
        A callable returning a bearer token for the dynamic-sessions scope,
        cached until shortly before it expires.
    """
    access_token: Optional[AccessToken] = None
    credential: Optional[DefaultAzureCredential] = None

    def access_token_provider() -> Optional[str]:
        nonlocal access_token, credential
        if (
            access_token is None
            or datetime.fromtimestamp(access_token.expires_on, timezone.utc)
            < datetime.now(timezone.utc) + _TOKEN_REFRESH_MARGIN
        ):
            if credential is None:
                credential = DefaultAzureCredential()
            access_token = credential.get_token(_TOKEN_SCOPE)
        return access_token.token

    return access_token_provider


def auth_headers(access_token: Optional[str]) -> dict[str, str]:
    """Build the Authorization/User-Agent pair every session request carries.

    Raises:
        ValueError: If `access_token` is falsy -- sending
            ``Authorization: Bearer None`` would only surface later as an
            opaque service 401.
    """
    if not access_token:
        raise ValueError(
            "access_token_provider returned no token; check the credential "
            "configuration"
        )
    return {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
    }
