"""Configuration for the Azure Blob Storage backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AzureBlobConfig:
    """Configuration for AzureBlobBackend.

    Five authentication methods are supported (mutually exclusive):

    1. **Connection string** -- set ``connection_string`` only.  Do **not** set
       ``account_url`` or any credential field alongside it.
    2. **Account key** -- set ``account_url`` + ``account_key``.
    3. **SAS token** -- set ``account_url`` + ``sas_token``.
    4. **Credential object** -- set ``account_url`` + ``credential`` (any
       Azure credential instance, e.g. ``ClientSecretCredential``).
    5. **Default (AAD)** -- set ``account_url`` only. Uses
       ``DefaultAzureCredential`` automatically.

    Raises:
        ValueError: If more than one credential source is provided, or if
            ``account_url`` is missing when required.
    """

    account_url: str = ""
    """Account URL (e.g., ``https://<account>.blob.core.windows.net``)."""

    container_name: str = ""
    """Target blob container name."""

    prefix: str = ""
    """Key namespace prefix within the container.

    Supports multi-agent isolation by scoping each agent to a prefix.
    """

    credential: Any = None
    """Azure credential object (e.g., ``ClientSecretCredential``).

    Mutually exclusive with ``connection_string``, ``account_key``, and
    ``sas_token``.
    """

    account_key: Optional[str] = None
    """Azure Storage account key for shared-key authentication.

    Mutually exclusive with ``connection_string``, ``sas_token``, and
    ``credential``.
    """

    sas_token: Optional[str] = None
    """Shared Access Signature token string.

    A leading ``?`` is accepted and stripped during validation. Mutually
    exclusive with ``connection_string``, ``account_key``, and ``credential``.
    """

    max_concurrency: int = 8
    """Maximum parallel blob operations for grep/glob."""

    encoding: str = "utf-8"
    """Text encoding for blob content."""

    connection_string: Optional[str] = None
    """Full connection string (e.g., from Azure Portal or for Azurite).

    Mutually exclusive with ``account_key``, ``sas_token``, and ``credential``.
    """

    api_version: str = "2025-11-05"
    """Azure Storage API version string."""

    def __post_init__(self) -> None:
        """Validate that at most one explicit credential source is configured."""
        # Reject empty strings -- credentials must be None or non-empty.
        for field_name in ("connection_string", "account_key", "sas_token"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(
                    f"{field_name} must be None or a non-empty string, "
                    f"got empty string."
                )

        # Normalize SAS token: strip whitespace and leading "?" so downstream
        # code doesn't need to.
        if self.sas_token is not None:
            normalized = self.sas_token.strip().lstrip("?")
            if not normalized:
                raise ValueError(
                    "sas_token contains only '?' characters -- expected a "
                    "valid token string."
                )
            self.sas_token = normalized

        cred_sources = [
            ("connection_string", self.connection_string is not None),
            ("account_key", self.account_key is not None),
            ("sas_token", self.sas_token is not None),
            ("credential", self.credential is not None),
        ]
        active = [name for name, is_set in cred_sources if is_set]

        if len(active) > 1:
            raise ValueError(
                f"Only one authentication method may be set, got: "
                f"{', '.join(active)}. Choose one of: connection_string, "
                f"account_key, sas_token, credential, or omit all to use "
                f"DefaultAzureCredential."
            )

        # connection_string is self-contained and must not be combined with
        # account_url.
        if self.connection_string and self.account_url.strip():
            raise ValueError(
                "connection_string and account_url are mutually exclusive. "
                "A connection string already contains the account endpoint."
            )

        # All other auth paths need account_url.
        if not self.connection_string and not self.account_url.strip():
            raise ValueError(
                "account_url is required unless connection_string is provided."
            )
