"""Publish the saved virtual report without changing its working copy."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from azure.core.credentials import TokenCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)
from deepagents.backends.utils import file_data_to_string
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool


def create_report_export_tool(credential: TokenCredential) -> BaseTool:
    """Create a tool that uploads the current report using caller-owned credentials.

    Args:
        credential: Azure credential kept alive by the application's exit stack.

    Returns:
        A coordinator tool returning a one-hour, read-only Blob download URL.

    Raises:
        KeyError: The Storage account URL or container is not configured.
    """
    account_url = os.environ["REPORT_STORAGE_ACCOUNT_URL"]
    container_name = os.environ["REPORT_STORAGE_CONTAINER"]

    @tool
    def publish_report(runtime: ToolRuntime) -> dict[str, str]:
        """Upload the current /final_report.md and return its download URL.

        Read the report from conversation state, never from model-supplied text.
        Call after saving and checking a completed report or revision. The virtual
        file remains available for later edits. Each export is a new private blob.

        Args:
            runtime: Injected tool runtime containing the conversation's files.

        Returns:
            File path, SHA256, read-only HTTPS download URL, and its UTC expiry.

        Raises:
            ValueError: The saved report is missing, empty, or not UTF-8 text.
            azure.core.exceptions.AzureError: Upload or SAS authorization fails.
        """
        file_data = runtime.state.get("files", {}).get("/final_report.md")
        if file_data is None or file_data.get("encoding", "utf-8") != "utf-8":
            raise ValueError("Save a UTF-8 /final_report.md before publishing it.")
        report = file_data_to_string(file_data)
        if not report.strip():
            raise ValueError("The saved /final_report.md is empty.")
        content = report.encode("utf-8")
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=5)
        expiry = now + timedelta(hours=1)
        blob_name = f"{uuid4().hex}/final_report.md"
        with BlobServiceClient(account_url, credential=credential) as storage:
            account_name = storage.account_name
            if not account_name:
                raise ValueError("Configure the Azure Blob Storage account URL.")
            delegation_key = storage.get_user_delegation_key(start, expiry)
            signature = generate_blob_sas(
                account_name=account_name,
                container_name=container_name,
                blob_name=blob_name,
                user_delegation_key=delegation_key,
                permission=BlobSasPermissions(read=True),
                start=start,
                expiry=expiry,
                protocol="https",
            )
            blob = storage.get_blob_client(container_name, blob_name)
            blob.upload_blob(
                content,
                overwrite=False,
                content_settings=ContentSettings(
                    content_type="text/markdown; charset=utf-8",
                    content_disposition='attachment; filename="final_report.md"',
                ),
            )
            return {
                "file_path": "/final_report.md",
                "sha256": hashlib.sha256(content).hexdigest(),
                "download_url": f"{blob.url}?{signature}",
                "expires_at": expiry.isoformat(),
            }

    return publish_report
