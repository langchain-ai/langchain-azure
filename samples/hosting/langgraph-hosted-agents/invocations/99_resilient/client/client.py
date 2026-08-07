"""Launch the full-screen client for the durable Invocations sample."""

from __future__ import annotations

import argparse
import asyncio
from uuid import uuid4

import httpx
from app import InvocationsCuiApp
from azure.identity.aio import DefaultAzureCredential
from conversation import Conversation

_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


def _invocations_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/invocations"):
        return normalized
    return f"{normalized}/invocations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the resilient LangGraph agent CUI."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8088",
        help="Agent host base URL or full protocol endpoint URL.",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Add an Azure AI bearer token from DefaultAzureCredential.",
    )
    parser.add_argument(
        "--reconnect-timeout",
        type=float,
        default=120.0,
        help="Seconds to retry a disconnected request.",
    )
    return parser


async def amain(args: argparse.Namespace) -> None:
    credential: DefaultAzureCredential | None = None
    headers: dict[str, str] = {}
    if args.auth:
        credential = DefaultAzureCredential()
        token = await credential.get_token(_AZURE_AI_SCOPE)
        headers["Authorization"] = f"Bearer {token.token}"

    try:
        async with httpx.AsyncClient(headers=headers, timeout=None) as client:
            conversation = Conversation(
                client,
                _invocations_url(args.url),
                session_id=str(uuid4()),
                reconnect_timeout=args.reconnect_timeout,
            )
            await InvocationsCuiApp(conversation).run_async()
    finally:
        if credential is not None:
            await credential.close()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
