"""Launch the full-screen client for the resilient Responses sample."""

from __future__ import annotations

import argparse
import asyncio
from uuid import uuid4

from app import ResponsesCuiApp
from azure.identity.aio import DefaultAzureCredential
from conversation import Conversation
from openai import AsyncOpenAI

_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


def _openai_base_url(responses_url: str) -> str:
    normalized = responses_url.rstrip("/")
    return normalized.removesuffix("/responses")


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
        "--session-id",
        help="Stable session ID. A random ID is generated when omitted.",
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

    async def get_token() -> str:
        assert credential is not None
        token = await credential.get_token(_AZURE_AI_SCOPE)
        return token.token

    api_key = "local"
    if args.auth:
        credential = DefaultAzureCredential()
        api_key = get_token

    try:
        async with AsyncOpenAI(
            base_url=_openai_base_url(args.url),
            api_key=api_key,
            max_retries=0,
        ) as client:
            conversation = Conversation(
                client,
                conversation_id=args.session_id or str(uuid4()),
                reconnect_timeout=args.reconnect_timeout,
            )
            await ResponsesCuiApp(conversation).run_async()
    finally:
        if credential is not None:
            await credential.close()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
