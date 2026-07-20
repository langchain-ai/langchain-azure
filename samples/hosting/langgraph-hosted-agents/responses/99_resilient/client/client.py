"""Launch the full-screen client for the resilient Responses sample."""

from __future__ import annotations

import argparse
import asyncio

from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential

from app import ResponsesCuiApp
from conversation import Conversation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the resilient LangGraph Responses CUI."
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Foundry project endpoint copied from the portal.",
    )
    parser.add_argument("--agent", required=True, help="Hosted agent name.")
    parser.add_argument(
        "--tokendelay",
        help="Server-side delay in seconds between fake-model tokens.",
    )
    return parser


async def amain(args: argparse.Namespace) -> None:
    async with (
        DefaultAzureCredential() as credential,
        AIProjectClient(
            endpoint=args.endpoint,
            credential=credential,
            allow_preview=True,
        ) as project,
        project.get_openai_client(agent_name=args.agent) as client,
    ):
        conversation = Conversation(client, token_delay=args.tokendelay)
        await ResponsesCuiApp(conversation).run_async()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
