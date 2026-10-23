"""Run research, preserve two named reports, and download both publications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from openai import OpenAI, omit
from openai.types.responses import Response


def download_report(response: Response, report_name: str, directory: Path) -> None:
    """Download a successful named publication and verify its digest.

    Args:
        response: Completed response containing publish_report tool output.
        report_name: Name selected for this demo turn.
        directory: Local destination for the downloaded Markdown file.

    Raises:
        RuntimeError: Publication is missing, targets another location, or fails
            digest verification. A handled tool failure never creates a file.
        httpx.HTTPError: Download request fails.
    """
    items = response.model_dump()["output"]
    call_ids = {
        item["call_id"]
        for item in items
        if item.get("type") == "function_call"
        and item.get("name") == "publish_report"
        and json.loads(item["arguments"]).get("report_name") == report_name
    }
    publications = []
    for item in items:
        if (
            item.get("type") != "function_call_output"
            or item.get("call_id") not in call_ids
        ):
            continue
        try:
            published = json.loads(item["output"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(published, dict) and published.get("download_url"):
            publications.append(published)
    if not publications:
        raise RuntimeError(
            f"No download link for {report_name}. The report remains saved; "
            f"continue from response {response.id} after checking the publishing error."
        )
    published = publications[-1]
    url = urlsplit(published["download_url"])
    account = urlsplit(os.environ["REPORT_STORAGE_ACCOUNT_URL"])
    container = os.environ["REPORT_STORAGE_CONTAINER"]
    if (
        published.get("file_path") != f"/{report_name}"
        or url.scheme != "https"
        or url.netloc != account.netloc
        or re.fullmatch(
            rf"/{re.escape(container)}/[0-9a-f]{{32}}/{re.escape(report_name)}",
            url.path,
        )
        is None
    ):
        raise RuntimeError(
            "Publication does not match the selected report and Storage."
        )
    downloaded = httpx.get(published["download_url"], timeout=30)
    downloaded.raise_for_status()
    if hashlib.sha256(downloaded.content).hexdigest() != published.get("sha256"):
        raise RuntimeError("Downloaded report does not match the publication digest.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / report_name
    destination.write_bytes(downloaded.content)
    print(f"Downloaded {destination}")


def run_demo(client: OpenAI, directory: Path, sessions: set[str]) -> None:
    """Generate, read and revise named reports using the latest response ID.

    Args:
        client: Caller-owned local or Foundry agent OpenAI client.
        directory: Destination for report-v1.md and report-v2.md.
        sessions: Updated with returned hosted session IDs for caller cleanup.

    Raises:
        RuntimeError: A turn fails, the hosted session changes, or publishing fails.
    """
    previous_response_id: str | None = None
    session_id: str | None = None

    def ask(question: str) -> Response:
        nonlocal previous_response_id, session_id
        response = client.responses.create(
            input=question,
            previous_response_id=previous_response_id or omit,
            extra_body={"agent_session_id": session_id} if session_id else {},
        )
        returned_session = (response.model_extra or {}).get("agent_session_id")
        if returned_session:
            sessions.add(returned_session)
            if session_id and returned_session != session_id:
                raise RuntimeError("Continuation changed its hosted session.")
            session_id = returned_session
        previous_response_id = response.id
        print(f"Response: {response.id}")
        if session_id:
            print(f"Session: {session_id}")
        print(response.output_text)
        if response.status != "completed":
            raise RuntimeError(f"Turn {response.id} did not complete: {response.error}")
        return response

    original = ask(
        "Research how LangGraph checkpoints support multi-turn agents. Use one "
        "research-agent and cite primary sources. Save /report-v1.md, read it back, "
        "and publish_report with report_name='report-v1.md'."
    )
    download_report(original, "report-v1.md", directory)
    ask(
        "Use read_file on all of /report-v1.md and return it verbatim. Do not research or publish again."
    )
    revised = ask(
        "Read /report-v1.md, write an unchanged copy to /report-v2.md, then use "
        "edit_file on that copy to add a Recovery limitations section explaining "
        "that saved checkpoints do not resume arbitrary in-flight work. Preserve "
        "the original findings, citations and URLs in both files. Read the entire "
        "revised file and publish_report(report_name='report-v2.md')."
    )
    download_report(revised, "report-v2.md", directory)
    ask(
        "Use read_file on the entire /report-v2.md and return it verbatim. "
        "Do not research, edit or publish another copy."
    )


def main() -> None:
    """Run locally by default, or target --agent-name in the configured project."""
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", default=os.environ.get("DEEP_AGENTS_NAME"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--output-dir", type=Path, default=Path.home() / "Downloads" / "deep-research"
    )
    args = parser.parse_args()
    sessions: set[str] = set()
    with ExitStack() as resources:
        client: OpenAI
        if args.agent_name:
            credential = resources.enter_context(DefaultAzureCredential())
            project = resources.enter_context(
                AIProjectClient(
                    endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                    credential=credential,
                )
            )
            client = resources.enter_context(
                project.get_openai_client(
                    agent_name=args.agent_name, timeout=600, max_retries=0
                )
            )
        else:
            client = resources.enter_context(
                OpenAI(
                    base_url=args.base_url, api_key="local", timeout=600, max_retries=0
                )
            )
        try:
            run_demo(client, args.output_dir, sessions)
        finally:
            if args.agent_name:
                failures = []
                for session_id in sorted(sessions):
                    try:
                        project.agents.stop_session(args.agent_name, session_id)
                    except AzureError:
                        failures.append(session_id)
                if failures:
                    raise RuntimeError(f"Could not stop test sessions: {failures}")


if __name__ == "__main__":
    main()
