"""Create two response branches and continue each one independently.

Run ``python client.py`` after starting ``python main.py``.
"""

import argparse
from collections.abc import Callable
from typing import Any

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def _create_response(
    client: httpx.Client,
    input: str,
    previous_response_id: str | None,
    token_provider: Callable[[], str] | None,
) -> str:
    body: dict[str, Any] = {"input": input, "store": True, "stream": False}

    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id

    if token_provider:
        headers = {"Authorization": f"Bearer {token_provider()}"}

    response = client.post("responses", json=body, headers=headers)

    response.raise_for_status()
    payload = response.json()
    if payload["status"] != "completed":
        raise RuntimeError(f"Response did not complete: {payload.get('error')}")

    response_id = str(payload["id"])
    text = "".join(
        part["text"]
        for item in payload["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )
    print(f"{response_id}\n{text}")
    return response_id


def main() -> None:
    """Send the five-turn branching example to a running Responses host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--azure", action="store_true", help="Authenticate to a Foundry-hosted agent."
    )
    args = parser.parse_args()

    token_provider: Callable[[], str] | None = None
    if args.azure:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://ai.azure.com/.default"
        )

    client = httpx.Client(
        base_url=args.base_url,
        params={"api-version": "v1"} if args.azure else None,
        timeout=120,
    )

    print("\nA: shared starting point")
    parent_id = _create_response(
        client,
        "Remember our shared trip constraints: one day in Seattle, no car, "
        "and vegetarian food. Acknowledge them in one sentence.",
        None,
        token_provider,
    )

    print("\nB: museum branch")
    museum_id = _create_response(
        client,
        "For this version of the trip, focus on museums. Suggest an itinerary.",
        parent_id,
        token_provider,
    )

    print("\nC: outdoor branch")
    outdoor_id = _create_response(
        client,
        "For this version of the trip, focus on parks and the waterfront. "
        "Suggest an itinerary.",
        parent_id,
        token_provider,
    )

    print("\nD: continue B")
    _create_response(
        client,
        "Summarize this version's activity preference, transport, and food.",
        museum_id,
        token_provider,
    )
    print("\nE: continue C")
    _create_response(
        client,
        "Summarize this version's activity preference, transport, and food.",
        outdoor_id,
        token_provider,
    )


if __name__ == "__main__":
    main()
