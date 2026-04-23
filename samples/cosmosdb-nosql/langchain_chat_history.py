"""Azure CosmosDB Chat Message History sample.

Demonstrates storing and retrieving chat message history in CosmosDB,
including multi-session support and TTL-based expiration.

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env  # fill in your values

Environment variables:
    COSMOSDB_ENDPOINT - CosmosDB account endpoint
    COSMOSDB_KEY      - CosmosDB account key
"""

import os

from azure.cosmos import CosmosClient
from dotenv import load_dotenv

from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

load_dotenv()

DATABASE_NAME = "sample-chathistory-db"


def main() -> None:
    """Run chat history sample."""
    client = CosmosClient(
        os.environ["COSMOSDB_ENDPOINT"],
        os.environ["COSMOSDB_KEY"],
    )
    try:
        # --- Session 1: Add messages ---
        print("=== Session 1: Adding messages ===")
        history = CosmosDBChatMessageHistory(
            cosmos_endpoint=os.environ["COSMOSDB_ENDPOINT"],
            credential=os.environ["COSMOSDB_KEY"],
            cosmos_database=DATABASE_NAME,
            cosmos_container="sample-chathistory-container",
            session_id="session-001",
            user_id="user-alice",
        )
        history.prepare_cosmos()

        history.add_user_message("Hi! I'm planning a trip to Tokyo.")
        history.add_ai_message(
            "That sounds exciting! Tokyo has amazing food, temples, and technology. "
            "When are you planning to go?"
        )
        history.add_user_message("Sometime in April for cherry blossom season.")
        history.add_ai_message(
            "Great choice! Late March to mid-April is peak cherry blossom season. "
            "I recommend visiting Ueno Park and Shinjuku Gyoen."
        )

        print(f"  Stored {len(history.messages)} messages")
        for msg in history.messages:
            content = msg.content
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"  [{msg.type}] {content}")
        print()

        # --- Session 2: Different session, same user ---
        print("=== Session 2: New session for same user ===")
        history2 = CosmosDBChatMessageHistory(
            cosmos_endpoint=os.environ["COSMOSDB_ENDPOINT"],
            credential=os.environ["COSMOSDB_KEY"],
            cosmos_database=DATABASE_NAME,
            cosmos_container="sample-chathistory-container",
            session_id="session-002",
            user_id="user-alice",
        )
        history2.prepare_cosmos()

        history2.add_user_message("What about restaurants in Tokyo?")
        history2.add_ai_message(
            "Tokyo has more Michelin-starred restaurants than any other city! "
            "Try Tsukiji Outer Market for fresh sushi."
        )

        print(f"  Session 2 has {len(history2.messages)} messages (independent)")
        print(f"  Session 1 still has {len(history.messages)} messages")
        print()

        # --- Retrieve messages from Session 1 ---
        print("=== Retrieving Session 1 messages ===")
        for msg in history.messages:
            print(f"  [{msg.type}] {msg.content}")
        print()

        # --- Clear Session 1 ---
        print("=== Clearing Session 1 ===")
        history.clear()
        print(f"  Session 1 messages after clear: {len(history.messages)}")
        print(f"  Session 2 messages (unaffected): {len(history2.messages)}")
        print()

        # --- TTL example ---
        print("=== TTL Example ===")
        history_ttl = CosmosDBChatMessageHistory(
            cosmos_endpoint=os.environ["COSMOSDB_ENDPOINT"],
            credential=os.environ["COSMOSDB_KEY"],
            cosmos_database=DATABASE_NAME,
            cosmos_container="sample-chathistory-container",
            session_id="session-ttl",
            user_id="user-bob",
            ttl=3600,  # messages expire after 1 hour
        )
        history_ttl.prepare_cosmos()
        history_ttl.add_user_message("This message will expire in 1 hour.")
        print("  Added message with TTL=3600s")
        print()
    finally:
        # --- Cleanup ---
        print("Cleaning up...")
        try:
            client.delete_database(DATABASE_NAME)
            print("Done! Database deleted.")
        except Exception:
            print("Database may not have been created; skipping cleanup.")


if __name__ == "__main__":
    main()
