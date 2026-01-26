"""Start the FastAPI server for Azure AI Foundry Agents."""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

# Try multiple locations for .env file
env_paths = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
        break

# Start server
import uvicorn
from langchain_azure_ai.server import app

if __name__ == "__main__":
    print("Starting Azure AI Foundry Agents Server...")
    print("Server will be available at http://localhost:8000")
    print("UI available at http://localhost:8000/chat")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000)
