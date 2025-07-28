"""Test script for Azure AI Agents integration with code interpreter example."""

# Setup for local development - add the local package to Python path
import os
import sys
from pathlib import Path

# Get the path to the langchain_azure_ai package in the current repo
current_dir = Path.cwd()
# Navigate to the libs/azure-ai directory regardless of where we are
azure_ai_path = None

# Try multiple possible locations
possible_paths = [
    current_dir / "langchain_azure_ai",  # If we're in the azure-ai directory
    current_dir.parent / "langchain_azure_ai",  # If we're in docs subdirectory
    current_dir / "libs" / "azure-ai" / "langchain_azure_ai",  # If we're in repo root
    current_dir.parent.parent / "langchain_azure_ai",  # If we're nested deeper
]

for path in possible_paths:
    if path.exists() and (path / "__init__.py").exists():
        azure_ai_path = path.parent
        break

if azure_ai_path:
    # Add to Python path if not already there
    azure_ai_str = str(azure_ai_path)
    if azure_ai_str not in sys.path:
        sys.path.insert(0, azure_ai_str)
    print(f"✓ Added local package path: {azure_ai_path}")

    # Verify the package is accessible
    try:
        import langchain_azure_ai

        print(
            "✓ Successfully imported langchain_azure_ai from: "
            f"{langchain_azure_ai.__file__}"
        )

        # Check if Azure AI Agents is available
        try:
            from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService

            print("✓ Azure AI Agents integration is available in the local build!")
        except ImportError as e:
            print(f"✗ Azure AI Agents not found: {e}")

    except ImportError as e:
        print(f"✗ Could not import langchain_azure_ai: {e}")
else:
    print("✗ Could not find langchain_azure_ai package in expected locations")
    print(
        "   Make sure you're running this notebook from within the "
        "langchain-azure repository"
    )
    print("   Current directory:", current_dir)


# Try to import Azure and LangChain modules

from azure.identity import DefaultAzureCredential

# Import other required modules
import langchain_azure_ai
from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService

# Set up environment variables for Azure AI Projects
# Replace with your actual connection string
os.environ["PROJECT_CONNECTION_STRING"] = (
    "/subscriptions/2375c423-6855-448c-bc16-d1326ab8ca77/resourceGroups/rg-mmhangami-0374/providers/Microsoft.CognitiveServices/accounts/mmhangami-0374-resource"
)
os.environ["PROJECT_ENDPOINT"] = (
    "https://mmhangami-0374-resource.services.ai.azure.com/api/projects/mmhangami-0374"
)
os.environ["MODEL_DEPLOYMENT_NAME"] = "gpt-4.1"


# First, let's create some sample data for the code interpreter to work with
import os
from pathlib import Path

import pandas as pd

# Create sample sales data
data = {
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "sales": [12000, 15000, 18000, 14000, 22000, 25000],
    "region": ["North", "South", "East", "West", "North", "South"],
}

df = pd.DataFrame(data)

# Create a CSV file in the current working directory
csv_path = Path.cwd() / "sample_sales_data.csv"

df.to_csv(
    csv_path, index=False, encoding="utf-8-sig"
)  # The code interpreter requires utf-8-sig

print(f"Created sample data file → {csv_path}")
print("\nSample data:")
print(df)

# Import required modules for tools using Azure AI Projects SDK
from azure.ai.agents.models import CodeInterpreterTool, FilePurpose, MessageRole

# from azure.ai.projects import AIProjectClient

try:
    agents_client = AzureAIAgentsService(
        credential=DefaultAzureCredential(),
        endpoint=os.environ["PROJECT_ENDPOINT"],  # Use the project endpoint
        # model_name="gpt-4.1",  # Use a model that's available in your project
    )._create_client()

    # # Upload the file for the code interpreter
    print("Uploading file for code interpreter...")
    uploaded_file = agents_client.files.upload_and_poll(
        file_path=str(csv_path),
        purpose=FilePurpose.AGENTS,  # Specify the purpose for the file
    )

    print(f"✓ Uploaded file, file ID: {uploaded_file.id}")

    # Create code interpreter tool with the uploaded file
    code_interpreter = CodeInterpreterTool(file_ids=[uploaded_file.id])

    print(code_interpreter.definitions)

    print(code_interpreter.resources)

    agent = agents_client.create_agent(
        model="gpt-4.1",
        name="code-interpreter-agent",
        instructions="""You are a data analyst agent. 
                Analyze the provided data and create visualizations 
                when helpful. Use Python code to explore and understand the data.""",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )
    print(f"Created agent, agent ID: {agent.id}")

    thread = agents_client.threads.create()
    print(f"Created thread, thread ID: {thread.id}")

    print("✓ Created AzureAIAgentsService with code interpreter tool")

    # Create a message
    message = agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=(
            "create a pie chart with the data showing sales by region and "
            "show it to me as a png image."
        ),
    )
    print(f"Created message, message ID: {message.id}")

    run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        # Check if you got "Rate limit is exceeded.", then you want to get more quota
        print(f"Run failed: {run.last_error}")

    # [START get_messages_and_save_files]
    messages = agents_client.messages.list(thread_id=thread.id)

    for msg in messages:
        # Save every image file in the message
        for img in msg.image_contents:
            file_id = img.image_file.file_id
            file_name = f"{file_id[:5]}_image_file.png"
            agents_client.files.save(
                file_id=file_id,
                file_name=file_name,
                target_dir="/workspaces/langchain-azure/libs/azure-ai/docs",
            )
            print(f"Saved image file to: {Path.cwd() / file_name}")

        # Print details of every file-path annotation
        for ann in msg.file_path_annotations:
            print("File Paths:")
            print(f"  Type: {ann.type}")
            print(f"  Text: {ann.text}")
            print(f"  File ID: {ann.file_path.file_id}")
            print(f"  Start Index: {ann.start_index}")
            print(f"  End Index: {ann.end_index}")
    # [END get_messages_and_save_files]

    last_msg = agents_client.messages.get_last_message_text_by_role(
        thread_id=thread.id, role=MessageRole.AGENT
    )
    if last_msg:
        print(f"Last Message: {last_msg.text.value}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your Azure AI Projects configuration and try again.")
    import traceback

    traceback.print_exc()
