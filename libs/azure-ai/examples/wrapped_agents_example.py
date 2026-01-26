"""Example: Using Azure AI Foundry Wrapped Agents.

This example demonstrates how to use the Azure AI Foundry wrappers
to create and use IT, Enterprise, and DeepAgents.
"""

import os
from langchain_core.tools import tool

# Example tools for demonstration
@tool
def search_knowledge_base(query: str) -> str:
    """Search the IT knowledge base for solutions."""
    return f"Found solution for: {query} - Please try resetting your password via the self-service portal."


@tool
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """Create a support ticket."""
    return f"Ticket created: ID-12345 - {title} (Priority: {priority})"


@tool
def check_ticket_status(ticket_id: str) -> str:
    """Check the status of a support ticket."""
    return f"Ticket {ticket_id} status: In Progress - Assigned to IT Support Team"


def example_it_agent():
    """Example: IT Helpdesk Agent."""
    from langchain_azure_ai.wrappers import ITHelpdeskWrapper

    # Create IT Helpdesk agent with tools
    helpdesk = ITHelpdeskWrapper(
        name="it-helpdesk",
        tools=[search_knowledge_base, create_ticket, check_ticket_status],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    )

    # Simple chat
    print("=== IT Helpdesk Example ===")
    response = helpdesk.chat("I forgot my password and need to reset it")
    print(f"Response: {response}\n")

    # Chat with session continuity
    session_id = "session-123"
    result = helpdesk.chat_with_session(
        message="Can you create a ticket for me?",
        session_id=session_id,
        user_id="user@example.com",
    )
    print(f"Session result: {result}\n")

    return helpdesk


def example_enterprise_agent():
    """Example: Research Agent."""
    from langchain_azure_ai.wrappers import ResearchAgentWrapper

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        return f"Search results for '{query}': [Result 1, Result 2, Result 3]"

    # Create Research agent
    research = ResearchAgentWrapper(
        name="research-agent",
        tools=[web_search],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    )

    print("=== Research Agent Example ===")
    
    # Simple chat
    response = research.chat("What are the latest trends in AI?")
    print(f"Response: {response}\n")

    # Analysis with context
    result = research.analyze(
        query="Summarize key findings",
        context="AI is transforming various industries including healthcare, finance, and manufacturing.",
        output_format="markdown",
    )
    print(f"Analysis result: {result}\n")

    return research


def example_deep_agent():
    """Example: IT Operations DeepAgent."""
    from langchain_azure_ai.wrappers import ITOperationsWrapper, SubAgentConfig

    @tool
    def check_server_health(server_id: str) -> str:
        """Check health of a server."""
        return f"Server {server_id}: CPU 45%, Memory 60%, Disk 55% - Healthy"

    @tool
    def restart_service(service_name: str) -> str:
        """Restart a service."""
        return f"Service {service_name} restarted successfully"

    @tool
    def analyze_logs(log_source: str, time_range: str = "1h") -> str:
        """Analyze logs for issues."""
        return f"Log analysis for {log_source} ({time_range}): No critical issues found"

    # Define sub-agents for the DeepAgent
    sub_agents = [
        SubAgentConfig(
            name="monitor",
            instructions="You are a monitoring agent. Check server health and metrics.",
            tools=[check_server_health, analyze_logs],
        ),
        SubAgentConfig(
            name="remediate",
            instructions="You are a remediation agent. Fix issues by restarting services.",
            tools=[restart_service],
        ),
    ]

    # Create IT Operations DeepAgent
    it_ops = ITOperationsWrapper(
        name="it-operations",
        sub_agents=sub_agents,
        tools=[check_server_health, analyze_logs, restart_service],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    )

    print("=== IT Operations DeepAgent Example ===")
    
    # Execute a workflow
    result = it_ops.execute_workflow(
        task="Check the health of server-001 and fix any issues",
        thread_id="workflow-123",
    )
    print(f"Workflow result: {result}\n")

    return it_ops


def example_with_azure_foundry():
    """Example: Using with Azure AI Foundry (when configured)."""
    from langchain_azure_ai.wrappers import EnterpriseAgentWrapper, WrapperConfig

    # Configure to use Azure AI Foundry
    config = WrapperConfig(
        use_azure_foundry=True,  # Enable Foundry mode
        enable_tracing=True,
        # App Insights and other settings will be read from environment
    )

    # Create agent with Foundry configuration
    agent = EnterpriseAgentWrapper(
        name="foundry-agent",
        agent_subtype="research",
        config=config,
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    )

    print("=== Azure AI Foundry Agent Example ===")
    response = agent.chat("Hello, how can you help me?")
    print(f"Response: {response}\n")

    return agent


def example_run_server():
    """Example: Running the FastAPI server."""
    print("=== Server Example ===")
    print("To run the server, use one of these methods:")
    print()
    print("1. Using uvicorn:")
    print("   uvicorn langchain_azure_ai.server:app --host 0.0.0.0 --port 8000")
    print()
    print("2. Using Python:")
    print("   python -m langchain_azure_ai.server")
    print()
    print("3. Using the module directly:")
    print("   from langchain_azure_ai.server import create_app")
    print("   app = create_app()")
    print()
    print("Endpoints available:")
    print("  - GET  /health        - Health check")
    print("  - GET  /agents        - List available agents")
    print("  - GET  /chat          - Chat UI")
    print("  - POST /api/it/{name}/chat        - IT agent chat")
    print("  - POST /api/enterprise/{name}/chat - Enterprise agent chat")
    print("  - POST /api/deepagent/{name}/chat  - DeepAgent chat")
    print("  - POST /api/deepagent/{name}/execute - DeepAgent workflow")
    print()


if __name__ == "__main__":
    print("Azure AI Foundry Wrapped Agents Examples")
    print("=" * 50)
    print()

    # Check for required environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Warning: AZURE_OPENAI_ENDPOINT not set")
        print("Set environment variables before running examples:")
        print("  export AZURE_OPENAI_ENDPOINT=<your-endpoint>")
        print("  export AZURE_OPENAI_API_KEY=<your-key>")
        print("  export AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment>")
        print()
        print("Running in demo mode (no actual API calls)...")
        print()
        example_run_server()
    else:
        # Run examples
        try:
            example_it_agent()
        except Exception as e:
            print(f"IT Agent example error: {e}")

        try:
            example_enterprise_agent()
        except Exception as e:
            print(f"Enterprise Agent example error: {e}")

        try:
            example_deep_agent()
        except Exception as e:
            print(f"DeepAgent example error: {e}")

        example_run_server()
