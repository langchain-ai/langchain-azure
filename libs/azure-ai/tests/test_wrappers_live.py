"""Test script to verify Azure AI Foundry wrapped agents work correctly.

This script tests the wrapper classes and server functionality.
"""

import os
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)
print(f"Loaded env from: {env_path}")

def test_config():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    from langchain_azure_ai.config import get_config
    
    config = get_config()
    print(f"Azure OpenAI Endpoint: {config.azure_openai.endpoint}")
    print(f"Deployment Name: {config.azure_openai.deployment_name}")
    print(f"Azure AI Foundry Enabled: {config.azure_foundry.is_enabled}")
    print(f"Project Endpoint: {config.azure_foundry.project_endpoint}")
    return config


def test_it_agent():
    """Test IT Agent wrapper."""
    print("\n=== Testing IT Agent ===")
    from langchain_azure_ai.wrappers import ITHelpdeskWrapper
    
    # Create agent with o4-mini (requires temperature=1)
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
    
    # For o4-mini, temperature must be 1.0
    agent = ITHelpdeskWrapper(
        name="test-helpdesk",
        model=model,
        temperature=1.0 if "o4" in model.lower() else 0.0,
    )
    
    print(f"Created agent: {agent.name}")
    print(f"Agent type: {agent.agent_type}")
    print(f"Subtype: {agent.agent_subtype}")
    
    # Test chat
    try:
        response = agent.chat("Hello, I need help with my computer")
        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        return True
    except Exception as e:
        print(f"Chat error: {e}")
        return False


def test_enterprise_agent():
    """Test Enterprise Agent wrapper."""
    print("\n=== Testing Enterprise Agent ===")
    from langchain_azure_ai.wrappers import ResearchAgentWrapper
    
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
    
    agent = ResearchAgentWrapper(
        name="test-research",
        model=model,
        temperature=1.0 if "o4" in model.lower() else 0.0,
    )
    
    print(f"Created agent: {agent.name}")
    print(f"Agent type: {agent.agent_type}")
    
    try:
        response = agent.chat("What are the benefits of cloud computing?")
        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        return True
    except Exception as e:
        print(f"Chat error: {e}")
        return False


def test_deep_agent():
    """Test DeepAgent wrapper."""
    print("\n=== Testing DeepAgent ===")
    from langchain_azure_ai.wrappers import ITOperationsWrapper
    
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
    
    agent = ITOperationsWrapper(
        name="test-it-ops",
        model=model,
        temperature=1.0 if "o4" in model.lower() else 0.0,
    )
    
    print(f"Created agent: {agent.name}")
    print(f"Agent type: {agent.agent_type}")
    
    try:
        response = agent.chat("Check the server status")
        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        return True
    except Exception as e:
        print(f"Chat error: {e}")
        return False


def test_server_startup():
    """Test server can start without errors."""
    print("\n=== Testing Server Startup ===")
    try:
        from langchain_azure_ai.server import app, registry, load_agents
        
        # Load agents
        load_agents()
        
        print(f"Server app created: {app.title}")
        print(f"Total agents registered: {registry.total_agents}")
        print(f"IT Agents: {list(registry.it_agents.keys())}")
        print(f"Enterprise Agents: {list(registry.enterprise_agents.keys())}")
        print(f"DeepAgents: {list(registry.deep_agents.keys())}")
        return True
    except Exception as e:
        print(f"Server startup error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Azure AI Foundry Wrapped Agents - Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test config
    try:
        test_config()
        results["config"] = True
    except Exception as e:
        print(f"Config test failed: {e}")
        results["config"] = False
    
    # Test IT Agent
    results["it_agent"] = test_it_agent()
    
    # Test Enterprise Agent
    results["enterprise_agent"] = test_enterprise_agent()
    
    # Test DeepAgent
    results["deep_agent"] = test_deep_agent()
    
    # Test server
    results["server"] = test_server_startup()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    return all_passed


if __name__ == "__main__":
    run_all_tests()
