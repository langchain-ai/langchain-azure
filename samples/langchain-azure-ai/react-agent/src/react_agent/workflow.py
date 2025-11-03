"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from react_agent.prompts import SYSTEM_PROMPT
from langchain_azure_ai.chat_models import AzureChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_azure_ai.agents import AgentServiceFactory
from langgraph_supervisor import create_supervisor
from azure.identity import DefaultAzureCredential


service = AgentServiceFactory()
model = ChatOpenAI(model="gpt-4.1")

# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = service.create_prompt_agent(
    name="math_expert",
    model="gpt-4.1",
    tools=[add, multiply],
    instructions="You are a math expert. Always use one tool at a time."
)

research_agent = service.create_prompt_agent(
    name="research_expert",
    model="gpt-4.1",
    tools=[web_search],
    instructions="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)