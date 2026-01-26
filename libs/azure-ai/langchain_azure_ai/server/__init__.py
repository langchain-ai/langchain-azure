"""FastAPI server for Azure AI Foundry wrapped agents.

This module provides a FastAPI server that exposes wrapped LangChain agents
through REST endpoints and a chat UI, mirroring the langchain-agents patterns.

Endpoints:
- /chat - Interactive chat UI
- /chatui - Alternative chat UI endpoint
- /api/conversation/ - IT agent endpoints
- /api/enterprise/{agent_type}/ - Enterprise agent endpoints
- /api/deepagent/{agent_type}/ - DeepAgent endpoints
- /health - Health check endpoint
- /agents - List available agents
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to find .env file in parent directories
    env_path = Path(__file__).resolve().parent.parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment from: {env_path}")
    else:
        # Try current working directory
        load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed. Environment variables must be set manually.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoints."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    user_id: Optional[str] = Field(None, description="User ID")
    stream: bool = Field(False, description="Whether to stream the response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    agent_type: str = Field(..., description="Type of agent")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AgentInfo(BaseModel):
    """Information about an available agent."""

    name: str
    type: str
    subtype: str
    description: str
    endpoints: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    agents_loaded: int
    azure_foundry_enabled: bool


# Agent Registry
class AgentRegistry:
    """Registry for managing wrapped agents."""

    def __init__(self):
        self.it_agents: Dict[str, Any] = {}
        self.enterprise_agents: Dict[str, Any] = {}
        self.deep_agents: Dict[str, Any] = {}
        self._initialized = False

    def register_it_agent(self, name: str, agent: Any) -> None:
        """Register an IT agent."""
        self.it_agents[name] = agent
        logger.info(f"Registered IT agent: {name}")

    def register_enterprise_agent(self, name: str, agent: Any) -> None:
        """Register an enterprise agent."""
        self.enterprise_agents[name] = agent
        logger.info(f"Registered Enterprise agent: {name}")

    def register_deep_agent(self, name: str, agent: Any) -> None:
        """Register a DeepAgent."""
        self.deep_agents[name] = agent
        logger.info(f"Registered DeepAgent: {name}")

    def get_it_agent(self, name: str) -> Optional[Any]:
        """Get an IT agent by name."""
        return self.it_agents.get(name)

    def get_enterprise_agent(self, name: str) -> Optional[Any]:
        """Get an enterprise agent by name."""
        return self.enterprise_agents.get(name)

    def get_deep_agent(self, name: str) -> Optional[Any]:
        """Get a DeepAgent by name."""
        return self.deep_agents.get(name)

    def list_agents(self) -> List[AgentInfo]:
        """List all registered agents."""
        agents = []

        for name, agent in self.it_agents.items():
            agents.append(
                AgentInfo(
                    name=name,
                    type="IT",
                    subtype=getattr(agent, "agent_subtype", "unknown"),
                    description=getattr(agent, "description", ""),
                    endpoints=[
                        f"/api/conversation/{name}",
                        f"/api/it/{name}/chat",
                    ],
                )
            )

        for name, agent in self.enterprise_agents.items():
            agents.append(
                AgentInfo(
                    name=name,
                    type="Enterprise",
                    subtype=getattr(agent, "agent_subtype", "unknown"),
                    description=getattr(agent, "description", ""),
                    endpoints=[
                        f"/api/enterprise/{name}/chat",
                        f"/api/enterprise/{name}/analyze",
                    ],
                )
            )

        for name, agent in self.deep_agents.items():
            agents.append(
                AgentInfo(
                    name=name,
                    type="DeepAgent",
                    subtype=getattr(agent, "agent_subtype", "unknown"),
                    description=getattr(agent, "description", ""),
                    endpoints=[
                        f"/api/deepagent/{name}/execute",
                        f"/api/deepagent/{name}/chat",
                    ],
                )
            )

        return agents

    @property
    def total_agents(self) -> int:
        """Total number of registered agents."""
        return len(self.it_agents) + len(self.enterprise_agents) + len(self.deep_agents)


# Global registry
registry = AgentRegistry()


def load_agents():
    """Load and register all wrapped agents.

    This function should be customized to load your specific agents.
    It reads configuration from environment variables.
    Uses lazy initialization for faster startup.
    """
    from langchain_azure_ai.wrappers import (
        ITHelpdeskWrapper,
        ServiceNowWrapper,
        HITLSupportWrapper,
        ResearchAgentWrapper,
        ContentAgentWrapper,
        DataAnalystWrapper,
        DocumentAgentWrapper,
        CodeAssistantWrapper,
        RAGAgentWrapper,
        DocumentIntelligenceWrapper,
        ITOperationsWrapper,
        SalesIntelligenceWrapper,
        RecruitmentWrapper,
    )

    # Check if Azure AI Foundry is enabled
    use_foundry = os.getenv("USE_AZURE_FOUNDRY", "true").lower() == "true"
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    # For o4-mini, temperature must be 1.0
    temperature = 1.0 if "o4" in model.lower() else 0.0

    logger.info(f"Loading agents with Azure AI Foundry: {use_foundry}")
    logger.info(f"Using model: {model}, temperature: {temperature}")

    # Load all IT Agents
    try:
        registry.register_it_agent(
            "helpdesk",
            ITHelpdeskWrapper(name="it-helpdesk", model=model, temperature=temperature),
        )
        logger.info("IT helpdesk agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load IT helpdesk agent: {e}")

    try:
        registry.register_it_agent(
            "servicenow",
            ServiceNowWrapper(name="servicenow-agent", model=model, temperature=temperature),
        )
        logger.info("ServiceNow agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load ServiceNow agent: {e}")

    try:
        registry.register_it_agent(
            "hitl_support",
            HITLSupportWrapper(name="hitl-support", model=model, temperature=temperature),
        )
        logger.info("HITL Support agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load HITL Support agent: {e}")

    # Load all Enterprise Agents
    try:
        registry.register_enterprise_agent(
            "research",
            ResearchAgentWrapper(name="research-agent", model=model, temperature=temperature),
        )
        logger.info("Enterprise research agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Enterprise research agent: {e}")

    try:
        registry.register_enterprise_agent(
            "content",
            ContentAgentWrapper(name="content-agent", model=model, temperature=temperature),
        )
        logger.info("Content agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Content agent: {e}")

    try:
        registry.register_enterprise_agent(
            "data_analyst",
            DataAnalystWrapper(name="data-analyst-agent", model=model, temperature=temperature),
        )
        logger.info("Data Analyst agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Data Analyst agent: {e}")

    try:
        registry.register_enterprise_agent(
            "document",
            DocumentAgentWrapper(name="document-agent", model=model, temperature=temperature),
        )
        logger.info("Document agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Document agent: {e}")

    try:
        registry.register_enterprise_agent(
            "code_assistant",
            CodeAssistantWrapper(name="code-assistant-agent", model=model, temperature=temperature),
        )
        logger.info("Code Assistant agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Code Assistant agent: {e}")

    try:
        registry.register_enterprise_agent(
            "rag",
            RAGAgentWrapper(name="rag-agent", model=model, temperature=temperature),
        )
        logger.info("RAG agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load RAG agent: {e}")

    try:
        registry.register_enterprise_agent(
            "document_intelligence",
            DocumentIntelligenceWrapper(name="document-intelligence-agent", model=model, temperature=temperature),
        )
        logger.info("Document Intelligence agent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Document Intelligence agent: {e}")

    # Load all DeepAgents
    try:
        registry.register_deep_agent(
            "it_operations",
            ITOperationsWrapper(name="it-operations", model=model, temperature=temperature),
        )
        logger.info("IT Operations DeepAgent loaded")
    except Exception as e:
        logger.warning(f"Failed to load IT Operations DeepAgent: {e}")

    try:
        registry.register_deep_agent(
            "sales_intelligence",
            SalesIntelligenceWrapper(name="sales-intelligence", model=model, temperature=temperature),
        )
        logger.info("Sales Intelligence DeepAgent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Sales Intelligence DeepAgent: {e}")

    try:
        registry.register_deep_agent(
            "recruitment",
            RecruitmentWrapper(name="recruitment", model=model, temperature=temperature),
        )
        logger.info("Recruitment DeepAgent loaded")
    except Exception as e:
        logger.warning(f"Failed to load Recruitment DeepAgent: {e}")

    registry._initialized = True
    logger.info(f"Total agents loaded: {registry.total_agents}")


def load_additional_agents():
    """Load additional agents on demand.
    
    Call this to load all remaining agents after initial startup.
    """
    from langchain_azure_ai.wrappers import (
        ServiceNowWrapper,
        HITLSupportWrapper,
        ContentAgentWrapper,
        DataAnalystWrapper,
        DocumentAgentWrapper,
        CodeAssistantWrapper,
        RAGAgentWrapper,
        DocumentIntelligenceWrapper,
        SalesIntelligenceWrapper,
        RecruitmentWrapper,
    )

    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    temperature = 1.0 if "o4" in model.lower() else 0.0

    # Load remaining IT Agents
    if "servicenow" not in registry.it_agents:
        try:
            registry.register_it_agent(
                "servicenow",
                ServiceNowWrapper(name="servicenow-agent", model=model, temperature=temperature),
            )
            registry.register_it_agent(
                "hitl_support",
                HITLSupportWrapper(name="hitl-support", model=model, temperature=temperature),
            )
        except Exception as e:
            logger.warning(f"Failed to load additional IT agents: {e}")

    # Load remaining Enterprise Agents
    if "content" not in registry.enterprise_agents:
        try:
            registry.register_enterprise_agent(
                "content",
                ContentAgentWrapper(name="content-agent", model=model, temperature=temperature),
            )
            registry.register_enterprise_agent(
                "data_analyst",
                DataAnalystWrapper(name="data-analyst-agent", model=model, temperature=temperature),
            )
            registry.register_enterprise_agent(
                "document",
                DocumentAgentWrapper(name="document-agent", model=model, temperature=temperature),
            )
            registry.register_enterprise_agent(
                "code_assistant",
                CodeAssistantWrapper(name="code-assistant-agent", model=model, temperature=temperature),
            )
            registry.register_enterprise_agent(
                "rag",
                RAGAgentWrapper(name="rag-agent", model=model, temperature=temperature),
            )
            registry.register_enterprise_agent(
                "document_intelligence",
                DocumentIntelligenceWrapper(name="document-intelligence-agent", model=model, temperature=temperature),
            )
        except Exception as e:
            logger.warning(f"Failed to load additional Enterprise agents: {e}")

    # Load remaining DeepAgents
    if "sales_intelligence" not in registry.deep_agents:
        try:
            registry.register_deep_agent(
                "sales_intelligence",
                SalesIntelligenceWrapper(name="sales-intelligence", model=model, temperature=temperature),
            )
            registry.register_deep_agent(
                "recruitment",
                RecruitmentWrapper(name="recruitment", model=model, temperature=temperature),
            )
        except Exception as e:
            logger.warning(f"Failed to load additional DeepAgents: {e}")

    logger.info(f"Total agents after additional loading: {registry.total_agents}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Azure AI Foundry Agent Server...")
    load_agents()
    yield
    # Shutdown
    logger.info("Shutting down Azure AI Foundry Agent Server...")


# Create FastAPI app
app = FastAPI(
    title="Azure AI Foundry Agent Server",
    description="LangChain agents wrapped with Azure AI Foundry integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        agents_loaded=registry.total_agents,
        azure_foundry_enabled=os.getenv("USE_AZURE_FOUNDRY", "true").lower() == "true",
    )


# Load all agents endpoint
@app.post("/agents/load-all")
async def load_all_agents():
    """Load all additional agents on demand."""
    load_additional_agents()
    return {
        "status": "success",
        "agents_loaded": registry.total_agents,
        "agents": registry.list_agents(),
    }


# List agents endpoint
@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """List all available agents."""
    return registry.list_agents()


# Chat UI endpoint - serve static file if exists, otherwise inline
@app.get("/chat", response_class=HTMLResponse)
@app.get("/chatui", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat UI."""
    static_file = Path(__file__).parent / "static" / "chat.html"
    if static_file.exists():
        return FileResponse(static_file, media_type="text/html")
    
    # Fallback minimal HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Azure AI Foundry Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; border-radius: 8px; padding: 20px; height: 500px; overflow-y: auto; }
            .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
            .user { background-color: #e3f2fd; text-align: right; }
            .assistant { background-color: #f5f5f5; }
            .input-container { display: flex; margin-top: 20px; }
            #message-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            #send-btn { padding: 10px 20px; background-color: #0078d4; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
        </style>
    </head>
    <body>
        <h1>Azure AI Foundry Chat</h1>
        <p><a href="/static/chat.html">Click here for enhanced UI</a></p>
        <div class="chat-container" id="chat-container"></div>
        <div class="input-container">
            <input type="text" id="message-input" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()">
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
        <script>
            let sessionId = crypto.randomUUID();
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                const container = document.getElementById('chat-container');
                container.innerHTML += '<div class="message user">' + message + '</div>';
                input.value = '';
                try {
                    const response = await fetch('/api/it/helpdesk/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message, session_id: sessionId })
                    });
                    const data = await response.json();
                    container.innerHTML += '<div class="message assistant">' + data.response + '</div>';
                } catch (error) {
                    container.innerHTML += '<div class="message assistant" style="color: red;">Error: ' + error.message + '</div>';
                }
                container.scrollTop = container.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# IT Agent endpoints
@app.post("/api/conversation/{agent_name}", response_model=ChatResponse)
@app.post("/api/it/{agent_name}/chat", response_model=ChatResponse)
async def it_agent_chat(agent_name: str, request: ChatRequest):
    """Chat with an IT agent."""
    agent = registry.get_it_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"IT agent '{agent_name}' not found")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = agent.chat(request.message, thread_id=session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_type=f"IT/{getattr(agent, 'agent_subtype', 'unknown')}",
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Error in IT agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Enterprise Agent endpoints
@app.post("/api/enterprise/{agent_name}/chat", response_model=ChatResponse)
async def enterprise_agent_chat(agent_name: str, request: ChatRequest):
    """Chat with an enterprise agent."""
    agent = registry.get_enterprise_agent(agent_name)
    if not agent:
        raise HTTPException(
            status_code=404, detail=f"Enterprise agent '{agent_name}' not found"
        )

    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = agent.chat(request.message, thread_id=session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_type=f"Enterprise/{getattr(agent, 'agent_subtype', 'unknown')}",
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Error in Enterprise agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoints."""

    query: str = Field(..., description="Analysis query")
    context: Optional[str] = Field(None, description="Context for analysis")
    output_format: str = Field("text", description="Output format")
    session_id: Optional[str] = Field(None, description="Session ID")


@app.post("/api/enterprise/{agent_name}/analyze")
async def enterprise_agent_analyze(agent_name: str, request: AnalyzeRequest):
    """Perform analysis with an enterprise agent."""
    agent = registry.get_enterprise_agent(agent_name)
    if not agent:
        raise HTTPException(
            status_code=404, detail=f"Enterprise agent '{agent_name}' not found"
        )

    try:
        result = agent.analyze(
            query=request.query,
            context=request.context,
            output_format=request.output_format,
            thread_id=request.session_id,
        )
        return result
    except Exception as e:
        logger.error(f"Error in Enterprise agent analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# DeepAgent endpoints
@app.post("/api/deepagent/{agent_name}/chat", response_model=ChatResponse)
async def deep_agent_chat(agent_name: str, request: ChatRequest):
    """Chat with a DeepAgent."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"DeepAgent '{agent_name}' not found")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = agent.chat(request.message, thread_id=session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_type=f"DeepAgent/{getattr(agent, 'agent_subtype', 'unknown')}",
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Error in DeepAgent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""

    task: str = Field(..., description="Task to execute")
    thread_id: Optional[str] = Field(None, description="Thread ID")
    max_iterations: int = Field(10, description="Maximum iterations")


@app.post("/api/deepagent/{agent_name}/execute")
async def deep_agent_execute(agent_name: str, request: WorkflowRequest):
    """Execute a workflow with a DeepAgent."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"DeepAgent '{agent_name}' not found")

    try:
        result = agent.execute_workflow(
            task=request.task,
            thread_id=request.thread_id,
            max_iterations=request.max_iterations,
        )
        return result
    except Exception as e:
        logger.error(f"Error in DeepAgent workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# SSE Streaming endpoint for DeepAgents
@app.post("/api/deepagent/{agent_name}/chat/stream")
async def deep_agent_chat_stream(agent_name: str, request: ChatRequest):
    """Stream chat with a DeepAgent using Server-Sent Events.

    Uses SSE to stream:
    - start: Session started
    - thinking: Agent's reasoning steps (collapsible in UI)
    - tool_start: When a tool is being called
    - tool_result: Tool execution results
    - todo_update: Todo list changes
    - token: Streaming response tokens
    - complete: Final response with all context

    Returns:
        SSE stream of events.
    """
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': f'DeepAgent {agent_name} not found'}})}\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
        )

    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from agent chat."""
        try:
            # Start event
            yield f"data: {json.dumps({'type': 'start', 'data': {'message': 'Processing your request...', 'session_id': session_id}})}\n\n"
            
            iteration = 0
            response_text = ""
            
            # Check if agent has streaming capability
            if hasattr(agent, 'astream_chat'):
                # Use native streaming if available
                async for event in agent.astream_chat(
                    message=request.message,
                    session_id=session_id,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                # Simulate streaming for non-streaming agents
                # Send thinking event
                iteration += 1
                yield f"data: {json.dumps({'type': 'thinking', 'data': {'iteration': iteration, 'phase': 'planning', 'summary': 'Analyzing request...', 'content': 'Processing your query and formulating response.'}})}\n\n"
                
                await asyncio.sleep(0.1)
                
                # Get response from agent without blocking the event loop
                response_text = await asyncio.to_thread(
                    agent.chat,
                    request.message,
                    thread_id=session_id,
                )
                
                # Send complete event
                yield f"data: {json.dumps({'type': 'complete', 'data': {'response': response_text, 'session_id': session_id}})}\n\n"
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Session management endpoints for DeepAgents
@app.get("/api/deepagent/{agent_name}/todos/{session_id}")
async def deep_agent_todos(agent_name: str, session_id: str):
    """Get the todo list for a DeepAgent session."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        return {"todos": []}
    
    # Try to get todos from agent if it supports it
    if hasattr(agent, 'get_todos'):
        return {"todos": agent.get_todos(session_id)}
    
    return {"todos": []}


@app.get("/api/deepagent/{agent_name}/files/{session_id}")
async def deep_agent_files(agent_name: str, session_id: str):
    """Get the workspace files for a DeepAgent session."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        return {"files": []}
    
    # Try to get files from agent if it supports it
    if hasattr(agent, 'get_files'):
        return {"files": agent.get_files(session_id)}
    
    return {"files": []}


@app.get("/api/deepagent/{agent_name}/subagents")
async def deep_agent_subagents(agent_name: str):
    """Get available subagents for a DeepAgent."""
    subagent_configs = {
        'it_operations': [
            {'name': 'Incident', 'description': 'Incident management and resolution'},
            {'name': 'Change', 'description': 'Change request validation and risk'},
            {'name': 'Problem', 'description': 'Root cause analysis and problem management'},
            {'name': 'Asset', 'description': 'CMDB and asset management'},
            {'name': 'SLA', 'description': 'SLA monitoring and compliance'},
            {'name': 'Knowledge', 'description': 'Knowledge base management'},
        ],
        'sales_intelligence': [
            {'name': 'Deal Qualifier', 'description': 'BANT/MEDDIC qualification'},
            {'name': 'Solution Architect', 'description': 'Solution mapping and design'},
            {'name': 'Proposal Writer', 'description': 'RFP/Proposal generation'},
            {'name': 'Pricing Analyst', 'description': 'Pricing optimization'},
            {'name': 'Competitive Strategist', 'description': 'Competitive analysis'},
        ],
        'recruitment': [
            {'name': 'Document Manager', 'description': 'Document handling and storage'},
            {'name': 'Resume Screener', 'description': 'Resume parsing and screening'},
            {'name': 'Question Generator', 'description': 'Interview question generation'},
            {'name': 'Answer Evaluator', 'description': 'Candidate evaluation'},
            {'name': 'Report Generator', 'description': 'Report and analytics generation'},
        ],
    }
    
    return {"subagents": subagent_configs.get(agent_name, [])}


# File upload endpoints
@app.post("/api/enterprise/{agent_name}/upload")
async def enterprise_agent_upload(
    agent_name: str,
    file: UploadFile = File(...),
    session_id: str = Form(None),
):
    """Upload a file for an enterprise agent to process."""
    agent = registry.get_enterprise_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Enterprise agent '{agent_name}' not found")

    session_id = session_id or str(uuid.uuid4())

    try:
        # Read file content
        content = await file.read()
        filename = file.filename
        
        # Check if agent has upload capability
        if hasattr(agent, 'upload_file'):
            result = agent.upload_file(content, filename, session_id)
            return {
                "success": True,
                "filename": filename,
                "session_id": session_id,
                "message": result.get("message", f"File {filename} uploaded successfully"),
            }
        else:
            return {
                "success": True,
                "filename": filename,
                "session_id": session_id,
                "message": f"File {filename} received (agent does not support file processing)",
            }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/deepagent/{agent_name}/upload")
async def deep_agent_upload(
    agent_name: str,
    file: UploadFile = File(...),
    session_id: str = Form(None),
):
    """Upload a file for a DeepAgent to process."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"DeepAgent '{agent_name}' not found")

    session_id = session_id or str(uuid.uuid4())

    try:
        # Read file content
        content = await file.read()
        filename = file.filename
        
        # Check if agent has upload capability
        if hasattr(agent, 'upload_file'):
            result = agent.upload_file(content, filename, session_id)
            return {
                "success": True,
                "filename": filename,
                "session_id": session_id,
                "message": result.get("message", f"File {filename} uploaded successfully"),
            }
        else:
            return {
                "success": True,
                "filename": filename,
                "session_id": session_id,
                "message": f"File {filename} received (agent does not support file processing)",
            }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    return app


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
