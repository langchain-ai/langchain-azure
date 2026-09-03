"""Serve the mortgage processing sample over HTTP."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Awaitable, Callable

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

STATIC_DIR = Path(__file__).with_name("static")
RUN_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
ALLOWED_WEBSOCKET_ORIGINS = {
    "http://127.0.0.1:8001",
    "http://localhost:8001",
}
logger = logging.getLogger(__name__)


class MortgageStreamAdapter:
    """Translate Deep Agents graph updates into browser-safe workflow events."""

    _FILE_TOOLS = {
        "ls",
        "glob",
        "grep",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
    }

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.message_agents: dict[str, str] = {}
        self.pending_calls: dict[str, dict[str, Any]] = {}
        self.emitted_calls: set[str] = set()
        self.completed_agents: set[str] = set()

    async def emit(self, event_type: str, **payload: Any) -> None:
        await self.websocket.send_json(
            {
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )

    async def observe(
        self, namespace: tuple[str, ...], stream_mode: str, data: Any
    ) -> None:
        if stream_mode == "messages":
            message, metadata = data
            if isinstance(message, (AIMessage, AIMessageChunk)) and message.id:
                self.message_agents[message.id] = str(
                    metadata.get("lc_agent_name") or "orchestrator"
                )
            return
        if stream_mode != "updates" or not isinstance(data, dict):
            return

        for update in data.values():
            if not isinstance(update, dict):
                continue
            messages = update.get("messages", [])
            if not isinstance(messages, list):
                messages = [messages]
            for message in messages:
                if isinstance(message, AIMessage):
                    await self._handle_tool_calls(message)
                elif isinstance(message, ToolMessage):
                    await self._handle_tool_result(message)

    async def _handle_tool_calls(self, message: AIMessage) -> None:
        agent_name = self.message_agents.get(str(message.id), "orchestrator")
        for call in message.tool_calls or []:
            call_id = str(call.get("id") or "")
            if not call_id or call_id in self.emitted_calls:
                continue
            self.emitted_calls.add(call_id)
            args = call.get("args") or {}
            tool_name = str(call.get("name") or "tool")
            if tool_name == "task":
                target = str(args.get("subagent_type") or "specialist")
                description = str(args.get("description") or "")
                self.pending_calls[call_id] = {
                    "kind": "delegation",
                    "targetAgent": target,
                }
                await self.emit(
                    "delegation.started",
                    agent=agent_name,
                    targetAgent=target,
                    summary=description.splitlines()[0] if description else target,
                )
                continue
            if tool_name not in self._FILE_TOOLS:
                continue
            path = args.get("file_path") or args.get("path") or args.get("pattern")
            if path is None:
                continue
            direction = (
                "write"
                if tool_name in {"write_file", "edit_file", "delete"}
                else "read"
            )
            self.pending_calls[call_id] = {
                "kind": "filesystem",
                "agent": agent_name,
                "tool": tool_name,
                "path": str(path),
                "direction": direction,
            }
            await self.emit(
                "filesystem.started",
                agent=agent_name,
                tool=tool_name,
                path=str(path),
                direction=direction,
            )

    async def _handle_tool_result(self, message: ToolMessage) -> None:
        pending = self.pending_calls.pop(str(message.tool_call_id or ""), None)
        if not pending or pending["kind"] != "delegation":
            return
        status = message.status or "success"
        target_agent = pending["targetAgent"]
        if status != "success":
            await self.emit(
                "delegation.failed",
                agent=target_agent,
                error=str(message.content),
            )
            return
        self.completed_agents.add(target_agent)
        await self.emit(
            "handoff.completed",
            agent=target_agent,
            handoff=len(self.completed_agents),
        )


def create_app(
    *,
    packet_id: str,
    model_name: str,
    account_url: str | None,
    source_container: str,
    source_prefix: str,
    guidance_container: str,
    output_container: str,
    output_prefix: str,
    source_backend: Any,
    output_backend: Any,
    process_prompt: Callable[..., Awaitable[Any]],
) -> FastAPI:
    """Create the HTTP adapter from initialized agent resources."""
    app = FastAPI(title="Mortgage Packet Processing")
    app.state.packet_id = packet_id
    app.state.model_name = model_name
    app.state.account_url = account_url
    app.state.source_container = source_container
    app.state.source_prefix = source_prefix
    app.state.guidance_container = guidance_container
    app.state.output_container = output_container
    app.state.output_prefix = output_prefix
    app.state.source_backend = source_backend
    app.state.output_backend = output_backend
    app.state.process_prompt = process_prompt
    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/shared/styles.css", shared_styles, methods=["GET"])
    app.add_api_route("/api/config", config, methods=["GET"])
    app.add_api_route("/api/runs", run_mortgage, methods=["POST"])
    app.add_api_websocket_route("/ws/runs", stream_mortgage_run)
    app.add_api_route(
        "/api/source/{file_path:path}", source_preview, methods=["GET"]
    )
    app.add_api_route(
        "/api/output/{run_id}/{file_path:path}", output_preview, methods=["GET"]
    )
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    return app


def _source_backend(request: Request) -> Any:
    return request.app.state.source_backend


async def _process_prompt(
    app: FastAPI,
    prompt: str,
    observe_stream: Any = None,
) -> Any:
    return await app.state.process_prompt(
        prompt,
        observe_stream=observe_stream,
    )


def _validated_blob_path(file_path: str) -> PurePosixPath:
    path = PurePosixPath("/" + file_path.strip("/"))
    if not file_path or any(part in {".", ".."} for part in path.parts):
        raise HTTPException(status_code=400, detail="Invalid Blob path")
    return path


async def _read_blob_text(backend: Any, path: PurePosixPath) -> str:
    result = await backend.aread(str(path), limit=300)
    if result.error is not None or result.file_data is None:
        raise HTTPException(status_code=404, detail=result.error or "Blob not found")
    content = result.file_data["content"]
    if result.file_data.get("encoding", "utf-8") != "utf-8":
        raise HTTPException(status_code=415, detail="Blob preview is not UTF-8 text")
    return content


async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def shared_styles() -> FileResponse:
    return FileResponse(STATIC_DIR / "base.css", media_type="text/css")


async def config(request: Request) -> dict[str, Any]:
    state = request.app.state
    listing = await _source_backend(request).als("/")
    if listing.error is not None:
        raise HTTPException(status_code=503, detail=listing.error)
    files = [
        {
            **entry,
            "virtualPath": f"/source{entry['path']}",
            "blobName": f"{state.source_prefix}{entry['path'].lstrip('/')}",
        }
        for entry in listing.entries or []
        if not entry.get("is_dir")
    ]
    return {
        "status": "ready",
        "packetId": state.packet_id,
        "model": state.model_name,
        "account": state.account_url,
        "source": f"{state.source_container}/{state.source_prefix}",
        "guidance": state.guidance_container,
        "outputs": f"{state.output_container}/{state.output_prefix}",
        "files": files,
    }


async def run_mortgage(
    request: Request, prompt: Annotated[str, Body(embed=True)]
) -> dict[str, Any]:
    try:
        result = await _process_prompt(request.app, prompt)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return asdict(result)


async def stream_mortgage_run(websocket: WebSocket) -> None:
    if websocket.headers.get("origin") not in ALLOWED_WEBSOCKET_ORIGINS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    adapter = MortgageStreamAdapter(websocket)
    try:
        body = await websocket.receive_json()
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Mortgage processing request cannot be empty")
        await adapter.emit("run.started")
        result = await _process_prompt(
            websocket.app,
            prompt,
            adapter.observe,
        )
        for artifact in result.artifacts:
            await adapter.emit("artifact.verified", artifact=asdict(artifact))
        await adapter.emit("run.completed", result=asdict(result))
    except WebSocketDisconnect:
        return
    except (RuntimeError, ValueError) as exc:
        await adapter.emit("run.failed", error=str(exc))
    except Exception:  # noqa: BLE001 - keep unexpected details in server logs
        logger.exception("Unexpected mortgage processing failure")
        await adapter.emit(
            "run.failed", error="Unexpected mortgage processing failure"
        )
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


async def source_preview(request: Request, file_path: str) -> dict[str, str]:
    path = _validated_blob_path(file_path)
    return {
        "path": f"/source{path}",
        "content": await _read_blob_text(_source_backend(request), path),
    }


async def output_preview(
    request: Request, run_id: str, file_path: str
) -> dict[str, str]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID")
    path = _validated_blob_path(f"{run_id}/{file_path}")
    return {
        "path": f"/output/{file_path.strip('/')}",
        "content": await _read_blob_text(
            request.app.state.output_backend, path
        ),
    }


async def run_server(
    app: FastAPI,
) -> None:
    """Run the optional browser server."""
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=8001)
    await uvicorn.Server(config).serve()