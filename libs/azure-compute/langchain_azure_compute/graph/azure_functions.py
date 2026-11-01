# Copyright (c) Microsoft. All rights reserved.

"""First-class Azure Functions Python v2 HTTP route integration for.

`langgraph-durabletask`.

Exposes standard HTTP endpoints for starting durable LangGraph workflows, polling status
and
pending human-in-the-loop interrupts, and submitting human responses to resume paused
runs:

  - `POST /api/graphs/{name}/start` (and `POST /api/workflow/{name}/run`)
  - `GET  /api/graphs/{name}/status/{instance_id}` (and `GET
    /api/workflow/{name}/status/{instance_id}`)
  - `POST /api/graphs/{name}/respond/{instance_id}`
  - `POST /api/graphs/{name}/respond/{instance_id}/{interrupt_id}`
  - `GET  /api/health`
"""

from __future__ import annotations

import atexit
import json
import threading
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

if TYPE_CHECKING:
    from langchain_azure_compute.graph import ClientProtocol, Registration

_MISSING = object()


def _require_azure_functions() -> Any:
    try:
        import azure.functions as func

        return func
    except ImportError as exc:
        raise ImportError(
            "azure-functions is required for Azure Functions HTTP route integration; "
            "install with: pip install 'langgraph-durabletask[azure-functions]'"
        ) from exc


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "type") and hasattr(obj, "content"):
        data: dict[str, Any] = {
            "type": getattr(obj, "type"),
            "content": getattr(obj, "content"),
        }
        if getattr(obj, "tool_calls", None):
            data["tool_calls"] = getattr(obj, "tool_calls")
        if getattr(obj, "tool_call_id", None):
            data["tool_call_id"] = getattr(obj, "tool_call_id")
        return data
    return str(obj)


def _json_response(payload: Any, *, status_code: int = 200) -> Any:
    func = _require_azure_functions()
    return func.HttpResponse(
        body=json.dumps(payload, default=_json_default),
        status_code=status_code,
        mimetype="application/json",
    )


def _extract_api_base(url: str) -> str:
    clean = url.split("?", 1)[0].rstrip("/")
    for marker in ("/graphs/", "/workflow/"):
        if marker in clean:
            return clean.split(marker, 1)[0]
    if clean.endswith("/health"):
        return clean[: -len("/health")]
    return clean


def _coerce_to_registration(item: Any, default_name: str | None = None) -> Registration:
    from langchain_azure_compute.graph import Registration as RegCls
    from langchain_azure_compute.graph import attach_durability

    if isinstance(item, RegCls):
        return item
    inferred_name = (
        default_name
        or getattr(item, "_durable_name", None)
        or getattr(item, "name", None)
    )
    if not inferred_name or str(inferred_name) == "LangGraph":
        raise ValueError(
            "Pass a `name` (or `{name: graph}` mapping / "
            "`attach_durability(graph, name=...)`) when registering a graph"
        )
    return attach_durability(item, name=str(inferred_name))


def _normalize_registrations(
    registrations: Registration | Iterable[Any] | Mapping[str, Any],
    *,
    default_name: str | None = None,
) -> dict[str, Registration]:
    from langchain_azure_compute.graph import Registration as RegCls

    if isinstance(registrations, RegCls):
        return {registrations.name: registrations}
    if isinstance(registrations, Mapping):
        out: dict[str, Registration] = {}
        for k, v in registrations.items():
            reg = _coerce_to_registration(v, default_name=str(k))
            out[str(k)] = reg
        return out
    if hasattr(registrations, "get_state") or hasattr(registrations, "invoke"):
        reg = _coerce_to_registration(registrations, default_name=default_name)
        return {reg.name: reg}
    out_iter: dict[str, Registration] = {}
    for item in registrations:
        reg = _coerce_to_registration(item, default_name=default_name)
        out_iter[reg.name] = reg
    return out_iter


def _is_client_sdk(obj: Any) -> bool:
    return hasattr(obj, "schedule_new_orchestration") and hasattr(
        obj, "get_orchestration_state"
    )


def _extract_caller_principal(req: Any) -> str | None:
    """Extract the authenticated caller identity from EasyAuth headers."""
    headers = getattr(req, "headers", None)
    if not headers or not hasattr(headers, "items"):
        return None
    lowered = {str(k).lower(): str(v).strip() for k, v in headers.items() if v}
    for header_name in ("x-ms-client-principal-id", "x-ms-client-principal-name"):
        val = lowered.get(header_name)
        if val:
            return val
    return None


class _FunctionAppState:
    """Internal per-`FunctionApp` state tracking registered graphs, DTS client, and.

    optional worker.
    """

    def __init__(
        self, sdk_or_options: Any = None, *, start_worker: bool = False
    ) -> None:
        self.registrations: dict[str, Registration] = {}
        self.thread_by_instance: dict[str, str] = {}
        self.owner_by_instance: dict[str, str] = {}
        self._sdk_or_options = sdk_or_options
        self._want_worker = bool(start_worker)
        self._client_sdk: Any = (
            sdk_or_options if _is_client_sdk(sdk_or_options) else None
        )
        self._owns_client = False
        self._worker: DurableTaskSchedulerWorker | None = None
        self._registered_on_worker: set[str] = set()
        self._lock = threading.Lock()

    def _resolve_options(self) -> dict[str, Any]:
        from langchain_azure_compute.graph import sdk_options

        if isinstance(self._sdk_or_options, Mapping):
            return dict(self._sdk_or_options)
        if isinstance(self._sdk_or_options, str):
            return sdk_options(self._sdk_or_options)
        return sdk_options()

    def add_registrations(
        self,
        regs: Mapping[str, Registration],
        *,
        sdk_or_options: Any = None,
        start_worker: bool = False,
    ) -> None:
        with self._lock:
            if sdk_or_options is not None:
                self._sdk_or_options = sdk_or_options
                if _is_client_sdk(sdk_or_options):
                    self._client_sdk = sdk_or_options
                    self._owns_client = False
            if start_worker:
                self._want_worker = True
            self.registrations.update(regs)
            if self._worker is not None:
                for name, reg in self.registrations.items():
                    if name not in self._registered_on_worker:
                        reg.register(self._worker)
                        self._registered_on_worker.add(name)

    def ensure_worker(self) -> DurableTaskSchedulerWorker | None:
        with self._lock:
            if self._want_worker and not _is_client_sdk(self._sdk_or_options):
                opts = self._resolve_options()
                self._ensure_worker_locked(opts)
            return self._worker

    def _ensure_worker_locked(self, opts: dict[str, Any]) -> None:
        if self._worker is None:
            worker = DurableTaskSchedulerWorker(**opts)
            for name, reg in self.registrations.items():
                reg.register(worker)
                self._registered_on_worker.add(name)
            worker.start()
            self._worker = worker
            atexit.register(self.stop)
        else:
            for name, reg in self.registrations.items():
                if name not in self._registered_on_worker:
                    reg.register(self._worker)
                    self._registered_on_worker.add(name)

    def get_sdk(self) -> ClientProtocol:
        with self._lock:
            if (
                self._want_worker
                and self._worker is None
                and not _is_client_sdk(self._sdk_or_options)
            ):
                opts = self._resolve_options()
                self._ensure_worker_locked(opts)
            if self._client_sdk is None:
                opts = self._resolve_options()
                self._client_sdk = DurableTaskSchedulerClient(**opts)
                self._owns_client = True
                atexit.register(self.stop)
            return self._client_sdk  # type: ignore[no-any-return]

    def stop(self) -> None:
        with self._lock:
            if self._worker is not None:
                try:
                    self._worker.stop()
                except Exception:
                    pass
                self._worker = None
                self._registered_on_worker.clear()
            if self._owns_client and self._client_sdk is not None:
                close_fn = getattr(self._client_sdk, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                self._client_sdk = None
                self._owns_client = False


def register_function_routes(
    app: Any,
    registrations: Registration | Iterable[Registration] | Mapping[str, Registration],
    sdk_or_options: Any = None,
    *,
    start_worker: bool = False,
) -> Any:
    """Register Durable LangGraph HTTP trigger routes onto an Azure Functions v2.

    `func.FunctionApp`.

    Can be called once with multiple `Registration`s or multiple times on the same `app`
    to add
    graphs incrementally without duplicating Azure Functions route definitions.

    Args:
        app: Function app receiving the HTTP routes.
        registrations: Graph registrations, individually or as a collection.
        sdk_or_options: Existing DTS client or SDK connection options.
        start_worker: Whether to start an embedded worker for these graphs.

    Returns:
        The supplied function app with routes registered.
    """
    _require_azure_functions()
    regs = _normalize_registrations(registrations)
    state: _FunctionAppState | None = getattr(app, "_langgraph_state", None)
    if state is not None:
        state.add_registrations(
            regs, sdk_or_options=sdk_or_options, start_worker=start_worker
        )
        return app

    state = _FunctionAppState(sdk_or_options, start_worker=start_worker)
    state.add_registrations(
        regs, sdk_or_options=sdk_or_options, start_worker=start_worker
    )
    setattr(app, "_langgraph_state", state)
    setattr(app, "start_worker", state.ensure_worker)
    setattr(app, "stop_worker", state.stop)

    def _handle_start(req: Any) -> Any:
        name = req.route_params.get("name", "")
        reg = state.registrations.get(name)
        if reg is None:
            return _json_response({"error": f"Unknown graph {name!r}"}, status_code=404)

        raw_body = req.get_body()
        if not raw_body:
            body: Any = None
        else:
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                return _json_response(
                    {"error": f"Invalid JSON body: {exc}"}, status_code=400
                )

        params = dict(req.params) if getattr(req, "params", None) else {}
        if isinstance(body, Mapping) and "input" in body:
            inp = body["input"]
            instance_id = (
                body.get("instance_id")
                or body.get("instanceId")
                or params.get("instance_id")
                or params.get("instanceId")
            )
            thread_id = (
                body.get("thread_id")
                or body.get("threadId")
                or params.get("thread_id")
                or params.get("threadId")
            )
            config = body.get("config")
            version = body.get("version") or params.get("version")
            reuse_existing = bool(body.get("reuse_existing", True))
            max_concurrency = body.get("max_concurrency")
            node_max_concurrency = body.get("node_max_concurrency")
        else:
            inp = body
            instance_id = params.get("instance_id") or params.get("instanceId")
            thread_id = params.get("thread_id") or params.get("threadId")
            config = None
            version = params.get("version")
            reuse_existing = True
            max_concurrency = None
            node_max_concurrency = None

        caller = _extract_caller_principal(req)
        if instance_id is not None:
            existing_owner = state.owner_by_instance.get(str(instance_id))
            if existing_owner is not None and existing_owner != caller:
                return _json_response(
                    {"error": "Caller is not authorized for this instance"},
                    status_code=403,
                )

        sdk = state.get_sdk()
        client = reg.client(sdk)
        run = client.start(
            inp,
            instance_id=str(instance_id) if instance_id is not None else None,
            thread_id=str(thread_id) if thread_id is not None else None,
            reuse_existing=reuse_existing,
            max_concurrency=int(max_concurrency)
            if max_concurrency is not None
            else None,
            node_max_concurrency=int(node_max_concurrency)
            if node_max_concurrency is not None
            else None,
            config=config,
            version=str(version) if version is not None else None,
        )
        state.thread_by_instance[run.instance_id] = run.thread_id
        if caller is not None and run.instance_id not in state.owner_by_instance:
            state.owner_by_instance[run.instance_id] = caller
        api_base = _extract_api_base(req.url)
        return _json_response(
            {
                "instanceId": run.instance_id,
                "threadId": run.thread_id,
                "statusQueryGetUri": (
                    f"{api_base}/graphs/{name}/status/{run.instance_id}"
                ),
                "respondUri": (
                    f"{api_base}/graphs/{name}/respond/"
                    f"{run.instance_id}/{{interruptId}}"
                ),
            },
            status_code=202,
        )

    def _handle_status(req: Any) -> Any:
        name = req.route_params.get("name", "")
        instance_id = req.route_params.get("instance_id", "")
        reg = state.registrations.get(name)
        if reg is None:
            return _json_response({"error": f"Unknown graph {name!r}"}, status_code=404)
        expected_owner = state.owner_by_instance.get(instance_id)
        if expected_owner is not None and expected_owner != _extract_caller_principal(
            req
        ):
            return _json_response(
                {"error": "Caller is not authorized for this instance"},
                status_code=403,
            )

        params = dict(req.params) if getattr(req, "params", None) else {}
        hint_thread = (
            params.get("thread_id")
            or params.get("threadId")
            or state.thread_by_instance.get(instance_id)
        )
        sdk = state.get_sdk()
        run = reg.client(sdk).attach_run(instance_id, hint_thread)
        runtime_status = run.status()
        if runtime_status == "NOT_FOUND":
            return _json_response(
                {
                    "instanceId": instance_id,
                    "runtimeStatus": "NOT_FOUND",
                    "error": f"Instance {instance_id!r} not found",
                },
                status_code=404,
            )

        api_base = _extract_api_base(req.url)
        interrupts = run.pending_interrupts()
        pending_payload = [
            {
                "interruptId": intr.interrupt_id,
                "value": intr.value,
                "step": intr.step,
                "ckpt": intr.ckpt,
                "respondUrl": (
                    f"{api_base}/graphs/{name}/respond/"
                    f"{instance_id}/{intr.interrupt_id}"
                ),
            }
            for intr in interrupts
        ]
        output: Any = None
        error_msg: str | None = None
        if runtime_status == "COMPLETED":
            output = run.wait_result(timeout=5)
        elif runtime_status == "FAILED":
            orch_state = sdk.get_orchestration_state(instance_id, fetch_payloads=True)
            if (
                orch_state is not None
                and getattr(orch_state, "failure_details", None) is not None
            ):
                error_msg = orch_state.failure_details.message

        resp_body: dict[str, Any] = {
            "instanceId": run.instance_id,
            "threadId": run.thread_id,
            "runtimeStatus": runtime_status,
            "pendingInterrupts": pending_payload,
            "output": output,
        }
        if error_msg is not None:
            resp_body["error"] = error_msg
        return _json_response(resp_body, status_code=200)

    def _handle_respond(req: Any) -> Any:
        from langchain_azure_compute.graph import RunError

        name = req.route_params.get("name", "")
        instance_id = req.route_params.get("instance_id", "")
        route_interrupt_id = req.route_params.get("interrupt_id")
        reg = state.registrations.get(name)
        if reg is None:
            return _json_response({"error": f"Unknown graph {name!r}"}, status_code=404)
        expected_owner = state.owner_by_instance.get(instance_id)
        if expected_owner is not None and expected_owner != _extract_caller_principal(
            req
        ):
            return _json_response(
                {"error": "Caller is not authorized for this instance"},
                status_code=403,
            )

        raw_body = req.get_body()
        has_body = bool(raw_body)
        if not has_body:
            body: Any = None
        else:
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                return _json_response(
                    {"error": f"Invalid JSON body: {exc}"}, status_code=400
                )

        params = dict(req.params) if getattr(req, "params", None) else {}
        if isinstance(body, Mapping) and "value" in body:
            value = body["value"]
            body_interrupt_id = body.get("interrupt_id") or body.get("interruptId")
            version = body.get("version") or params.get("version")
            as_node = body.get("as_node") or params.get("as_node")
            values = body["values"] if "values" in body else _MISSING
        else:
            value = body
            body_interrupt_id = None
            version = params.get("version")
            as_node = params.get("as_node")
            values = _MISSING

        eff_interrupt_id = (
            route_interrupt_id
            or body_interrupt_id
            or params.get("interrupt_id")
            or params.get("interruptId")
        )

        sdk = state.get_sdk()
        hint_thread = (
            params.get("thread_id")
            or params.get("threadId")
            or state.thread_by_instance.get(instance_id)
        )
        run = reg.client(sdk).attach_run(instance_id, hint_thread)
        if run.status() == "NOT_FOUND":
            return _json_response(
                {
                    "instanceId": instance_id,
                    "error": f"Instance {instance_id!r} not found",
                },
                status_code=404,
            )

        resume_kwargs: dict[str, Any] = {
            "version": str(version) if version is not None else None,
            "as_node": str(as_node) if as_node is not None else None,
        }
        if values is not _MISSING:
            resume_kwargs["values"] = values

        try:
            if eff_interrupt_id is not None:
                run.resume(str(eff_interrupt_id), value, **resume_kwargs)
            elif has_body:
                run.resume(value, **resume_kwargs)
            else:
                run.resume(**resume_kwargs)
        except RunError as exc:
            return _json_response(
                {"instanceId": instance_id, "error": str(exc)}, status_code=409
            )
        except ValueError as exc:
            return _json_response(
                {"instanceId": instance_id, "error": str(exc)}, status_code=400
            )

        return _json_response(
            {
                "instanceId": run.instance_id,
                "threadId": run.thread_id,
                "interruptId": str(eff_interrupt_id)
                if eff_interrupt_id is not None
                else None,
                "resumed": True,
            },
            status_code=200,
        )

    @app.route(route="graphs/{name}/start", methods=["POST"])
    def langgraph_start(req: Any) -> Any:
        return _handle_start(req)

    @app.route(route="workflow/{name}/run", methods=["POST"])
    def langgraph_workflow_run(req: Any) -> Any:
        return _handle_start(req)

    @app.route(route="graphs/{name}/status/{instance_id}", methods=["GET"])
    def langgraph_status(req: Any) -> Any:
        return _handle_status(req)

    @app.route(route="workflow/{name}/status/{instance_id}", methods=["GET"])
    def langgraph_workflow_status(req: Any) -> Any:
        return _handle_status(req)

    @app.route(route="graphs/{name}/respond/{instance_id}", methods=["POST"])
    def langgraph_respond(req: Any) -> Any:
        return _handle_respond(req)

    @app.route(
        route="graphs/{name}/respond/{instance_id}/{interrupt_id}", methods=["POST"]
    )
    def langgraph_respond_interrupt(req: Any) -> Any:
        return _handle_respond(req)

    @app.route(route="workflow/{name}/respond/{instance_id}", methods=["POST"])
    def langgraph_workflow_respond(req: Any) -> Any:
        return _handle_respond(req)

    @app.route(
        route="workflow/{name}/respond/{instance_id}/{interrupt_id}", methods=["POST"]
    )
    def langgraph_workflow_respond_interrupt(req: Any) -> Any:
        return _handle_respond(req)

    @app.route(route="health", methods=["GET"])
    def langgraph_health(req: Any) -> Any:
        return _json_response(
            {"status": "healthy", "graphs": list(state.registrations.keys())},
            status_code=200,
        )

    return app


def create_function_app(
    registrations: Registration | Iterable[Any] | Mapping[str, Any] | Any = None,
    dts_options: Any = None,
    *,
    graph: Any = None,
    graphs: Mapping[str, Any] | Iterable[Any] | None = None,
    name: str | None = None,
    auth_level: Any = None,
    start_worker: bool = True,
) -> Any:
    """Create an Azure Functions v2 `func.FunctionApp` wired with durable LangGraph.

    HTTP routes.

    Accepts either `Registration` object(s) OR raw compiled LangGraph `StateGraph`s (via
    `graph=..., name=...`
    or `graphs={"expense_approval": graph}`), automatically calling
    `attach_durability()` when needed.
    Also attaches a `.serve(host="127.0.0.1", port=7071)` method onto the returned `app`
    so you can
    run `python function_app.py` directly during local development without requiring
    `func start`.

    Args:
        registrations: Graph registrations or compiled graphs to expose.
        dts_options: DTS client or connection options.
        graph: Single compiled graph, used when registrations and graphs are omitted.
        graphs: Named graphs or graph collection, used when registrations is omitted.
        name: Name for a single raw graph.
        auth_level: HTTP authorization level; defaults to `func.AuthLevel.FUNCTION`.
        start_worker: Whether to start an embedded DTS worker.

    Returns:
        A function app with graph routes and a local serve method.

    Raises:
        ValueError: If no graph or registration is supplied.
    """
    func = _require_azure_functions()
    eff_auth = auth_level if auth_level is not None else func.AuthLevel.FUNCTION
    app = func.FunctionApp(http_auth_level=eff_auth)
    target = (
        registrations
        if registrations is not None
        else (graphs if graphs is not None else graph)
    )
    if target is None:
        raise ValueError(
            "Provide `registrations`, `graph`, or `graphs` to "
            "create_function_app / LangGraphFunctionApp"
        )
    norm = _normalize_registrations(target, default_name=name)
    register_function_routes(app, norm, dts_options, start_worker=start_worker)

    def _serve(*, host: str = "127.0.0.1", port: int = 7071) -> None:
        state: _FunctionAppState | None = getattr(app, "_langgraph_state", None)
        if state is not None:
            state.ensure_worker()
        server = FunctionAppHttpServer(app, host=host, port=port)
        server.start()
        print(
            f"LangGraphFunctionApp listening at {server.base_url}/api "
            f"(graphs={list(norm.keys())})",
            flush=True,
        )
        try:
            while True:
                threading.Event().wait(3600)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            stop_function_app(app)

    setattr(app, "serve", _serve)
    return app


def LangGraphFunctionApp(
    registrations: Registration | Iterable[Any] | Mapping[str, Any] | Any = None,
    dts_options: Any = None,
    *,
    graph: Any = None,
    graphs: Mapping[str, Any] | Iterable[Any] | None = None,
    name: str | None = None,
    auth_level: Any = None,
    start_worker: bool = True,
) -> Any:
    """One-line Azure Functions host for durable LangGraph workflows (counterpart to.

    MAF's `AgentFunctionApp`).

    Args:
        registrations: Registrations or raw compiled graphs to expose.
        dts_options: DTS client or connection options.
        graph: Single raw graph to register.
        graphs: Named graphs or a collection of graphs.
        name: Name for a single raw graph.
        auth_level: HTTP authorization level; defaults to `func.AuthLevel.FUNCTION`.
        start_worker: Whether to start an embedded DTS worker.

    Returns:
        The function app created by create_function_app.

    Raises:
        ValueError: If no graph or registration is supplied.

    Examples:
        `app = LangGraphFunctionApp(graph=expense_graph, name="expense_approval")`
        `app = LangGraphFunctionApp(graphs={"expense_approval": expense_graph, "triage":
        triage_graph})`
    """
    return create_function_app(
        registrations,
        dts_options,
        graph=graph,
        graphs=graphs,
        name=name,
        auth_level=auth_level,
        start_worker=start_worker,
    )


def stop_function_app(app: Any) -> None:
    """Stop any embedded `DurableTaskSchedulerWorker` and close the managed DTS client.

    on `app`.
    """
    state: _FunctionAppState | None = getattr(app, "_langgraph_state", None)
    if state is not None:
        state.stop()


def _match_route_template(template: str, path: str) -> dict[str, str] | None:
    t_parts = [p for p in template.strip("/").split("/") if p]
    p_parts = [p for p in path.strip("/").split("/") if p]
    if len(t_parts) != len(p_parts):
        return None
    params: dict[str, str] = {}
    for t_seg, p_seg in zip(t_parts, p_parts):
        if t_seg.startswith("{") and t_seg.endswith("}"):
            key = t_seg[1:-1].rstrip("?")
            params[key] = urllib.parse.unquote(p_seg)
        elif t_seg != p_seg:
            return None
    return params


def dispatch_function_request(app: Any, req: Any) -> Any:
    """Dispatch an `azure.functions.HttpRequest` to its matching `@app.route` handler.

    on `app`.

    Matches the request URL path (stripping any `/api/` prefix) and HTTP method against
    the function
    builders registered on `app`, populates `route_params` and query `params` if not
    already set,
    and invokes the user function directly.
    """
    func = _require_azure_functions()
    parsed = urllib.parse.urlsplit(req.url)
    rel_path = parsed.path.lstrip("/")
    if rel_path.startswith("api/"):
        rel_path = rel_path[len("api/") :]
    elif rel_path == "api":
        rel_path = ""

    method = str(req.method or "GET").upper()
    builders = getattr(app, "_function_builders", [])
    auth_level = getattr(app, "auth_level", func.AuthLevel.ANONYMOUS)

    for fb in builders:
        fn = fb.build(auth_level)
        trigger = fn.get_trigger()
        if trigger is None or getattr(trigger, "type", None) != "httpTrigger":
            continue
        allowed_methods = [
            str(getattr(m, "value", m)).upper()
            for m in (getattr(trigger, "methods", None) or [])
        ]
        if allowed_methods and method not in allowed_methods:
            continue
        route_template = getattr(trigger, "route", None) or fn.get_function_name()
        matched_params = _match_route_template(route_template, rel_path)
        if matched_params is None:
            continue

        existing_route_params = (
            dict(req.route_params) if getattr(req, "route_params", None) else {}
        )
        merged_route_params = {**matched_params, **existing_route_params}
        query_params = {
            k: v[0]
            for k, v in urllib.parse.parse_qs(
                parsed.query, keep_blank_values=True
            ).items()
        }
        existing_query_params = dict(req.params) if getattr(req, "params", None) else {}
        merged_query_params = {**query_params, **existing_query_params}

        bound_req = func.HttpRequest(
            method=method,
            url=req.url,
            headers=dict(req.headers) if getattr(req, "headers", None) else {},
            params=merged_query_params,
            route_params=merged_route_params,
            body=req.get_body(),
        )
        user_fn = fn.get_user_function()
        return user_fn(bound_req)

    return _json_response(
        {"error": f"No Azure Function route matched {method} {parsed.path}"},
        status_code=404,
    )


class FunctionAppHttpServer:
    """Lightweight multithreaded HTTP server that hosts an.

    `azure.functions.FunctionApp` over TCP HTTP.

    Translates incoming HTTP requests into real `azure.functions.HttpRequest` objects
    and writes the
    returned `azure.functions.HttpResponse` status, headers, and body back to the
    client. Ideal for
    local multi-worker testing, CI pipelines, and standalone execution without requiring
    the external
    `func` CLI host binary.
    """

    def __init__(self, app: Any, *, host: str = "127.0.0.1", port: int = 0) -> None:
        """Bind the function app to a local HTTP address.

        Args:
            app: Azure Functions app whose routes are dispatched.
            host: Interface to bind; defaults to loopback.
            port: Listening port, or zero to select an available port.
        """
        _require_azure_functions()
        self.app = app
        self.host = host
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self.port: int = int(self._server.server_address[1])
        self.base_url: str = f"http://{self.host}:{self.port}"
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        func = _require_azure_functions()
        app_ref = self.app
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def _handle_all(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length) if length > 0 else b""
                full_url = f"{server_ref.base_url}{self.path}"
                headers_dict = {k: v for k, v in self.headers.items()}
                az_req = func.HttpRequest(
                    method=self.command,
                    url=full_url,
                    headers=headers_dict,
                    params={},
                    route_params={},
                    body=raw_body,
                )
                try:
                    az_resp = dispatch_function_request(app_ref, az_req)
                except Exception as exc:
                    az_resp = _json_response({"error": str(exc)}, status_code=500)

                body_bytes: bytes = az_resp.get_body() or b""
                self.send_response(int(az_resp.status_code))
                resp_headers = dict(getattr(az_resp, "headers", {}) or {})
                if "content-type" not in {k.lower() for k in resp_headers}:
                    self.send_header(
                        "Content-Type", getattr(az_resp, "mimetype", "application/json")
                    )
                for hk, hv in resp_headers.items():
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                if body_bytes:
                    self.wfile.write(body_bytes)

            def do_GET(self) -> None:  # noqa: N802
                self._handle_all()

            def do_POST(self) -> None:  # noqa: N802
                self._handle_all()

            def log_message(self, format: str, *args: Any) -> None:
                return

        return _Handler

    def start(self) -> FunctionAppHttpServer:
        """Start the HTTP server and any configured embedded worker.

        Returns:
            This server, ready to accept requests.
        """
        state: _FunctionAppState | None = getattr(self.app, "_langgraph_state", None)
        if state is not None:
            state.ensure_worker()
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the HTTP server and release the function app's worker and client."""
        if self._thread is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
            self._thread = None
        stop_function_app(self.app)

    def __enter__(self) -> FunctionAppHttpServer:
        """Start the server when entering a context manager."""
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Stop the server when leaving a context manager."""
        self.stop()
