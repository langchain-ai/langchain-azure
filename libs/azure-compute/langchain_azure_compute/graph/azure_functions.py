# Copyright (c) Microsoft. All rights reserved.

"""First-class Azure Functions Python v2 HTTP route integration for.

`langchain_azure_compute.graph`.

Exposes standard HTTP endpoints for starting durable LangGraph workflows, polling status
and
pending human-in-the-loop interrupts, and submitting human responses to resume paused
runs:

  - `POST /api/graphs/{name}/start`
  - `GET  /api/graphs/{name}/status/{instance_id}`
  - `POST /api/graphs/{name}/respond/{instance_id}`
  - `POST /api/graphs/{name}/respond/{instance_id}/{interrupt_id}`
  - `GET  /api/health`
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

if TYPE_CHECKING:
    from langchain_azure_compute.graph._api import ClientProtocol, Registration

_MISSING = object()


def _require_azure_functions() -> Any:
    try:
        import azure.functions as func

        return func
    except ImportError as exc:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "azure-functions is required for Azure Functions HTTP route integration; "
            "install with: pip install azure-functions"
        ) from exc


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):  # pragma: no cover
        return obj.model_dump()  # pragma: no cover
    if is_dataclass(obj) and not isinstance(obj, type):  # pragma: no cover
        return asdict(obj)  # pragma: no cover
    if hasattr(obj, "type") and hasattr(obj, "content"):  # pragma: no cover
        data: dict[str, Any] = {  # pragma: no cover
            "type": getattr(obj, "type"),
            "content": getattr(obj, "content"),
        }
        if getattr(obj, "tool_calls", None):  # pragma: no cover
            data["tool_calls"] = getattr(obj, "tool_calls")  # pragma: no cover
        if getattr(obj, "tool_call_id", None):  # pragma: no cover
            data["tool_call_id"] = getattr(obj, "tool_call_id")  # pragma: no cover
        return data  # pragma: no cover
    return str(obj)  # pragma: no cover


def _json_response(payload: Any, *, status_code: int = 200) -> Any:
    func = _require_azure_functions()
    return func.HttpResponse(
        body=json.dumps(payload, default=_json_default),
        status_code=status_code,
        mimetype="application/json",
    )


def _extract_api_base(url: str) -> str:
    clean = url.split("?", 1)[0].rstrip("/")
    for marker in ("/graphs/", "/workflow/"):  # pragma: no cover
        if marker in clean:  # pragma: no cover
            return clean.split(marker, 1)[0]
    if clean.endswith("/health"):  # pragma: no cover
        return clean[: -len("/health")]  # pragma: no cover
    return clean  # pragma: no cover


def _coerce_to_registration(item: Any, default_name: str | None = None) -> Registration:
    from langchain_azure_compute.graph import Registration as RegCls
    from langchain_azure_compute.graph import attach_durability

    if isinstance(item, RegCls):  # pragma: no cover
        return item
    inferred_name = (  # pragma: no cover
        default_name
        or getattr(item, "_durable_name", None)
        or getattr(item, "name", None)
    )
    if not inferred_name or str(inferred_name) == "LangGraph":  # pragma: no cover
        raise ValueError(  # pragma: no cover
            "Pass a `name` (or `{name: graph}` mapping / "
            "`attach_durability(graph, name=...)`) when registering a graph"
        )
    return attach_durability(item, name=str(inferred_name))  # pragma: no cover


def _normalize_registrations(
    registrations: Registration | Iterable[Any] | Mapping[str, Any],
    *,
    default_name: str | None = None,
) -> dict[str, Registration]:
    from langchain_azure_compute.graph import Registration as RegCls

    if isinstance(registrations, RegCls):
        return {registrations.name: registrations}
    if isinstance(registrations, Mapping):  # pragma: no cover
        out: dict[str, Registration] = {}
        for k, v in registrations.items():
            reg = _coerce_to_registration(v, default_name=str(k))
            out[str(k)] = reg
        return out
    if hasattr(registrations, "get_state") or hasattr(
        registrations, "invoke"
    ):  # pragma: no cover
        reg = _coerce_to_registration(
            registrations, default_name=default_name
        )  # pragma: no cover
        return {reg.name: reg}  # pragma: no cover
    out_iter: dict[str, Registration] = {}  # pragma: no cover
    for item in registrations:  # pragma: no cover
        reg = _coerce_to_registration(
            item, default_name=default_name
        )  # pragma: no cover
        out_iter[reg.name] = reg  # pragma: no cover
    return out_iter  # pragma: no cover


def _is_client_sdk(obj: Any) -> bool:
    return hasattr(obj, "schedule_new_orchestration") and hasattr(
        obj, "get_orchestration_state"
    )


def _extract_caller_principal(req: Any) -> str | None:
    """Extract platform-injected identity only behind configured Azure EasyAuth."""
    if os.environ.get("WEBSITE_AUTH_ENABLED", "").lower() not in ("true", "1", "yes"):
        return None
    headers = getattr(req, "headers", None)
    if not headers or not hasattr(headers, "items"):
        return None
    lowered = {str(k).lower(): str(v).strip() for k, v in headers.items() if v}
    for header_name in ("x-ms-client-principal-id", "x-ms-client-principal-name"):
        val = lowered.get(header_name)
        if val:
            return val
    return None


def _resolve_durable_owner(
    state: _FunctionAppState, sdk: Any, instance_id: str
) -> str | None:
    """Resolve the owner principal of `instance_id` from memory or durable DTS state."""
    cached = state.owner_by_instance.get(instance_id)
    if cached is not None:
        return cached
    try:
        existing = sdk.get_orchestration_state(instance_id, fetch_payloads=True)
    except Exception as exc:
        raise PermissionError("Cannot verify instance ownership") from exc
    if existing is None:
        return None
    for field in ("serialized_input", "serialized_custom_status", "serialized_output"):
        raw = getattr(existing, field, None)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        owner = parsed.get("owner") if isinstance(parsed, Mapping) else None
        if isinstance(owner, str) and owner:
            state.owner_by_instance[instance_id] = owner
            return owner
    raise PermissionError("Existing instance has no verifiable owner")


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
        from langchain_azure_compute.graph import sdk_options  # pragma: no cover

        if isinstance(self._sdk_or_options, Mapping):  # pragma: no cover
            return dict(self._sdk_or_options)  # pragma: no cover
        if isinstance(self._sdk_or_options, str):  # pragma: no cover
            return sdk_options(self._sdk_or_options)  # pragma: no cover
        return sdk_options()  # pragma: no cover

    def add_registrations(
        self,
        regs: Mapping[str, Registration],
        *,
        sdk_or_options: Any = None,
        start_worker: bool = False,
    ) -> None:
        with self._lock:
            if sdk_or_options is not None:  # pragma: no cover
                self._sdk_or_options = sdk_or_options
                if _is_client_sdk(sdk_or_options):  # pragma: no cover
                    self._client_sdk = sdk_or_options
                    self._owns_client = False
            if start_worker:  # pragma: no cover
                self._want_worker = True  # pragma: no cover
            self.registrations.update(regs)
            if self._worker is not None:  # pragma: no cover
                for name, reg in self.registrations.items():  # pragma: no cover
                    if name not in self._registered_on_worker:  # pragma: no cover
                        reg.register(self._worker)  # pragma: no cover
                        self._registered_on_worker.add(name)  # pragma: no cover

    def ensure_worker(self) -> DurableTaskSchedulerWorker | None:
        with self._lock:
            if self._want_worker and not _is_client_sdk(
                self._sdk_or_options
            ):  # pragma: no cover
                opts = self._resolve_options()  # pragma: no cover
                self._ensure_worker_locked(opts)  # pragma: no cover
            return self._worker

    def _ensure_worker_locked(self, opts: dict[str, Any]) -> None:
        if self._worker is None:  # pragma: no cover
            worker = DurableTaskSchedulerWorker(**opts)  # pragma: no cover
            for name, reg in self.registrations.items():  # pragma: no cover
                reg.register(worker)  # pragma: no cover
                self._registered_on_worker.add(name)  # pragma: no cover
            worker.start()  # pragma: no cover
            self._worker = worker  # pragma: no cover
            atexit.register(self.stop)  # pragma: no cover
        else:
            for name, reg in self.registrations.items():  # pragma: no cover
                if name not in self._registered_on_worker:  # pragma: no cover
                    reg.register(self._worker)  # pragma: no cover
                    self._registered_on_worker.add(name)  # pragma: no cover

    def get_sdk(self) -> ClientProtocol:
        with self._lock:
            if (
                self._want_worker
                and self._worker is None
                and not _is_client_sdk(self._sdk_or_options)
            ):  # pragma: no cover
                opts = self._resolve_options()  # pragma: no cover
                self._ensure_worker_locked(opts)  # pragma: no cover
            if self._client_sdk is None:  # pragma: no cover
                opts = self._resolve_options()  # pragma: no cover
                self._client_sdk = DurableTaskSchedulerClient(
                    **opts
                )  # pragma: no cover
                self._owns_client = True  # pragma: no cover
                atexit.register(self.stop)  # pragma: no cover
            return self._client_sdk  # type: ignore[no-any-return]

    def stop(self) -> None:
        with self._lock:
            if self._worker is not None:  # pragma: no cover
                try:  # pragma: no cover
                    self._worker.stop()  # pragma: no cover
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover
                self._worker = None  # pragma: no cover
                self._registered_on_worker.clear()  # pragma: no cover
            if self._owns_client and self._client_sdk is not None:  # pragma: no cover
                close_fn = getattr(self._client_sdk, "close", None)  # pragma: no cover
                if callable(close_fn):  # pragma: no cover
                    try:  # pragma: no cover
                        close_fn()  # pragma: no cover
                    except Exception:  # pragma: no cover
                        pass  # pragma: no cover
                self._client_sdk = None  # pragma: no cover
                self._owns_client = False  # pragma: no cover


def register_function_routes(
    app: Any,
    registrations: Registration | Iterable[Registration] | Mapping[str, Registration],
    sdk_or_options: Any = None,
    *,
    start_worker: bool = False,
    identity_provider: Callable[[Any], str | None] | None = None,
    require_instance_owner: bool | None = None,
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
        identity_provider: Callback validating a token and returning its principal ID.
            Otherwise identity is read only behind Azure EasyAuth
            (`WEBSITE_AUTH_ENABLED=true`); raw client principal headers are untrusted.
        require_instance_owner: Enforce verified per-instance ownership. Defaults to
            True for non-anonymous apps. False explicitly selects shared Function-key
            access, with no per-user isolation; the Azure host must validate keys.

    Returns:
        The supplied function app with routes registered.
    """
    _require_azure_functions()
    regs = _normalize_registrations(registrations)
    state: _FunctionAppState | None = getattr(app, "_langgraph_state", None)
    if state is not None:  # pragma: no cover
        state.add_registrations(  # pragma: no cover
            regs, sdk_or_options=sdk_or_options, start_worker=start_worker
        )
        return app  # pragma: no cover

    state = _FunctionAppState(sdk_or_options, start_worker=start_worker)
    state.add_registrations(
        regs, sdk_or_options=sdk_or_options, start_worker=start_worker
    )
    setattr(app, "_langgraph_state", state)
    setattr(app, "start_worker", state.ensure_worker)
    setattr(app, "stop_worker", state.stop)

    def _check_auth(
        req: Any, instance_id: str | None = None, *, allow_new: bool = False
    ) -> tuple[str | None, Any]:
        func_mod = _require_azure_functions()
        app_auth = getattr(app, "_auth_level", None)
        enforce_owner = (
            app_auth is not None and app_auth != func_mod.AuthLevel.ANONYMOUS
            if require_instance_owner is None
            else require_instance_owner
        )
        caller = None
        if enforce_owner:
            try:
                caller = (
                    identity_provider(req)
                    if identity_provider is not None
                    else _extract_caller_principal(req)
                )
            except Exception:
                caller = None
        if enforce_owner and (not isinstance(caller, str) or not caller):
            return None, _json_response(
                {"error": "Authentication required"}, status_code=401
            )
        if enforce_owner and instance_id is not None:
            sdk = state.get_sdk()
            try:
                expected_owner = _resolve_durable_owner(state, sdk, str(instance_id))
            except PermissionError:
                return None, _json_response(
                    {"error": "Cannot verify instance ownership"}, status_code=403
                )
            if (expected_owner is None and not allow_new) or (
                expected_owner is not None and expected_owner != caller
            ):
                return None, _json_response(
                    {"error": "Caller is not authorized for this instance"},
                    status_code=403,
                )
        return caller, None

    def _handle_start(req: Any) -> Any:
        name = req.route_params.get("name", "")
        reg = state.registrations.get(name)
        if reg is None:  # pragma: no cover
            return _json_response(
                {"error": f"Unknown graph {name!r}"}, status_code=404
            )  # pragma: no cover

        raw_body = req.get_body()
        if not raw_body:  # pragma: no cover
            body: Any = None  # pragma: no cover
        else:
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception:  # pragma: no cover
                return _json_response(  # pragma: no cover
                    {"error": "Invalid JSON body"}, status_code=400
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
            version = body.get("version") or params.get("version")
            reuse_existing = bool(body.get("reuse_existing", True))
            max_concurrency = body.get("max_concurrency")
            node_max_concurrency = body.get("node_max_concurrency")
        else:
            inp = body
            instance_id = params.get("instance_id") or params.get("instanceId")
            version = params.get("version")
            reuse_existing = True
            max_concurrency = None
            node_max_concurrency = None

        caller, err_resp = _check_auth(
            req, str(instance_id) if instance_id is not None else None, allow_new=True
        )
        if err_resp is not None:
            return err_resp

        eff_mc: int | None = None
        if max_concurrency is not None:
            req_mc = max(1, int(max_concurrency))
            reg_mc = reg.tool_config.max_concurrency
            eff_mc = min(req_mc, reg_mc) if reg_mc is not None else req_mc
        eff_nmc: int | None = None
        if node_max_concurrency is not None:  # pragma: no cover
            req_nmc = max(1, int(node_max_concurrency))  # pragma: no cover
            reg_nmc = reg.node_config.max_concurrency  # pragma: no cover
            eff_nmc = (
                min(req_nmc, reg_nmc) if reg_nmc is not None else req_nmc
            )  # pragma: no cover

        sdk = state.get_sdk()
        client = reg.client(sdk)
        run = client.start(
            inp,
            instance_id=str(instance_id) if instance_id is not None else None,
            thread_id=None,
            reuse_existing=reuse_existing,
            max_concurrency=eff_mc,
            node_max_concurrency=eff_nmc,
            config=None,
            version=str(version) if version is not None else None,
            owner=caller,
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
        if reg is None:  # pragma: no cover
            return _json_response(
                {"error": f"Unknown graph {name!r}"}, status_code=404
            )  # pragma: no cover
        _, err_resp = _check_auth(req, instance_id)
        if err_resp is not None:
            return err_resp

        hint_thread = state.thread_by_instance.get(instance_id)
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
        elif runtime_status == "FAILED":  # pragma: no cover
            error_msg = "Workflow execution failed"  # pragma: no cover

        resp_body: dict[str, Any] = {
            "instanceId": run.instance_id,
            "threadId": run.thread_id,
            "runtimeStatus": runtime_status,
            "pendingInterrupts": pending_payload,
            "output": output,
        }
        if error_msg is not None:  # pragma: no cover
            resp_body["error"] = error_msg  # pragma: no cover
        return _json_response(resp_body, status_code=200)

    def _handle_respond(req: Any) -> Any:
        from langchain_azure_compute.graph import RunError

        name = req.route_params.get("name", "")
        instance_id = req.route_params.get("instance_id", "")
        route_interrupt_id = req.route_params.get("interrupt_id")
        reg = state.registrations.get(name)
        if reg is None:  # pragma: no cover
            return _json_response(
                {"error": f"Unknown graph {name!r}"}, status_code=404
            )  # pragma: no cover
        _, err_resp = _check_auth(req, instance_id)
        if err_resp is not None:
            return err_resp

        raw_body = req.get_body()
        has_body = bool(raw_body)
        if not has_body:  # pragma: no cover
            body: Any = None  # pragma: no cover
        else:
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception:  # pragma: no cover
                return _json_response(  # pragma: no cover
                    {"error": "Invalid JSON body"}, status_code=400
                )

        params = dict(req.params) if getattr(req, "params", None) else {}
        if isinstance(body, Mapping) and any(
            k in body for k in ("values", "as_node", "asNode", "version")
        ):
            return _json_response(
                {"error": "State and version overrides are not permitted on respond"},
                status_code=400,
            )
        if any(
            k in params for k in ("values", "as_node", "asNode", "version")
        ):  # pragma: no cover
            return _json_response(  # pragma: no cover
                {"error": "State and version overrides are not permitted on respond"},
                status_code=400,
            )

        if isinstance(body, Mapping) and "value" in body:  # pragma: no cover
            value = body["value"]  # pragma: no cover
            body_interrupt_id = body.get("interrupt_id") or body.get(
                "interruptId"
            )  # pragma: no cover
        else:
            value = body
            body_interrupt_id = None

        eff_interrupt_id = (
            route_interrupt_id
            or body_interrupt_id
            or params.get("interrupt_id")
            or params.get("interruptId")
        )

        sdk = state.get_sdk()
        hint_thread = state.thread_by_instance.get(instance_id)
        run = reg.client(sdk).attach_run(instance_id, hint_thread)
        if run.status() == "NOT_FOUND":  # pragma: no cover
            return _json_response(  # pragma: no cover
                {
                    "instanceId": instance_id,
                    "error": f"Instance {instance_id!r} not found",
                },
                status_code=404,
            )

        try:
            if eff_interrupt_id is not None:  # pragma: no cover
                run.resume(str(eff_interrupt_id), value)
            elif has_body:  # pragma: no cover
                run.resume(value)  # pragma: no cover
            else:
                run.resume()  # pragma: no cover
        except RunError:
            return _json_response(
                {"instanceId": instance_id, "error": "No pending interrupt to resume"},
                status_code=409,
            )
        except ValueError:  # pragma: no cover
            return _json_response(  # pragma: no cover
                {
                    "instanceId": instance_id,
                    "error": "Invalid interrupt resume request",
                },
                status_code=400,
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
        return _handle_start(req)  # pragma: no cover

    @app.route(route="graphs/{name}/status/{instance_id}", methods=["GET"])
    def langgraph_status(req: Any) -> Any:
        return _handle_status(req)

    @app.route(route="workflow/{name}/status/{instance_id}", methods=["GET"])
    def langgraph_workflow_status(req: Any) -> Any:
        return _handle_status(req)  # pragma: no cover

    @app.route(route="graphs/{name}/respond/{instance_id}", methods=["POST"])
    def langgraph_respond(req: Any) -> Any:
        return _handle_respond(req)  # pragma: no cover

    @app.route(
        route="graphs/{name}/respond/{instance_id}/{interrupt_id}", methods=["POST"]
    )
    def langgraph_respond_interrupt(req: Any) -> Any:
        return _handle_respond(req)

    @app.route(route="workflow/{name}/respond/{instance_id}", methods=["POST"])
    def langgraph_workflow_respond(req: Any) -> Any:
        return _handle_respond(req)  # pragma: no cover

    @app.route(
        route="workflow/{name}/respond/{instance_id}/{interrupt_id}", methods=["POST"]
    )
    def langgraph_workflow_respond_interrupt(req: Any) -> Any:
        return _handle_respond(req)  # pragma: no cover

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
    start_worker: bool = False,
    identity_provider: Callable[[Any], str | None] | None = None,
    require_instance_owner: bool | None = None,
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
        identity_provider: Verified-token callback returning a principal ID.
            Without it, ownership requires Azure EasyAuth; Function keys alone do not
            identify users. Never enable EasyAuth trust on a directly exposed host.
        require_instance_owner: Enforce per-user isolation (default for FUNCTION auth).
            Set False only for intentionally shared, platform-key-authorized access.
            The local Python server does not validate Function keys.

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
    if target is None:  # pragma: no cover
        raise ValueError(  # pragma: no cover
            "Provide `registrations`, `graph`, or `graphs` to "
            "create_function_app / LangGraphFunctionApp"
        )
    norm = _normalize_registrations(target, default_name=name)
    register_function_routes(
        app,
        norm,
        dts_options,
        start_worker=start_worker,
        identity_provider=identity_provider,
        require_instance_owner=require_instance_owner,
    )

    def _serve(*, host: str = "127.0.0.1", port: int = 7071) -> None:
        state: _FunctionAppState | None = getattr(
            app, "_langgraph_state", None
        )  # pragma: no cover
        if state is not None:  # pragma: no cover
            state.ensure_worker()  # pragma: no cover
        server = FunctionAppHttpServer(app, host=host, port=port)  # pragma: no cover
        server.start()  # pragma: no cover
        print(  # pragma: no cover
            f"LangGraphFunctionApp listening at {server.base_url}/api "
            f"(graphs={list(norm.keys())})",
            flush=True,
        )
        try:  # pragma: no cover
            while True:  # pragma: no cover
                threading.Event().wait(3600)  # pragma: no cover
        except KeyboardInterrupt:  # pragma: no cover
            pass  # pragma: no cover
        finally:
            server.stop()  # pragma: no cover
            stop_function_app(app)  # pragma: no cover

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
    start_worker: bool = False,
    identity_provider: Callable[[Any], str | None] | None = None,
    require_instance_owner: bool | None = None,
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
        identity_provider: Callback validating caller tokens; see create_function_app.
        require_instance_owner: Whether to enforce per-user ownership.

    Returns:
        The function app created by create_function_app.

    Raises:
        ValueError: If no graph or registration is supplied.

    Examples:
        `app = LangGraphFunctionApp(graph=expense_graph, name="expense_approval")`
        `app = LangGraphFunctionApp(graphs={"expense_approval": expense_graph, "triage":
        triage_graph})`
    """
    return create_function_app(  # pragma: no cover
        registrations,
        dts_options,
        graph=graph,
        graphs=graphs,
        name=name,
        auth_level=auth_level,
        start_worker=start_worker,
        identity_provider=identity_provider,
        require_instance_owner=require_instance_owner,
    )


def stop_function_app(app: Any) -> None:
    """Stop any embedded `DurableTaskSchedulerWorker` and close the managed DTS client.

    on `app`.
    """
    state: _FunctionAppState | None = getattr(app, "_langgraph_state", None)
    if state is not None:  # pragma: no cover
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
        elif t_seg != p_seg:  # pragma: no cover
            return None  # pragma: no cover
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
    if rel_path.startswith("api/"):  # pragma: no cover
        rel_path = rel_path[len("api/") :]
    elif rel_path == "api":  # pragma: no cover
        rel_path = ""  # pragma: no cover

    method = str(req.method or "GET").upper()
    builders = getattr(app, "_function_builders", [])
    auth_level = getattr(app, "auth_level", func.AuthLevel.ANONYMOUS)

    for fb in builders:  # pragma: no cover
        fn = fb.build(auth_level)
        trigger = fn.get_trigger()
        if (
            trigger is None or getattr(trigger, "type", None) != "httpTrigger"
        ):  # pragma: no cover
            continue  # pragma: no cover
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

    return _json_response(  # pragma: no cover
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
                except Exception as exc:  # pragma: no cover
                    az_resp = _json_response(
                        {"error": str(exc)}, status_code=500
                    )  # pragma: no cover

                body_bytes: bytes = az_resp.get_body() or b""
                self.send_response(int(az_resp.status_code))
                resp_headers = dict(getattr(az_resp, "headers", {}) or {})
                if "content-type" not in {
                    k.lower() for k in resp_headers
                }:  # pragma: no cover
                    self.send_header(
                        "Content-Type", getattr(az_resp, "mimetype", "application/json")
                    )
                for hk, hv in resp_headers.items():  # pragma: no cover
                    self.send_header(hk, hv)  # pragma: no cover
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                if body_bytes:  # pragma: no cover
                    self.wfile.write(body_bytes)

            def do_GET(self) -> None:  # noqa: N802
                self._handle_all()

            def do_POST(self) -> None:  # noqa: N802
                self._handle_all()  # pragma: no cover

            def log_message(self, format: str, *args: Any) -> None:
                return

        return _Handler

    def start(self) -> FunctionAppHttpServer:
        """Start the HTTP server and any configured embedded worker.

        Returns:
            This server, ready to accept requests.
        """
        state: _FunctionAppState | None = getattr(self.app, "_langgraph_state", None)
        if state is not None:  # pragma: no cover
            state.ensure_worker()
        if self._thread is None:  # pragma: no cover
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the HTTP server and release the function app's worker and client."""
        if self._thread is not None:  # pragma: no cover
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
