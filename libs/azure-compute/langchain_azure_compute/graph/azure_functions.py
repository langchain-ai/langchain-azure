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

Production topology A (recommended): thin HTTP routes with start_worker=False on
Functions, ACA, or App Service, plus standalone DurableTaskSchedulerWorker replicas
on ACA/AKS using the azure-durabletask-scheduler KEDA scaler, or a Dedicated App
Service plan with Always On. Topology B: start_worker=True only for local development
(func start) or a Dedicated Always-On plan. Embedded workers are not supported on
Consumption/Flex: HTTP scaling does not track DTS work or keep workers alive.
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import re
import threading
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

if TYPE_CHECKING:
    from langchain_azure_compute.graph._api import ClientProtocol, Registration

_MISSING = object()
MAX_HTTP_BODY_BYTES = 1_000_000
_INSTANCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_RESERVED_RESUME_KEYS = frozenset(("__packed__", "__resume_map__", "checkpoint_bundle"))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        text = str(value).strip().lower()
        if text in ("1", "true", "yes"):
            return True
        if text in ("0", "false", "no", ""):
            return False
    raise ValueError("Expected a boolean (true/false, 1/0, or yes/no)")


def _parse_int(value: Any, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be an integer >= {minimum}") from exc
    if parsed < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return parsed


def _validate_instance_id(value: Any) -> None:
    if not isinstance(value, str) or not _INSTANCE_ID_PATTERN.fullmatch(value):
        raise ValueError("instance_id must match [A-Za-z0-9_.:@-]{1,128}")


def _read_json_body(req: Any) -> tuple[Any, bool]:
    raw = req.get_body()
    if len(raw) > MAX_HTTP_BODY_BYTES:
        raise OverflowError("Request body exceeds 1000000 bytes")
    if not raw:
        return None, False
    try:
        return json.loads(raw.decode("utf-8")), True
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid JSON body") from exc


def _has_reserved_resume_keys(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(_RESERVED_RESUME_KEYS.intersection(value)) or any(
            _has_reserved_resume_keys(v) for v in value.values()
        )
    if isinstance(value, list):
        return any(_has_reserved_resume_keys(v) for v in value)
    return False


class _BoundedCache(OrderedDict[str, str]):
    """FIFO hints with serialized insertions; authorization reads the backend."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __setitem__(self, key: str, value: str) -> None:
        with self._lock:
            super().__setitem__(key, value)
            while len(self) > 1024:
                self.popitem(last=False)


def _easyauth_enabled() -> bool:
    return os.environ.get("WEBSITE_AUTH_ENABLED", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )


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
    """Extract platform-injected identity only behind configured Azure EasyAuth."""
    if not _easyauth_enabled():
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
    state: _FunctionAppState,
    sdk: Any,
    instance_id: str,
    *,
    bypass_cache: bool = True,
) -> str | None:
    """Resolve ownership from DTS, never from a potentially stale host cache.

    bypass_cache is retained for explicit post-start verification; all lookups
    bypass the cache regardless of its value.
    """
    try:
        existing = sdk.get_orchestration_state(instance_id, fetch_payloads=True)
    except Exception as exc:
        raise PermissionError("Cannot verify instance ownership") from exc
    if existing is None:
        state.owner_by_instance.pop(instance_id, None)
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
        self._route_security: (
            tuple[Callable[[Any], str | None] | None, bool, Callable[..., bool] | None]
            | None
        ) = None
        self.thread_by_instance = _BoundedCache()
        self.owner_by_instance = _BoundedCache()
        self._sdk_or_options = (
            dict(sdk_or_options)
            if isinstance(sdk_or_options, Mapping)
            else sdk_or_options
        )
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
        old_worker = None
        with self._lock:
            for name, reg in regs.items():
                if name != reg.name:
                    raise ValueError("Registration mapping keys must match graph names")
                if name in self.registrations and self.registrations[name] is not reg:
                    raise ValueError(f"Graph {name!r} is already registered")
            previous = (
                self.registrations.copy(),
                self._sdk_or_options,
                self._want_worker,
                self._client_sdk,
                self._owns_client,
                self._worker,
                self._registered_on_worker.copy(),
            )
            try:
                if sdk_or_options is not None:
                    if (
                        self._worker is not None or self._client_sdk is not None
                    ) and sdk_or_options != self._sdk_or_options:
                        raise ValueError(
                            "Cannot change connection options on a live app"
                        )
                    self._sdk_or_options = (
                        dict(sdk_or_options)
                        if isinstance(sdk_or_options, Mapping)
                        else sdk_or_options
                    )
                    if _is_client_sdk(sdk_or_options):
                        self._client_sdk = sdk_or_options
                        self._owns_client = False
                self._want_worker = self._want_worker or start_worker
                added = set(regs).difference(self.registrations)
                self.registrations.update(regs)
                if self._worker is not None and added:
                    old_worker, self._worker = self._worker, None
                if self._want_worker and not _is_client_sdk(self._sdk_or_options):
                    self._ensure_worker_locked(self._resolve_options())
            except BaseException:
                (
                    self.registrations,
                    self._sdk_or_options,
                    self._want_worker,
                    self._client_sdk,
                    self._owns_client,
                    self._worker,
                    self._registered_on_worker,
                ) = previous
                raise
        if old_worker is not None:
            try:
                old_worker.stop()
            except Exception:
                pass

    def ensure_worker(self) -> DurableTaskSchedulerWorker | None:
        with self._lock:
            if self._want_worker and not _is_client_sdk(self._sdk_or_options):
                opts = self._resolve_options()
                self._ensure_worker_locked(opts)
            return self._worker

    def _ensure_worker_locked(self, opts: dict[str, Any]) -> None:
        if self._worker is not None:
            return
        worker = DurableTaskSchedulerWorker(**opts)
        failures: list[BaseException] = []

        def start() -> None:
            try:
                for reg in self.registrations.values():
                    reg.register(worker)
                worker.start()
            except BaseException as exc:
                failures.append(exc)

        # SDK 1.13 creates _runLoop with no daemon argument. Starting from a
        # daemon bootstrap makes it inherit daemon=True *before* it starts.
        bootstrap = threading.Thread(target=start, daemon=True)
        bootstrap.start()
        bootstrap.join()
        if failures:
            try:
                worker.stop()
            except Exception:
                pass
            raise failures[0]
        self._worker = worker
        self._registered_on_worker = set(self.registrations)
        atexit.register(self.stop)

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
                if _is_client_sdk(self._sdk_or_options):
                    self._client_sdk = self._sdk_or_options
                    return self._client_sdk  # type: ignore[no-any-return]
                opts = self._resolve_options()
                self._client_sdk = DurableTaskSchedulerClient(**opts)
                self._owns_client = True
                atexit.register(self.stop)
            return self._client_sdk  # type: ignore[no-any-return]

    def stop(self) -> None:
        with self._lock:
            worker, self._worker = self._worker, None
            client, self._client_sdk = self._client_sdk, None
            owns_client, self._owns_client = self._owns_client, False
            self._registered_on_worker.clear()
            self._want_worker = False
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if owns_client and client is not None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def register_function_routes(
    app: Any,
    registrations: Registration | Iterable[Registration] | Mapping[str, Registration],
    sdk_or_options: Any = None,
    *,
    start_worker: bool = False,
    identity_provider: Callable[[Any], str | None] | None = None,
    require_instance_owner: bool | None = None,
    respond_authorizer: Callable[..., bool] | None = None,
) -> Any:
    """Atomically register thin HTTP routes with fail-closed ownership by default.

    Embedded workers (start_worker=True) are for local development or Dedicated
    Always-On plans only; use standalone ACA/AKS workers for Consumption/Flex.
    respond_authorizer optionally accepts (caller, instance_id[, owner]) and
    replaces the default owner-only response policy, but never authentication.
    Incremental registrations inherit the app's security policy; changing it
    or a live app's connection options requires a new app.
    """
    builders = list(getattr(app, "_function_builders", []))
    attributes = {
        key: getattr(app, key, _MISSING)
        for key in ("_langgraph_state", "start_worker", "stop_worker")
    }
    existing_security = getattr(attributes["_langgraph_state"], "_route_security", None)
    if existing_security is not None:
        requested = (identity_provider, require_instance_owner, respond_authorizer)
        if any(
            value is not None and value != previous
            for value, previous in zip(requested, existing_security)
        ):
            raise ValueError("Route security cannot change on an existing app")
        identity_provider, require_instance_owner, respond_authorizer = (
            existing_security
        )
    try:
        return _register_function_routes(
            app,
            registrations,
            sdk_or_options,
            start_worker=start_worker,
            identity_provider=identity_provider,
            require_instance_owner=require_instance_owner,
            respond_authorizer=respond_authorizer,
        )
    except BaseException:
        new_state = getattr(app, "_langgraph_state", None)
        if new_state is not None and new_state is not attributes["_langgraph_state"]:
            new_state.stop()
        app._function_builders[:] = builders
        for key, value in attributes.items():
            if value is _MISSING:
                if hasattr(app, key):
                    delattr(app, key)
            else:
                setattr(app, key, value)
        raise


def _register_function_routes(
    app: Any,
    registrations: Registration | Iterable[Registration] | Mapping[str, Registration],
    sdk_or_options: Any = None,
    *,
    start_worker: bool = False,
    identity_provider: Callable[[Any], str | None] | None = None,
    require_instance_owner: bool | None = None,
    respond_authorizer: Callable[..., bool] | None = None,
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
        start_worker: False by default; True only for local/Dedicated Always-On.
        identity_provider: Callback validating a token and returning its principal ID.
            Otherwise identity is read only behind Azure EasyAuth
            (`WEBSITE_AUTH_ENABLED=true`); raw client principal headers are untrusted.
        require_instance_owner: Ownership defaults to True, including ANONYMOUS
            behind EasyAuth or identity_provider. ANONYMOUS without an identity source
            requires explicit False for unauthenticated local/shared-key access.
        respond_authorizer: Optional (caller, instance_id[, owner]) policy for
            authenticated approvers; None requires the caller to own the instance.

    Returns:
        The supplied function app with routes registered.
    """
    func_mod = _require_azure_functions()
    app_auth = getattr(app, "_auth_level", None)
    if (
        app_auth == func_mod.AuthLevel.ANONYMOUS
        and require_instance_owner is None
        and identity_provider is None
        and not _easyauth_enabled()
    ):
        raise ValueError(
            "AuthLevel.ANONYMOUS requires explicit require_instance_owner=False "
            "or an identity_provider / WEBSITE_AUTH_ENABLED=true"
        )
    enforce_owner = True if require_instance_owner is None else require_instance_owner
    respond_arity = 3
    if respond_authorizer is not None:
        signature = inspect.signature(respond_authorizer)
        try:
            signature.bind("caller", "instance", "owner")
        except TypeError:
            try:
                signature.bind("caller", "instance")
            except TypeError as exc:
                raise ValueError(
                    "respond_authorizer must accept 2 or 3 arguments"
                ) from exc
            respond_arity = 2
    regs = _normalize_registrations(registrations)
    if not regs:
        raise ValueError("Provide at least one graph registration")
    state: _FunctionAppState | None = getattr(app, "_langgraph_state", None)
    if state is not None:
        state.add_registrations(
            regs, sdk_or_options=sdk_or_options, start_worker=start_worker
        )
        return app

    state = _FunctionAppState(sdk_or_options, start_worker=start_worker)
    state._route_security = (identity_provider, enforce_owner, respond_authorizer)
    state.add_registrations(
        regs, sdk_or_options=sdk_or_options, start_worker=start_worker
    )
    setattr(app, "_langgraph_state", state)
    setattr(app, "start_worker", state.ensure_worker)
    setattr(app, "stop_worker", state.stop)

    def _check_auth(
        req: Any,
        instance_id: str | None = None,
        *,
        allow_new: bool = False,
        responding: bool = False,
    ) -> tuple[str | None, Any]:
        custom_respond = responding and respond_authorizer is not None
        needs_identity = enforce_owner or custom_respond
        caller = None
        if needs_identity:
            try:
                caller = (
                    identity_provider(req)
                    if identity_provider is not None
                    else _extract_caller_principal(req)
                )
            except Exception:
                caller = None
        if needs_identity and (not isinstance(caller, str) or not caller):
            return None, _json_response(
                {"error": "Authentication required"}, status_code=401
            )
        if needs_identity and instance_id is not None:
            try:
                sdk = state.get_sdk()
                expected_owner = _resolve_durable_owner(state, sdk, str(instance_id))
            except Exception:
                return None, _json_response(
                    {"error": "Cannot verify instance ownership"}, status_code=403
                )
            if custom_respond:
                allowed = False
                if expected_owner is not None and respond_authorizer is not None:
                    try:
                        args = (caller, instance_id, expected_owner)[:respond_arity]
                        allowed = respond_authorizer(*args) is True
                    except Exception:
                        allowed = False
                if not allowed:
                    return None, _json_response(
                        {"error": "Caller is not authorized to respond"},
                        status_code=403,
                    )
            elif (expected_owner is None and not allow_new) or (
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
        if reg is None:
            return _json_response({"error": f"Unknown graph {name!r}"}, status_code=404)

        body, _ = _read_json_body(req)

        params = dict(req.params) if getattr(req, "params", None) else {}
        controls = (
            {**params, **body}
            if isinstance(body, Mapping) and "input" in body
            else params
        )
        if isinstance(body, Mapping) and "input" in body:
            inp = body["input"]
        else:
            inp = body
        instance_id = controls.get("instance_id", controls.get("instanceId"))
        if instance_id is not None:
            _validate_instance_id(instance_id)
        version = controls.get("version")
        if version is not None:
            from langchain_azure_compute.graph import RunError

            try:
                reg._resolve_graph(str(version))
            except RunError as exc:
                raise ValueError("Unknown graph version") from exc
        reuse_existing = _parse_bool(controls.get("reuse_existing", True))
        if "wait" in controls:
            _parse_bool(controls["wait"])
        if "timeout" in controls:
            _parse_int(controls["timeout"], "timeout")
        max_concurrency = controls.get("max_concurrency")
        node_max_concurrency = controls.get("node_max_concurrency")

        caller, err_resp = _check_auth(
            req, str(instance_id) if instance_id is not None else None, allow_new=True
        )
        if err_resp is not None:
            return err_resp

        eff_mc: int | None = None
        if max_concurrency is not None:
            req_mc = _parse_int(max_concurrency, "max_concurrency")
            reg_mc = reg.tool_config.max_concurrency
            eff_mc = min(req_mc, reg_mc) if reg_mc is not None else req_mc
        eff_nmc: int | None = None
        if node_max_concurrency is not None:
            req_nmc = _parse_int(node_max_concurrency, "node_max_concurrency")
            reg_nmc = reg.node_config.max_concurrency
            eff_nmc = min(req_nmc, reg_nmc) if reg_nmc is not None else req_nmc

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
        if enforce_owner and caller is not None:
            durable_owner = _resolve_durable_owner(
                state, sdk, run.instance_id, bypass_cache=True
            )
            if durable_owner != caller:
                raise PermissionError("Cannot verify instance ownership")
        state.thread_by_instance[run.instance_id] = run.thread_id
        api_base = _extract_api_base(req.url)
        quoted_id = urllib.parse.quote(run.instance_id, safe="")
        return _json_response(
            {
                "instanceId": run.instance_id,
                "threadId": run.thread_id,
                "statusQueryGetUri": (f"{api_base}/graphs/{name}/status/{quoted_id}"),
                "respondUri": (
                    f"{api_base}/graphs/{name}/respond/{quoted_id}/{{interruptId}}"
                ),
            },
            status_code=202,
        )

    def _handle_status(req: Any) -> Any:
        name = req.route_params.get("name", "")
        instance_id = req.route_params.get("instance_id", "")
        _validate_instance_id(instance_id)
        reg = state.registrations.get(name)
        if reg is None:
            return _json_response({"error": f"Unknown graph {name!r}"}, status_code=404)
        _, err_resp = _check_auth(req, instance_id)
        if err_resp is not None:
            return err_resp

        sdk = state.get_sdk()
        run = reg.client(sdk).attach_run(instance_id)
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
                    f"{urllib.parse.quote(instance_id, safe='')}/"
                    f"{urllib.parse.quote(intr.interrupt_id, safe='')}"
                ),
            }
            for intr in interrupts
        ]
        output: Any = None
        error_msg: str | None = None
        if runtime_status == "COMPLETED":
            output = run.wait_result(timeout=5)
        elif runtime_status == "FAILED":
            error_msg = "Workflow execution failed"

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
        _validate_instance_id(instance_id)
        route_interrupt_id = req.route_params.get("interrupt_id")
        reg = state.registrations.get(name)
        if reg is None:
            return _json_response({"error": f"Unknown graph {name!r}"}, status_code=404)
        caller, err_resp = _check_auth(req, instance_id, responding=True)
        if err_resp is not None:
            return err_resp

        body, has_body = _read_json_body(req)

        params = dict(req.params) if getattr(req, "params", None) else {}
        if any(k in params for k in ("values", "as_node", "asNode", "version")):
            return _json_response(
                {"error": "State and version overrides are not permitted on respond"},
                status_code=400,
            )

        if isinstance(body, Mapping) and "value" in body:
            value = body["value"]
            body_interrupt_id = body.get("interrupt_id") or body.get("interruptId")
        else:
            value = body
            body_interrupt_id = None

        eff_interrupt_id = (
            route_interrupt_id
            or body_interrupt_id
            or params.get("interrupt_id")
            or params.get("interruptId")
        )
        if (
            _has_reserved_resume_keys(value)
            or (
                isinstance(body, Mapping)
                and "value" in body
                and any(
                    k in body
                    for k in ("kind", "values", "as_node", "asNode", "version")
                )
            )
            or (
                eff_interrupt_id is None
                and isinstance(body, Mapping)
                and "value" not in body
                and any(k in body for k in ("kind", "values", "as_node", "asNode"))
            )
        ):
            raise ValueError("Internal orchestration envelopes are not permitted")
        if caller is not None and isinstance(value, dict):
            value = {**value, "responder": caller, "reviewer": caller}

        sdk = state.get_sdk()
        run = reg.client(sdk).attach_run(instance_id)
        if run.status() == "NOT_FOUND":
            return _json_response(
                {
                    "instanceId": instance_id,
                    "error": f"Instance {instance_id!r} not found",
                },
                status_code=404,
            )

        try:
            if eff_interrupt_id is not None:
                run.resume(str(eff_interrupt_id), value)
            elif has_body:
                run.resume(value)
            else:
                run.resume()
        except RunError:
            return _json_response(
                {"instanceId": instance_id, "error": "No pending interrupt to resume"},
                status_code=409,
            )
        except ValueError:
            return _json_response(
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

    def _safe_handler(handler: Callable[[Any], Any], req: Any) -> Any:
        from langchain_azure_compute.graph import RunError

        try:
            return handler(req)
        except OverflowError:
            return _json_response({"error": "Request body too large"}, status_code=413)
        except PermissionError:
            return _json_response(
                {"error": "Cannot verify instance ownership"}, status_code=403
            )
        except ValueError:
            # Validation messages are generated above; SDK exceptions are not echoed.
            return _json_response(
                {"error": "Invalid request parameters or JSON body"}, status_code=400
            )
        except RunError:
            return _json_response(
                {"error": "Run conflicts with existing state"}, status_code=409
            )
        except Exception:
            return _json_response({"error": "Internal server error"}, status_code=500)

    @app.route(route="graphs/{name}/start", methods=["POST"])
    def langgraph_start(req: Any) -> Any:
        return _safe_handler(_handle_start, req)

    @app.route(route="workflow/{name}/run", methods=["POST"])
    def langgraph_workflow_run(req: Any) -> Any:
        return _safe_handler(_handle_start, req)

    @app.route(route="graphs/{name}/status/{instance_id}", methods=["GET"])
    def langgraph_status(req: Any) -> Any:
        return _safe_handler(_handle_status, req)

    @app.route(route="workflow/{name}/status/{instance_id}", methods=["GET"])
    def langgraph_workflow_status(req: Any) -> Any:
        return _safe_handler(_handle_status, req)

    @app.route(route="graphs/{name}/respond/{instance_id}", methods=["POST"])
    def langgraph_respond(req: Any) -> Any:
        return _safe_handler(_handle_respond, req)

    @app.route(
        route="graphs/{name}/respond/{instance_id}/{interrupt_id}", methods=["POST"]
    )
    def langgraph_respond_interrupt(req: Any) -> Any:
        return _safe_handler(_handle_respond, req)

    @app.route(route="workflow/{name}/respond/{instance_id}", methods=["POST"])
    def langgraph_workflow_respond(req: Any) -> Any:
        return _safe_handler(_handle_respond, req)

    @app.route(
        route="workflow/{name}/respond/{instance_id}/{interrupt_id}", methods=["POST"]
    )
    def langgraph_workflow_respond_interrupt(req: Any) -> Any:
        return _safe_handler(_handle_respond, req)

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
    respond_authorizer: Callable[..., bool] | None = None,
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
        start_worker: False (thin routes) by default; True only for local development
            or Dedicated Always-On hosting, never Consumption/Flex.
        identity_provider: Verified-token callback returning a principal ID.
            Without it, ownership requires Azure EasyAuth; Function keys alone do not
            identify users. Never enable EasyAuth trust on a directly exposed host.
        require_instance_owner: Enforce per-user isolation, also with ANONYMOUS auth
            behind EasyAuth/identity_provider. Anonymous local access needs explicit
            False; omission without an identity source raises ValueError.
            Set False only for intentionally shared, platform-key-authorized access.
            The local Python server does not validate Function keys.
        respond_authorizer: Optional (caller, instance_id[, owner]) approver policy.
            None requires the owner. Verified callers are stamped into dict resumes
            as responder and reviewer, replacing untrusted request fields.

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
    register_function_routes(
        app,
        norm,
        dts_options,
        start_worker=start_worker,
        identity_provider=identity_provider,
        require_instance_owner=require_instance_owner,
        respond_authorizer=respond_authorizer,
    )

    def _serve(*, host: str = "127.0.0.1", port: int = 7071) -> None:
        app._langgraph_state.ensure_worker()
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
    start_worker: bool = False,
    identity_provider: Callable[[Any], str | None] | None = None,
    require_instance_owner: bool | None = None,
    respond_authorizer: Callable[..., bool] | None = None,
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
        start_worker: False by default; True only for local/Dedicated Always-On.
        identity_provider: Callback validating caller tokens; see create_function_app.
        require_instance_owner: Whether to enforce per-user ownership.
        respond_authorizer: Optional authenticated approver policy; see
            create_function_app.

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
        identity_provider=identity_provider,
        require_instance_owner=require_instance_owner,
        respond_authorizer=respond_authorizer,
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
                az_resp = None
                try:
                    length = _parse_int(
                        self.headers.get("Content-Length", "0") or "0",
                        "Content-Length",
                        minimum=0,
                    )
                    if length > MAX_HTTP_BODY_BYTES:
                        az_resp = _json_response(
                            {"error": "Request body too large"}, status_code=413
                        )
                except (ValueError, TypeError):
                    az_resp = _json_response(
                        {"error": "Invalid Content-Length"}, status_code=400
                    )
                if az_resp is None:
                    raw_body = self.rfile.read(length) if length > 0 else b""
                    az_req = func.HttpRequest(
                        method=self.command,
                        url=f"{server_ref.base_url}{self.path}",
                        headers={k: v for k, v in self.headers.items()},
                        params={},
                        route_params={},
                        body=raw_body,
                    )
                    try:
                        az_resp = dispatch_function_request(app_ref, az_req)
                    except Exception:
                        az_resp = _json_response(
                            {"error": "Internal server error"}, status_code=500
                        )

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
