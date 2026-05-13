# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Async HTTP operations for the Foundry checkpoint storage REST API.

This module is a self-contained, vendored copy of the operations originally
provided by ``azure-ai-agentserver-core``. It does not depend on that
package.
"""

from __future__ import annotations

import json
from abc import ABC
from typing import Any, ClassVar, Dict, List, MutableMapping, Optional, Type

from azure.core import AsyncPipelineClient
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ResourceNotModifiedError,
    map_error,
)
from azure.core.pipeline.transport import AsyncHttpResponse, HttpRequest
from azure.core.tracing.decorator_async import distributed_trace_async

from ._models import (
    CheckpointItem,
    CheckpointItemId,
    CheckpointItemRequest,
    CheckpointItemResponse,
    CheckpointSession,
    CheckpointSessionRequest,
    CheckpointSessionResponse,
    ListCheckpointItemIdsResponse,
)

# API version pinned to the Foundry checkpoint preview surface.
API_VERSION: str = "2025-11-15-preview"

ErrorMapping = MutableMapping[int, Type[HttpResponseError]]


class _BaseOperations(ABC):
    """Common HTTP send/parse helpers for checkpoint operations."""

    DEFAULT_ERROR_MAP: ClassVar[ErrorMapping] = {
        401: ClientAuthenticationError,
        404: ResourceNotFoundError,
        409: ResourceExistsError,
        304: ResourceNotModifiedError,
    }

    def __init__(
        self,
        client: AsyncPipelineClient,
        error_map: Optional[ErrorMapping] = None,
    ) -> None:
        self._client = client
        self._error_map = self._prepare_error_map(error_map)

    @classmethod
    def _prepare_error_map(
        cls, custom_error_map: Optional[ErrorMapping] = None
    ) -> MutableMapping[int, Type[HttpResponseError]]:
        error_map: MutableMapping[int, Type[HttpResponseError]] = dict(
            cls.DEFAULT_ERROR_MAP
        )
        if custom_error_map:
            error_map.update(custom_error_map)
        return error_map

    async def _send_request(
        self, request: HttpRequest, *, stream: bool = False, **kwargs: Any
    ) -> AsyncHttpResponse:
        response: AsyncHttpResponse = await self._client.send_request(
            request, stream=stream, **kwargs
        )
        self._handle_response_error(response)
        return response

    def _handle_response_error(self, response: AsyncHttpResponse) -> None:
        if response.status_code != 200:
            map_error(
                status_code=response.status_code,
                response=response,
                error_map=self._error_map,
            )
            raise HttpResponseError(response=response)

    def _extract_response_json(self, response: AsyncHttpResponse) -> Any:
        try:
            payload_text = response.text()
            return json.loads(payload_text) if payload_text else {}
        except AttributeError:
            payload_bytes = response.body()
            return json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}


class CheckpointSessionOperations(_BaseOperations):
    """Operations for managing checkpoint sessions."""

    _HEADERS: ClassVar[Dict[str, str]] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    _QUERY_PARAMS: ClassVar[Dict[str, Any]] = {"api-version": API_VERSION}

    def _session_path(self, session_id: Optional[str] = None) -> str:
        base = "/checkpoints/sessions"
        return f"{base}/{session_id}" if session_id else base

    def _build_upsert_request(self, session: CheckpointSession) -> HttpRequest:
        request_model = CheckpointSessionRequest.from_session(session)
        return self._client.put(
            self._session_path(session.session_id),
            params=self._QUERY_PARAMS,
            headers=self._HEADERS,
            content=request_model.model_dump(by_alias=True),
        )

    def _build_read_request(self, session_id: str) -> HttpRequest:
        return self._client.get(
            self._session_path(session_id),
            params=self._QUERY_PARAMS,
            headers=self._HEADERS,
        )

    def _build_delete_request(self, session_id: str) -> HttpRequest:
        return self._client.delete(
            self._session_path(session_id),
            params=self._QUERY_PARAMS,
            headers=self._HEADERS,
        )

    @distributed_trace_async
    async def upsert(self, session: CheckpointSession) -> CheckpointSession:
        """Create or update a checkpoint session."""
        request = self._build_upsert_request(session)
        response = await self._send_request(request)
        async with response:
            json_response = self._extract_response_json(response)
            session_response = CheckpointSessionResponse.model_validate(json_response)
        return session_response.to_session()

    @distributed_trace_async
    async def read(self, session_id: str) -> Optional[CheckpointSession]:
        """Read a checkpoint session by ID. Returns ``None`` if not found."""
        request = self._build_read_request(session_id)
        try:
            response = await self._send_request(request)
            async with response:
                json_response = self._extract_response_json(response)
                session_response = CheckpointSessionResponse.model_validate(
                    json_response
                )
            return session_response.to_session()
        except ResourceNotFoundError:
            return None

    @distributed_trace_async
    async def delete(self, session_id: str) -> None:
        """Delete a checkpoint session."""
        request = self._build_delete_request(session_id)
        response = await self._send_request(request)
        async with response:
            pass


class CheckpointItemOperations(_BaseOperations):
    """Operations for managing checkpoint items."""

    _HEADERS: ClassVar[Dict[str, str]] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    _QUERY_PARAMS: ClassVar[Dict[str, Any]] = {"api-version": API_VERSION}

    def _items_path(self, item_id: Optional[str] = None) -> str:
        base = "/checkpoints/items"
        return f"{base}/{item_id}" if item_id else base

    def _build_create_batch_request(
        self, items: List[CheckpointItem]
    ) -> HttpRequest:
        request_models = [CheckpointItemRequest.from_item(item) for item in items]
        return self._client.post(
            self._items_path(),
            params=self._QUERY_PARAMS,
            headers=self._HEADERS,
            content=[model.model_dump(by_alias=True) for model in request_models],
        )

    def _build_read_request(self, item_id: CheckpointItemId) -> HttpRequest:
        params = dict(self._QUERY_PARAMS)
        params["sessionId"] = item_id.session_id
        if item_id.parent_id:
            params["parentId"] = item_id.parent_id
        return self._client.get(
            self._items_path(item_id.item_id),
            params=params,
            headers=self._HEADERS,
        )

    def _build_delete_request(self, item_id: CheckpointItemId) -> HttpRequest:
        params = dict(self._QUERY_PARAMS)
        params["sessionId"] = item_id.session_id
        if item_id.parent_id:
            params["parentId"] = item_id.parent_id
        return self._client.delete(
            self._items_path(item_id.item_id),
            params=params,
            headers=self._HEADERS,
        )

    def _build_list_ids_request(
        self, session_id: str, parent_id: Optional[str] = None
    ) -> HttpRequest:
        params = dict(self._QUERY_PARAMS)
        params["sessionId"] = session_id
        if parent_id:
            params["parentId"] = parent_id
        return self._client.get(
            self._items_path(),
            params=params,
            headers=self._HEADERS,
        )

    @distributed_trace_async
    async def create_batch(
        self, items: List[CheckpointItem]
    ) -> List[CheckpointItem]:
        """Create checkpoint items in batch."""
        if not items:
            return []

        request = self._build_create_batch_request(items)
        response = await self._send_request(request)
        async with response:
            json_response = self._extract_response_json(response)
            if isinstance(json_response, list):
                return [
                    CheckpointItemResponse.model_validate(item).to_item()
                    for item in json_response
                ]
            return [CheckpointItemResponse.model_validate(json_response).to_item()]

    @distributed_trace_async
    async def read(self, item_id: CheckpointItemId) -> Optional[CheckpointItem]:
        """Read a checkpoint item by ID. Returns ``None`` if not found."""
        request = self._build_read_request(item_id)
        try:
            response = await self._send_request(request)
            async with response:
                json_response = self._extract_response_json(response)
                item_response = CheckpointItemResponse.model_validate(json_response)
            return item_response.to_item()
        except ResourceNotFoundError:
            return None

    @distributed_trace_async
    async def delete(self, item_id: CheckpointItemId) -> bool:
        """Delete a checkpoint item. Returns ``False`` if not found."""
        request = self._build_delete_request(item_id)
        try:
            response = await self._send_request(request)
            async with response:
                pass
            return True
        except ResourceNotFoundError:
            return False

    @distributed_trace_async
    async def list_ids(
        self, session_id: str, parent_id: Optional[str] = None
    ) -> List[CheckpointItemId]:
        """List checkpoint item IDs for a session."""
        request = self._build_list_ids_request(session_id, parent_id)
        response = await self._send_request(request)
        async with response:
            json_response = self._extract_response_json(response)
            list_response = ListCheckpointItemIdsResponse.model_validate(json_response)
        return [item.to_item_id() for item in list_response.value]
