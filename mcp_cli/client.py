from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from . import telemetry
from .models import (
    ClientCapabilities,
    ClientInfo,
    ContentBlock,
    HandshakeResult,
    PromptDefinition,
    PromptRenderResult,
    ResourceContent,
    ResourceDescriptor,
    ResourceTemplate,
    SamplingRequest,
    SamplingResponse,
    ServerCapabilities,
    ServerInfo,
    ToolCallResult,
    ToolDefinition,
)
from .sampling import SamplingProvider
from .transport import TransportClosed, TransportEndpoint

RequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
NotificationHandler = Callable[[Dict[str, Any]], Awaitable[None] | None]


class AsyncMCPClient:
    """Async MCP client implementing the initialization handshake."""

    def __init__(
        self,
        capabilities: ClientCapabilities,
        client_info: ClientInfo,
        *,
        protocol_version: str = "2025-06-18",
    ) -> None:
        self.capabilities = capabilities
        self.client_info = client_info
        self.protocol_version = protocol_version
        self._endpoint: Optional[TransportEndpoint] = None
        self._request_counter = 0
        self._sampling_provider: Optional[SamplingProvider] = None
        self._request_handlers: Dict[str, RequestHandler] = {}
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger("mcp_cli.client")
        self._notification_handlers: Dict[str, NotificationHandler] = {}
        self.register_request_handler(
            "sampling/createMessage",
            self._handle_sampling_create_message,
        )
        self._logger.debug(
            "Client instantiated with protocol=%s capabilities=%s",
            protocol_version,
            capabilities.to_payload(),
        )

    async def connect(self, endpoint: TransportEndpoint) -> None:
        self._logger.debug("Connecting to transport endpoint.")
        self._endpoint = endpoint
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    def set_sampling_provider(self, provider: SamplingProvider) -> None:
        self._sampling_provider = provider
        self._logger.debug("Sampling provider configured: %s", provider)

    def register_request_handler(
        self,
        method: str,
        handler: RequestHandler,
    ) -> None:
        self._request_handlers[method] = handler

    def register_notification_handler(
        self,
        method: str,
        handler: NotificationHandler,
    ) -> None:
        self._notification_handlers[method] = handler

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    async def initialize(self) -> HandshakeResult:
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": self.capabilities.to_payload(),
                "clientInfo": self.client_info.to_payload(),
            },
        )

        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Server response missing result payload.")

        server_caps = ServerCapabilities.from_payload(result.get("capabilities", {}))
        server_info = ServerInfo.from_payload(result.get("serverInfo", {}))

        instructions = result.get("instructions")
        protocol_version = result.get("protocolVersion", self.protocol_version)

        await self._send_notification("notifications/initialized")

        handshake = HandshakeResult(
            client_capabilities=self.capabilities,
            server_capabilities=server_caps,
            protocol_version=protocol_version,
            request_id=response["id"],
            server_info=server_info,
            client_info=self.client_info,
            instructions=instructions,
        )

        self._logger.debug("Handshake result constructed: %s", handshake)
        return handshake

    async def list_tools(self) -> List[ToolDefinition]:
        response = await self._send_request("tools/list", {})
        result = response.get("result", {})
        tools_payload = result.get("tools", [])
        tools = [ToolDefinition.from_payload(item) for item in tools_payload]
        self._logger.debug("Received %d tools from server.", len(tools))
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolCallResult:
        response = await self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )
        result_payload = response.get("result", {})
        call_result = ToolCallResult.from_payload(result_payload)
        self._logger.debug(
            "Tool '%s' call returned is_error=%s",
            name,
            call_result.is_error,
        )
        return call_result

    async def list_resources(self) -> List[ResourceDescriptor]:
        response = await self._send_request("resources/list", {})
        result = response.get("result", {})
        resources_payload = result.get("resources", [])
        resources = [
            ResourceDescriptor.from_payload(item) for item in resources_payload
        ]
        self._logger.debug("Received %d resources from server.", len(resources))
        return resources

    async def read_resource(self, uri: str) -> List[ResourceContent]:
        response = await self._send_request(
            "resources/read",
            {"uri": uri},
        )
        result = response.get("result", {})
        contents_payload = result.get("contents", [])
        contents = [
            ResourceContent.from_payload(item) for item in contents_payload
        ]
        self._logger.debug(
            "Read resource %s with %d content blocks.",
            uri,
            len(contents),
        )
        return contents

    async def subscribe_resource(self, uri: str) -> Dict[str, Any]:
        response = await self._send_request(
            "resources/subscribe",
            {"uri": uri},
        )
        self._logger.debug("Subscribed to resource %s", uri)
        return response.get("result", {})

    async def list_resource_templates(self) -> List[ResourceTemplate]:
        response = await self._send_request("resources/templates/list", {})
        result = response.get("result", {})
        templates_payload = result.get("resourceTemplates", [])
        templates = [
            ResourceTemplate.from_payload(item) for item in templates_payload
        ]
        self._logger.debug(
            "Received %d resource template(s) from server.", len(templates)
        )
        return templates

    async def list_prompts(self) -> List[PromptDefinition]:
        response = await self._send_request("prompts/list", {})
        result = response.get("result", {})
        prompts_payload = result.get("prompts", [])
        prompts = [
            PromptDefinition.from_payload(item) for item in prompts_payload
        ]
        self._logger.debug("Received %d prompts from server.", len(prompts))
        return prompts

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> PromptRenderResult:
        response = await self._send_request(
            "prompts/get",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )
        result_payload = response.get("result", {})
        prompt_result = PromptRenderResult.from_payload(result_payload)
        self._logger.debug(
            "Prompt '%s' rendered with %d messages.",
            name,
            len(prompt_result.messages),
        )
        return prompt_result

    async def close(self) -> None:
        if self._endpoint is None:
            return
        try:
            shutdown_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/shutdown",
            }
            self._logger.debug("Sending shutdown notification payload=%s", shutdown_payload)
            telemetry.record_event(
                role="client",
                direction="outgoing",
                payload=shutdown_payload,
                channel="notifications/shutdown",
            )
            await self._endpoint.send(shutdown_payload)
        except TransportClosed:
            self._logger.debug("Transport already closed during shutdown.")
        finally:
            await self._endpoint.close()
            self._endpoint = None
            self._logger.debug("Transport endpoint released.")
            if self._listener_task is not None:
                self._listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._listener_task

    async def _listen(self) -> None:
        if self._endpoint is None:
            return
        try:
            while True:
                try:
                    message = await self._endpoint.receive()
                except TransportClosed:
                    self._logger.debug("Transport closed; listener exiting.")
                    break

                self._logger.debug("Listener received message: %s", message)
                if not isinstance(message, dict):
                    continue

                method = message.get("method")
                message_id = message.get("id")

                if method:
                    telemetry.record_event(
                        role="client",
                        direction="incoming",
                        payload=message,
                        channel=method,
                    )
                    if message_id is not None:
                        await self._handle_request(message)
                    else:
                        await self._handle_notification(method, message.get("params"))
                    continue

                if message_id is not None:
                    pending = self._pending_requests.pop(message_id, None)
                    if pending is not None and not pending.done():
                        telemetry.record_event(
                            role="client",
                            direction="incoming",
                            payload=message,
                            channel="response",
                        )
                        pending.set_result(message)
                    else:
                        self._logger.debug(
                            "Received unmatched response id=%s payload=%s",
                            message_id,
                            message,
                        )
        except asyncio.CancelledError:
            self._logger.debug("Listener task cancelled.")
            raise
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Listener encountered error: %s", exc)

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._endpoint is None:
            raise RuntimeError("Client is not connected to a server.")

        request_id = self._next_request_id()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        self._logger.debug("Sending request id=%s method=%s params=%s", request_id, method, params)
        telemetry.record_event(
            role="client",
            direction="outgoing",
            payload=request,
            channel=method,
        )
        await self._endpoint.send(request)

        response = await future
        if "error" in response:
            error = response["error"]
            raise RuntimeError(
                f"Server error for method '{method}': {error}"
            )
        return response

    async def _send_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._endpoint is None:
            raise RuntimeError("Client is not connected to a server.")
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params
        self._logger.debug("Sending notification method=%s params=%s", method, params)
        telemetry.record_event(
            role="client",
            direction="outgoing",
            payload=notification,
            channel=method,
        )
        await self._endpoint.send(notification)

    async def _handle_request(self, message: Dict[str, Any]) -> None:
        if self._endpoint is None:
            raise RuntimeError("Client is not connected to a server.")

        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        if not isinstance(params, dict):
            params = {}

        handler = self._request_handlers.get(method or "")
        if handler is None:
            await self._send_error_response(
                request_id,
                code=-32601,
                message=f"Method '{method}' not supported by client.",
            )
            return

        try:
            result_payload = await handler(params)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Error handling request method=%s id=%s", method, request_id
            )
            await self._send_error_response(
                request_id,
                code=-32099,
                message=str(exc),
            )
            return

        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result_payload,
        }
        self._logger.debug("Sending response for %s: %s", method, response)
        telemetry.record_event(
            role="client",
            direction="outgoing",
            payload=response,
            channel=method,
        )
        await self._endpoint.send(response)

    async def _handle_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]],
    ) -> None:
        handler = self._notification_handlers.get(method)
        if handler is None:
            self._logger.info("Received notification: %s", method)
            return

        payload = params if isinstance(params, dict) else {}
        result = handler(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _send_error_response(
        self,
        request_id: Optional[int],
        *,
        code: int,
        message: str,
    ) -> None:
        if self._endpoint is None:
            return
        error_payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
        self._logger.debug("Sending error response payload=%s", error_payload)
        telemetry.record_event(
            role="client",
            direction="outgoing",
            payload=error_payload,
            channel="error",
        )
        await self._endpoint.send(error_payload)

    async def _handle_sampling_create_message(
        self,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._logger.debug("Handling sampling/createMessage with params=%s", params)
        if self._sampling_provider is None:
            raise RuntimeError("No sampling provider configured.")

        sampling_request = SamplingRequest.from_payload(params)
        if not sampling_request.messages:
            raise RuntimeError("Sampling request requires at least one message.")

        sampling_response = await self._sampling_provider.create_message(
            sampling_request
        )

        if not sampling_response.content.text:
            sampling_response = SamplingResponse(
                role=sampling_response.role,
                content=ContentBlock(
                    type="text",
                    text=sampling_response.content.text or "",
                ),
                model=sampling_response.model,
                stop_reason=sampling_response.stop_reason,
            )

        return sampling_response.to_payload()
