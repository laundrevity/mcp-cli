from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from . import telemetry
from .models import (
    ClientCapabilities,
    ClientInfo,
    ContentBlock,
    PromptDefinition,
    PromptRenderResult,
    ResourceContent,
    ResourceDescriptor,
    ResourceTemplate,
    RootDescriptor,
    ElicitationResponse,
    SamplingMessage,
    SamplingRequest,
    SamplingResponse,
    ServerCapabilities,
    ServerInfo,
    ToolCallResult,
    ToolDefinition,
)
from .transport import TransportClosed, TransportEndpoint

ToolHandler = Callable[[Dict[str, Any]], Awaitable[ToolCallResult] | ToolCallResult]
PromptHandler = Callable[[Dict[str, Any]], Awaitable[PromptRenderResult] | PromptRenderResult]


class AsyncMCPServer:
    """Minimal asynchronous MCP server for testing capability negotiation."""

    _LOG_LEVELS: Dict[str, int] = {
        "debug": 10,
        "info": 20,
        "notice": 25,
        "warning": 30,
        "error": 40,
        "critical": 50,
        "alert": 60,
        "emergency": 70,
    }

    _LOGGER_METHODS: Dict[str, str] = {
        "debug": "debug",
        "info": "info",
        "notice": "info",
        "warning": "warning",
        "error": "error",
        "critical": "critical",
        "alert": "critical",
        "emergency": "critical",
    }

    def __init__(
        self,
        capabilities: ServerCapabilities,
        server_info: ServerInfo,
        *,
        protocol_version: str = "2025-06-18",
        instructions: str | None = None,
        tools: Optional[List[Tuple[ToolDefinition, ToolHandler]]] = None,
        resources: Optional[
            List[Tuple[ResourceDescriptor, ResourceContent]]
        ] = None,
        prompts: Optional[List[Tuple[PromptDefinition, PromptHandler]]] = None,
        resource_templates: Optional[List[ResourceTemplate]] = None,
    ) -> None:
        self.capabilities = capabilities
        self.server_info = server_info
        self.protocol_version = protocol_version
        self.instructions = instructions

        self.last_seen_client_capabilities: Optional[ClientCapabilities] = None
        self.last_seen_client_info: Optional[ClientInfo] = None
        self.received_messages: List[dict] = []

        self._endpoint: Optional[TransportEndpoint] = None
        self._running = False
        self._logger = logging.getLogger("mcp_cli.server")
        self._tools: Dict[str, Tuple[ToolDefinition, ToolHandler]] = {}
        self._resources: Dict[str, Tuple[ResourceDescriptor, ResourceContent]] = {}
        self._prompts: Dict[str, Tuple[PromptDefinition, PromptHandler]] = {}
        self._resource_templates: List[ResourceTemplate] = []
        self._resource_subscribers: Dict[str, bool] = {}
        self._request_counter = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._log_level: str = "info"

        if tools:
            for definition, handler in tools:
                self.register_tool(definition, handler)
        if resources:
            for descriptor, content in resources:
                self.register_resource(descriptor, content)
        if prompts:
            for definition, handler in prompts:
                self.register_prompt(definition, handler)
        if resource_templates:
            for template in resource_templates:
                self.register_resource_template(template)

        self._logger.debug(
            "Server instantiated with protocol=%s capabilities=%s",
            protocol_version,
            capabilities.to_payload(),
        )

    async def serve(self, endpoint: TransportEndpoint) -> None:
        self._endpoint = endpoint
        self._running = True
        self._logger.debug("Server entering serve loop.")

        try:
            while self._running:
                try:
                    message = await endpoint.receive()
                except TransportClosed:
                    self._logger.debug("Transport closed; stopping server loop.")
                    break

                if not isinstance(message, dict):
                    self._logger.warning("Ignoring non-dict message: %s", message)
                    continue

                self.received_messages.append(message)
                method = message.get("method")
                message_id = message.get("id")
                self._logger.debug("Received message method=%s payload=%s", method, message)
                telemetry.record_event(
                    role="server",
                    direction="incoming",
                    payload=message,
                    channel=method or "response",
                )

                if method:
                    await self._handle_request_message(method, message)
                    continue

                if message_id is not None:
                    pending = self._pending_requests.pop(message_id, None)
                    if pending is not None and not pending.done():
                        pending.set_result(message)
                    else:
                        self._logger.debug(
                            "Received unmatched response id=%s payload=%s",
                            message_id,
                            message,
                        )
        finally:
            self._running = False
            self._logger.debug("Server loop terminated.")

    async def shutdown(self) -> None:
        self._running = False
        self._logger.debug("Shutdown requested.")
        if self._endpoint is not None:
            await self._endpoint.stop()
            self._logger.debug("Transport stop signal sent.")

    def register_tool(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        self._tools[definition.name] = (definition, handler)
        self._logger.debug("Registered tool %s", definition.name)

    def register_resource(
        self,
        descriptor: ResourceDescriptor,
        content: ResourceContent,
    ) -> None:
        self._resources[descriptor.uri] = (descriptor, content)
        self._logger.debug("Registered resource %s", descriptor.uri)

    def register_prompt(
        self,
        definition: PromptDefinition,
        handler: PromptHandler,
    ) -> None:
        self._prompts[definition.name] = (definition, handler)
        self._logger.debug("Registered prompt %s", definition.name)

    def register_resource_template(self, template: ResourceTemplate) -> None:
        self._resource_templates.append(template)
        self._logger.debug("Registered resource template %s", template.name)

    async def request_sampling(
        self,
        *,
        messages: List[SamplingMessage],
        model_preferences: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> SamplingResponse:
        params = SamplingRequest(
            messages=messages,
            model_preferences=model_preferences,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        response = await self._send_request(
            "sampling/createMessage",
            params.to_payload(),
        )
        result = response.get("result", {})
        return SamplingResponse.from_payload(result)

    async def list_client_roots(self) -> List[RootDescriptor]:
        response = await self._send_request("roots/list", {})
        result = response.get("result", {})
        roots_payload = result.get("roots", [])
        roots = [RootDescriptor.from_payload(item) for item in roots_payload]
        self._logger.debug("Received %d root(s) from client.", len(roots))
        return roots

    async def request_elicitation(
        self,
        *,
        message: str,
        requested_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ElicitationResponse:
        params: Dict[str, Any] = {"message": message}
        if requested_schema is not None:
            params["requestedSchema"] = requested_schema
        if metadata is not None:
            params["metadata"] = metadata

        response = await self._send_request("elicitation/create", params)
        result = response.get("result", {})
        elicitation = ElicitationResponse.from_payload(result)
        await self.emit_log_message(
            "info",
            logger_name="server.elicitation",
            data={
                "event": "elicitation_completed",
                "action": elicitation.action,
            },
        )
        return elicitation

    async def _handle_request_message(self, method: str, message: Dict[str, Any]) -> None:
        if method == "initialize":
            await self._handle_initialize(message)
        elif method == "tools/list":
            await self._handle_tools_list(message)
        elif method == "tools/call":
            await self._handle_tools_call(message)
        elif method == "resources/list":
            await self._handle_resources_list(message)
        elif method == "resources/read":
            await self._handle_resources_read(message)
        elif method == "resources/subscribe":
            await self._handle_resources_subscribe(message)
        elif method == "resources/templates/list":
            await self._handle_resource_templates_list(message)
        elif method == "prompts/list":
            await self._handle_prompts_list(message)
        elif method == "prompts/get":
            await self._handle_prompts_get(message)
        elif method == "logging/setLevel":
            await self._handle_logging_set_level(message)
        elif method in {"notifications/shutdown", "client/shutdown"}:
            self._logger.debug("Shutdown notification received; exiting loop.")
            self._running = False
        else:
            if "id" in message:
                await self._send_error(
                    message.get("id"),
                    code=-32601,
                    message=f"Method '{method}' not found.",
                )

    async def _handle_initialize(self, message: Dict[str, Any]) -> None:
        params = message.get("params", {})
        if not isinstance(params, dict):
            self._logger.warning(
                "Initialize params not dict; defaulting to empty: %s", params
            )
            params = {}
        client_caps = ClientCapabilities.from_payload(
            params.get("capabilities", {})
        )
        self.last_seen_client_capabilities = client_caps
        self.last_seen_client_info = ClientInfo.from_payload(
            params.get("clientInfo", {})
        )
        self._logger.debug(
            "Client info=%s capabilities=%s",
            self.last_seen_client_info,
            client_caps,
        )

        response: dict = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": self.protocol_version,
                "capabilities": self.capabilities.to_payload(),
                "serverInfo": self.server_info.to_payload(),
            },
        }

        if self.instructions is not None:
            response["result"]["instructions"] = self.instructions

        self._logger.debug("Sending initialize response payload=%s", response)
        if self._endpoint is not None:
            await self._send_with_telemetry(response)

    async def _handle_tools_list(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to tools/list; endpoint is not set.")
            return
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "tools": [
                    definition.to_payload()
                    for definition, _ in self._tools.values()
                ],
            },
        }
        self._logger.debug("Responding to tools/list with %s", response)
        await self._send_with_telemetry(response)

    async def _handle_tools_call(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to tools/call; endpoint is not set.")
            return
        params = message.get("params", {})
        if not isinstance(params, dict):
            params = {}

        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        if name not in self._tools:
            await self._send_error(
                message.get("id"),
                code=-32001,
                message=f"Unknown tool: {name}",
            )
            return

        definition, handler = self._tools[name]
        self._logger.debug(
            "Invoking tool '%s' with arguments=%s", definition.name, arguments
        )

        try:
            result = handler(arguments)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, ToolCallResult):
                raise TypeError("Tool handler must return ToolCallResult.")
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Tool '%s' failed", definition.name)
            result = ToolCallResult(
                content=[
                    ContentBlock(type="text", text=f"Tool error: {exc}")
                ],
                is_error=True,
            )

        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": result.to_payload(),
        }
        self._logger.debug("Tool '%s' result payload=%s", definition.name, response)
        await self._send_with_telemetry(response)
        await self.emit_log_message(
            "info",
            logger_name="server.tools",
            data={
                "event": "tool_call",
                "tool": definition.name,
                "isError": result.is_error,
            },
        )

    async def _handle_resources_list(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to resources/list; endpoint is not set.")
            return
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "resources": [
                    descriptor.to_payload()
                    for descriptor, _ in self._resources.values()
                ],
            },
        }
        self._logger.debug("Responding to resources/list with %s", response)
        await self._send_with_telemetry(response)

    async def _handle_resources_read(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to resources/read; endpoint is not set.")
            return

        params = message.get("params", {})
        if not isinstance(params, dict):
            params = {}

        uri = params.get("uri")
        if uri not in self._resources:
            await self._send_error(
                message.get("id"),
                code=-32002,
                message=f"Resource not found: {uri}",
            )
            return

        descriptor, content = self._resources[uri]
        if not content.text and not content.blob:
            content = ResourceContent(
                uri=descriptor.uri,
                name=descriptor.name,
                title=descriptor.title,
                mime_type=descriptor.mime_type,
                text=descriptor.description or "",
                annotations=descriptor.annotations,
            )

        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "contents": [content.to_payload()],
            },
        }
        self._logger.debug("Responding to resources/read with %s", response)
        await self._send_with_telemetry(response)

    async def _handle_resources_subscribe(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to resources/subscribe; endpoint is not set.")
            return

        params = message.get("params", {})
        if not isinstance(params, dict):
            params = {}

        uri = params.get("uri")
        if uri not in self._resources:
            await self._send_error(
                message.get("id"),
                code=-32002,
                message=f"Resource not found: {uri}",
            )
            return

        self._resource_subscribers[uri] = True
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {"uri": uri},
        }
        self._logger.debug("Registered subscription for resource %s", uri)
        await self._send_with_telemetry(response)

    async def _handle_resource_templates_list(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning(
                "Cannot respond to resources/templates/list; endpoint is not set."
            )
            return

        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "resourceTemplates": [
                    template.to_payload() for template in self._resource_templates
                ],
            },
        }
        self._logger.debug(
            "Responding to resources/templates/list with %s",
            response,
        )
        await self._send_with_telemetry(response)

    async def _handle_prompts_list(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to prompts/list; endpoint is not set.")
            return
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "prompts": [
                    definition.to_payload()
                    for definition, _ in self._prompts.values()
                ],
            },
        }
        self._logger.debug("Responding to prompts/list with %s", response)
        await self._send_with_telemetry(response)

    async def _handle_prompts_get(self, message: dict) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot respond to prompts/get; endpoint is not set.")
            return
        params = message.get("params", {})
        if not isinstance(params, dict):
            params = {}

        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        if name not in self._prompts:
            await self._send_error(
                message.get("id"),
                code=-32602,
                message=f"Prompt not found: {name}",
            )
            return

        definition, handler = self._prompts[name]
        self._logger.debug("Rendering prompt '%s' with arguments=%s", name, arguments)

        try:
            rendered = handler(arguments)
            if inspect.isawaitable(rendered):
                rendered = await rendered
            if not isinstance(rendered, PromptRenderResult):
                raise TypeError("Prompt handler must return PromptRenderResult.")
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Prompt '%s' handler failed", name)
            await self._send_error(
                message.get("id"),
                code=-32099,
                message=str(exc),
            )
            return

        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": rendered.to_payload(),
        }
        self._logger.debug("Prompt '%s' render payload=%s", name, response)
        await self._send_with_telemetry(response)

    async def notify_resource_updated(
        self,
        uri: str,
        *,
        title: Optional[str] = None,
    ) -> None:
        if self._endpoint is None:
            return
        if not self._resource_subscribers.get(uri):
            self._logger.debug(
                "Skipping resource update for %s; no subscribers registered.",
                uri,
            )
            return

        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": {
                "uri": uri,
            },
        }
        if title is not None:
            notification["params"]["title"] = title

        self._logger.debug("Sending resource update notification: %s", notification)
        await self._send_with_telemetry(notification)
        await self.emit_log_message(
            "debug",
            logger_name="server.resources",
            data={"event": "resource_updated", "uri": uri, "title": title},
        )

    async def notify_resources_list_changed(self) -> None:
        await self._send_list_changed_notification("resources")

    async def notify_tools_list_changed(self) -> None:
        await self._send_list_changed_notification("tools")

    async def notify_prompts_list_changed(self) -> None:
        await self._send_list_changed_notification("prompts")

    async def _send_list_changed_notification(self, category: str) -> None:
        if self._endpoint is None:
            return
        allowed = {"resources", "tools", "prompts"}
        if category not in allowed:
            self._logger.warning("Unsupported list-changed category: %s", category)
            return
        notification = {
            "jsonrpc": "2.0",
            "method": f"notifications/{category}/list_changed",
        }
        self._logger.debug("Sending %s list_changed notification", category)
        await self._send_with_telemetry(notification)
        await self.emit_log_message(
            "notice",
            logger_name="server.%s" % category,
            data={"event": "list_changed", "category": category},
        )

    async def _send_error(self, request_id: Optional[int], *, code: int, message: str) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot send error; endpoint is not set.")
            return
        error_payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
        self._logger.debug("Sending error response: %s", error_payload)
        await self._send_with_telemetry(error_payload)

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._endpoint is None:
            raise RuntimeError("Server transport endpoint not ready.")

        request_id = self._next_request_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future
        self._logger.debug("Sending request id=%s method=%s params=%s", request_id, method, params)
        await self._send_with_telemetry(payload)
        response = await future

        if "error" in response:
            raise RuntimeError(response["error"])

        return response

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    async def _send_with_telemetry(self, payload: Dict[str, Any]) -> None:
        if self._endpoint is None:
            self._logger.warning("Cannot send payload; endpoint not set.")
            return

        channel: Optional[str] = None
        if "method" in payload:
            channel = str(payload.get("method"))
        elif "result" in payload:
            channel = "response"
        elif "error" in payload:
            channel = "error"

        telemetry.record_event(
            role="server",
            direction="outgoing",
            payload=payload,
            channel=channel,
        )
        await self._endpoint.send(payload)

    async def _handle_logging_set_level(self, message: Dict[str, Any]) -> None:
        params = message.get("params", {})
        if not isinstance(params, dict):
            await self._send_error(
                message.get("id"),
                code=-32602,
                message="Invalid params for logging/setLevel.",
            )
            return

        level_value = params.get("level")
        if not isinstance(level_value, str):
            await self._send_error(
                message.get("id"),
                code=-32602,
                message="Log level must be a string.",
            )
            return

        normalized = level_value.lower()
        if normalized not in self._LOG_LEVELS:
            await self._send_error(
                message.get("id"),
                code=-32602,
                message=f"Unsupported log level: {level_value}",
            )
            return

        self._log_level = normalized
        self._logger.info("Client set minimum log level to %s", normalized)

        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {},
        }
        await self._send_with_telemetry(response)
        await self.emit_log_message(
            "notice",
            logger_name="server.logging",
            data={"event": "log_level_set", "level": normalized},
        )

    async def emit_log_message(
        self,
        level: str,
        *,
        logger_name: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._endpoint is None:
            return

        normalized = level.lower()
        if normalized not in self._LOG_LEVELS:
            normalized = "info"

        current_threshold = self._LOG_LEVELS.get(self._log_level, 20)
        message_level_value = self._LOG_LEVELS[normalized]
        if message_level_value < current_threshold:
            return

        logger_method_name = self._LOGGER_METHODS.get(normalized, "info")
        logger_method = getattr(self._logger, logger_method_name, self._logger.info)
        logger_method(
            "Emitting log notification level=%s logger=%s data=%s",
            normalized,
            logger_name,
            data,
        )

        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": normalized,
            },
        }
        if logger_name is not None:
            payload["params"]["logger"] = logger_name
        if data:
            payload["params"]["data"] = data

        await self._send_with_telemetry(payload)
