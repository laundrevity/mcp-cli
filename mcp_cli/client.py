from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .models import (
    ClientCapabilities,
    ClientInfo,
    HandshakeResult,
    ServerCapabilities,
    ServerInfo,
)
from .transport import TransportClosed, TransportEndpoint


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
        self._logger = logging.getLogger("mcp_cli.client")
        self._logger.debug(
            "Client instantiated with protocol=%s capabilities=%s",
            protocol_version,
            capabilities.to_payload(),
        )

    async def connect(self, endpoint: TransportEndpoint) -> None:
        self._logger.debug("Connecting to transport endpoint.")
        self._endpoint = endpoint

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    async def initialize(self) -> HandshakeResult:
        if self._endpoint is None:
            raise RuntimeError("Client must connect before initialization.")

        request_id = self._next_request_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": self.capabilities.to_payload(),
                "clientInfo": self.client_info.to_payload(),
            },
        }

        self._logger.debug("Sending initialize request id=%s payload=%s", request_id, message)
        await self._endpoint.send(message)

        response = await self._endpoint.receive()
        self._logger.debug("Received response for id=%s payload=%s", request_id, response)

        if not isinstance(response, dict) or response.get("jsonrpc") != "2.0":
            raise RuntimeError("Server returned an unexpected payload.")
        if response.get("id") != request_id:
            raise RuntimeError("Server responded with unexpected request id.")
        if "error" in response:
            raise RuntimeError(f"Server rejected handshake: {response['error']}")

        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Server response missing result payload.")

        server_caps = ServerCapabilities.from_payload(result.get("capabilities", {}))
        server_info = ServerInfo.from_payload(result.get("serverInfo", {}))

        instructions = result.get("instructions")
        protocol_version = result.get("protocolVersion", self.protocol_version)

        initialized_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        self._logger.debug("Sending initialized notification payload=%s", initialized_payload)
        await self._endpoint.send(initialized_payload)

        # allow server to process notification before returning
        await asyncio.sleep(0)

        handshake = HandshakeResult(
            client_capabilities=self.capabilities,
            server_capabilities=server_caps,
            protocol_version=protocol_version,
            request_id=request_id,
            server_info=server_info,
            client_info=self.client_info,
            instructions=instructions,
        )

        self._logger.debug("Handshake result constructed: %s", handshake)
        return handshake

    async def close(self) -> None:
        if self._endpoint is None:
            return
        try:
            shutdown_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/shutdown",
            }
            self._logger.debug("Sending shutdown notification payload=%s", shutdown_payload)
            await self._endpoint.send(shutdown_payload)
        except TransportClosed:
            self._logger.debug("Transport already closed during shutdown.")
        finally:
            await self._endpoint.close()
            self._endpoint = None
            self._logger.debug("Transport endpoint released.")
