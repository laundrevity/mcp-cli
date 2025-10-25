from __future__ import annotations

from typing import List, Optional

from .models import ClientCapabilities, ClientInfo, ServerCapabilities, ServerInfo
from .transport import TransportClosed, TransportEndpoint


class AsyncMCPServer:
    """Minimal asynchronous MCP server for testing capability negotiation."""

    def __init__(
        self,
        capabilities: ServerCapabilities,
        server_info: ServerInfo,
        *,
        protocol_version: str = "2025-06-18",
        instructions: str | None = None,
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

    async def serve(self, endpoint: TransportEndpoint) -> None:
        self._endpoint = endpoint
        self._running = True

        try:
            while self._running:
                try:
                    message = await endpoint.receive()
                except TransportClosed:
                    break

                if not isinstance(message, dict):
                    continue

                self.received_messages.append(message)
                method = message.get("method")

                if method == "initialize":
                    params = message.get("params", {})
                    if not isinstance(params, dict):
                        params = {}
                    client_caps = ClientCapabilities.from_payload(
                        params.get("capabilities", {})
                    )
                    self.last_seen_client_capabilities = client_caps
                    self.last_seen_client_info = ClientInfo.from_payload(
                        params.get("clientInfo", {})
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

                    await endpoint.send(response)

                elif method in {"notifications/shutdown", "client/shutdown"}:
                    break
        finally:
            self._running = False

    async def shutdown(self) -> None:
        self._running = False
        if self._endpoint is not None:
            await self._endpoint.stop()
