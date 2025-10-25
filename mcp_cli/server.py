from __future__ import annotations

from typing import Optional

from .models import ClientCapabilities, ServerCapabilities
from .transport import TransportClosed, TransportEndpoint


class AsyncMCPServer:
    """Minimal asynchronous MCP server for testing capability negotiation."""

    def __init__(self, capabilities: ServerCapabilities) -> None:
        self.capabilities = capabilities
        self.last_seen_client_capabilities: Optional[ClientCapabilities] = None
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

                message_type = message.get("type")
                if message_type == "initialize":
                    payload = message.get("client_capabilities", {})
                    client_caps = ClientCapabilities.from_payload(payload)
                    self.last_seen_client_capabilities = client_caps
                    await endpoint.send(
                        {
                            "type": "initialize/ok",
                            "server_capabilities": self.capabilities.to_payload(),
                        }
                    )
                elif message_type in {"client/shutdown", "server/shutdown"}:
                    break
        finally:
            self._running = False

    async def shutdown(self) -> None:
        self._running = False
        if self._endpoint is not None:
            await self._endpoint.stop()
