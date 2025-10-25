from __future__ import annotations

from typing import Optional

from .models import ClientCapabilities, HandshakeResult, ServerCapabilities
from .transport import TransportClosed, TransportEndpoint


class AsyncMCPClient:
    """Lightweight async MCP client used for exercising the handshake flow."""

    def __init__(self, capabilities: ClientCapabilities) -> None:
        self.capabilities = capabilities
        self._endpoint: Optional[TransportEndpoint] = None

    async def connect(self, endpoint: TransportEndpoint) -> None:
        self._endpoint = endpoint

    async def initialize(self) -> HandshakeResult:
        if self._endpoint is None:
            raise RuntimeError("Client must connect before initialization.")

        await self._endpoint.send(
            {
                "type": "initialize",
                "client_capabilities": self.capabilities.to_payload(),
            }
        )

        response = await self._endpoint.receive()
        if not isinstance(response, dict):
            raise RuntimeError("Server returned an unexpected payload.")
        if response.get("type") != "initialize/ok":
            raise RuntimeError(f"Server rejected handshake: {response!r}")

        server_caps = ServerCapabilities.from_payload(
            response.get("server_capabilities", {})
        )

        return HandshakeResult(
            client_capabilities=self.capabilities,
            server_capabilities=server_caps,
        )

    async def close(self) -> None:
        if self._endpoint is None:
            return
        try:
            await self._endpoint.send({"type": "client/shutdown"})
        except TransportClosed:
            pass
        finally:
            await self._endpoint.close()
            self._endpoint = None
