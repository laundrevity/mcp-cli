from __future__ import annotations

import asyncio
from typing import Any


class TransportClosed(Exception):
    """Raised when an in-memory transport endpoint is closed."""


_CLOSE_SENTINEL = object()


class TransportEndpoint:
    def __init__(self, incoming: asyncio.Queue, outgoing: asyncio.Queue):
        self._incoming = incoming
        self._outgoing = outgoing

    async def send(self, message: Any) -> None:
        await self._outgoing.put(message)

    async def receive(self) -> Any:
        message = await self._incoming.get()
        if message is _CLOSE_SENTINEL:
            raise TransportClosed
        return message

    async def close(self) -> None:
        await self._outgoing.put(_CLOSE_SENTINEL)

    async def stop(self) -> None:
        await self._incoming.put(_CLOSE_SENTINEL)


class InMemoryTransport:
    """Bidirectional, asyncio-based transport for exercising client/server flows."""

    def __init__(self) -> None:
        self._client_to_server: asyncio.Queue = asyncio.Queue()
        self._server_to_client: asyncio.Queue = asyncio.Queue()

    def client_endpoint(self) -> TransportEndpoint:
        return TransportEndpoint(self._server_to_client, self._client_to_server)

    def server_endpoint(self) -> TransportEndpoint:
        return TransportEndpoint(self._client_to_server, self._server_to_client)
