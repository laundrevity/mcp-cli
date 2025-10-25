import asyncio
import contextlib

import pytest

from mcp_cli.client import AsyncMCPClient, ClientCapabilities, HandshakeResult
from mcp_cli.server import AsyncMCPServer, ServerCapabilities
from mcp_cli.transport import InMemoryTransport


@pytest.mark.asyncio
async def test_client_initializes_against_capable_server():
    transport = InMemoryTransport()

    server_caps = ServerCapabilities(
        resources=True,
        prompts=["bootstrap"],
        tools=["echo"],
        sampling=True,
        roots=True,
        elicitation=True,
        protocol_version="2025-06-18",
    )
    client_caps = ClientCapabilities(
        sampling=True,
        roots=True,
        elicitation=True,
        protocol_version="2025-06-18",
    )

    server = AsyncMCPServer(capabilities=server_caps)
    client = AsyncMCPClient(capabilities=client_caps)

    server_task = asyncio.create_task(server.serve(transport.server_endpoint()))

    try:
        await client.connect(transport.client_endpoint())
        handshake: HandshakeResult = await client.initialize()

        assert handshake.server_capabilities == server_caps
        assert handshake.client_capabilities == client_caps
        assert server.last_seen_client_capabilities == client_caps
    finally:
        await client.close()
        await server.shutdown()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
