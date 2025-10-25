import asyncio
import contextlib

import pytest

from mcp_cli.client import AsyncMCPClient
from mcp_cli.models import (
    ClientCapabilities,
    ClientInfo,
    HandshakeResult,
    ServerCapabilities,
    ServerInfo,
)
from mcp_cli.server import AsyncMCPServer
from mcp_cli.transport import InMemoryTransport
from tests.utils.wait import wait_for_condition


@pytest.mark.asyncio
async def test_client_initializes_with_jsonrpc_handshake():
    transport = InMemoryTransport()

    server_caps = ServerCapabilities(
        logging={},
        prompts={"listChanged": True},
        resources={"subscribe": True, "listChanged": True},
        tools={"listChanged": True},
    )
    client_caps = ClientCapabilities(
        sampling={},
        roots={"listChanged": True},
        elicitation={},
    )

    server_info = ServerInfo(
        name="ExampleServer",
        version="0.1.0",
        title="Example Server Display Name",
    )
    client_info = ClientInfo(
        name="ExampleClient",
        version="0.1.0",
        title="Example Client Display Name",
    )

    server = AsyncMCPServer(capabilities=server_caps, server_info=server_info)
    client = AsyncMCPClient(
        capabilities=client_caps,
        client_info=client_info,
        protocol_version="2025-06-18",
    )

    server_task = asyncio.create_task(server.serve(transport.server_endpoint()))

    try:
        await client.connect(transport.client_endpoint())
        handshake: HandshakeResult = await client.initialize()

        assert handshake.server_capabilities == server_caps
        assert handshake.client_capabilities == client_caps
        assert handshake.server_info == server_info
        assert handshake.client_info == client_info
        assert handshake.protocol_version == "2025-06-18"

        await wait_for_condition(lambda: len(server.received_messages) >= 2)

        initialize_call = server.received_messages[0]
        assert initialize_call["jsonrpc"] == "2.0"
        assert initialize_call["method"] == "initialize"
        assert initialize_call["params"]["protocolVersion"] == "2025-06-18"
        assert initialize_call["params"]["clientInfo"]["name"] == "ExampleClient"
        assert initialize_call["id"] == handshake.request_id

        initialized_note = server.received_messages[1]
        assert initialized_note["jsonrpc"] == "2.0"
        assert initialized_note["method"] == "notifications/initialized"

        assert server.last_seen_client_capabilities == client_caps
    finally:
        await client.close()
        await server.shutdown()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
