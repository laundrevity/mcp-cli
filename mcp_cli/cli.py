from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from typing import Iterable, Optional

from .client import AsyncMCPClient
from .models import ClientCapabilities, ClientInfo, ServerCapabilities, ServerInfo
from .server import AsyncMCPServer
from .transport import InMemoryTransport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-cli",
        description="MCP demo CLI with client/server handshake simulation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser(
        "demo",
        help="Run a default client/server handshake simulation and print the results.",
    )
    demo.add_argument(
        "--instructions",
        default="Review capabilities and proceed with sampling delegation when ready.",
        help="Custom server instructions to include in the handshake response.",
    )

    parser.set_defaults(
        command="demo",
        instructions=(
            "Review capabilities and proceed with sampling delegation when ready."
        ),
    )
    return parser


async def run_demo(*, instructions: Optional[str] = None) -> int:
    transport = InMemoryTransport()

    server = AsyncMCPServer(
        capabilities=ServerCapabilities(
            logging={},
            prompts={"listChanged": True},
            resources={"subscribe": True, "listChanged": True},
            tools={"listChanged": True},
        ),
        server_info=ServerInfo(
            name="ExampleServer",
            version="0.1.0",
            title="Example Server Display Name",
        ),
        instructions=instructions,
    )
    client = AsyncMCPClient(
        capabilities=ClientCapabilities(
            sampling={},
            roots={"listChanged": True},
            elicitation={},
        ),
        client_info=ClientInfo(
            name="ExampleClient",
            version="0.1.0",
            title="Example Client Display Name",
        ),
    )

    server_task = asyncio.create_task(server.serve(transport.server_endpoint()))

    try:
        await client.connect(transport.client_endpoint())
        handshake = await client.initialize()

        payload = {
            "protocolVersion": handshake.protocol_version,
            "client": handshake.client_info.to_payload(),
            "server": handshake.server_info.to_payload(),
            "clientCapabilities": handshake.client_capabilities.to_payload(),
            "serverCapabilities": handshake.server_capabilities.to_payload(),
        }
        if handshake.instructions:
            payload["instructions"] = handshake.instructions

        print("Handshake succeeded between ExampleClient and ExampleServer.")
        print(json.dumps(payload, indent=2, sort_keys=True))

        return 0
    finally:
        await client.close()
        await server.shutdown()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "demo":
        return asyncio.run(run_demo(instructions=args.instructions))

    parser.error("Unknown command.")
    return 2
