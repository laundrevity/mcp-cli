import asyncio
import contextlib

import pytest

from mcp_cli.client import AsyncMCPClient
from mcp_cli.models import (
    ClientCapabilities,
    ClientInfo,
    ContentBlock,
    HandshakeResult,
    PromptArgument,
    PromptDefinition,
    PromptRenderResult,
    ResourceContent,
    ResourceDescriptor,
    SamplingMessage,
    SamplingRequest,
    SamplingResponse,
    ServerCapabilities,
    ServerInfo,
    ToolCallResult,
    ToolDefinition,
)
from mcp_cli.sampling import SamplingProvider
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

    echo_definition = ToolDefinition(
        name="echo",
        description="Echo arguments.",
        title="Echo Tool",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    )

    async def echo_handler(arguments: dict) -> ToolCallResult:
        message = str(arguments.get("message", ""))
        return ToolCallResult(
            content=[ContentBlock(type="text", text=f"ECHO: {message}")],
            is_error=False,
        )

    notes_descriptor = ResourceDescriptor(
        uri="memory:///tests/demo-note",
        name="demo-note.md",
        title="Demo Note",
        description="Short reference note for tests.",
        mime_type="text/markdown",
    )
    notes_content = ResourceContent(
        uri=notes_descriptor.uri,
        name=notes_descriptor.name,
        title=notes_descriptor.title,
        mime_type=notes_descriptor.mime_type,
        text="### Demo Note\n\nThis content is used during tests.",
    )

    prompt_definition = PromptDefinition(
        name="summarize-note",
        title="Summarize Note",
        description="Creates a summary instruction for a test note.",
        arguments=[
            PromptArgument(
                name="uri",
                description="Resource URI to summarize",
                required=True,
            )
        ],
    )

    resource_map = {notes_descriptor.uri: notes_content}

    async def prompt_handler(arguments: dict) -> PromptRenderResult:
        uri = str(arguments.get("uri", notes_descriptor.uri))
        content = resource_map.get(uri, notes_content)
        return PromptRenderResult(
            description="Summarize the requested note.",
            messages=[
                SamplingMessage(
                    role="user",
                    content=ContentBlock(
                        type="text",
                        text=f"Please summarize:\n{content.text}",
                    ),
                )
            ],
        )

    server = AsyncMCPServer(
        capabilities=server_caps,
        server_info=server_info,
        tools=[(echo_definition, echo_handler)],
        resources=[(notes_descriptor, notes_content)],
        prompts=[(prompt_definition, prompt_handler)],
    )
    client = AsyncMCPClient(
        capabilities=client_caps,
        client_info=client_info,
        protocol_version="2025-06-18",
    )
    client.set_sampling_provider(_StubSamplingProvider())

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

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

        tool_result = await client.call_tool("echo", {"message": "hello"})
        assert tool_result.is_error is False
        assert tool_result.content[0].text == "ECHO: hello"

        resources = await client.list_resources()
        assert len(resources) == 1
        resource_descriptor = resources[0]
        assert resource_descriptor.uri == "memory:///tests/demo-note"

        contents = await client.read_resource(resource_descriptor.uri)
        assert len(contents) == 1
        assert "Demo Note" in (contents[0].text or "")

        prompts = await client.list_prompts()
        assert len(prompts) == 1
        assert prompts[0].name == "summarize-note"

        prompt_result = await client.get_prompt(
            "summarize-note",
            {"uri": resource_descriptor.uri},
        )
        assert "summarize" in (prompt_result.description or "").lower()
        assert prompt_result.messages[0].content.type == "text"

        sampling_result = await server.request_sampling(
            messages=[
                SamplingMessage(
                    role="user",
                    content=ContentBlock(type="text", text="Provide a friendly greeting."),
                )
            ],
            system_prompt="You are a stub assistant.",
            max_tokens=64,
        )

        assert sampling_result.content.text == "Stubbed sampling response."
        assert sampling_result.stop_reason == "endTurn"
    finally:
        await client.close()
        await server.shutdown()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


class _StubSamplingProvider(SamplingProvider):
    async def create_message(self, request: SamplingRequest) -> SamplingResponse:
        return SamplingResponse(
            role="assistant",
            content=ContentBlock(
                type="text",
                text="Stubbed sampling response.",
            ),
            model="stub",
            stop_reason="endTurn",
        )
