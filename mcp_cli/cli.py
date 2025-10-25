from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from typing import Any, Dict, Iterable, Optional

from .client import AsyncMCPClient
from .logging_utils import get_current_log_file, setup_logging
from .models import (
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
    ServerCapabilities,
    ServerInfo,
    ToolCallResult,
    ToolDefinition,
)
from .sampling import LocalLLMSamplingProvider
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
    logger = logging.getLogger("mcp_cli.cli")
    log_path = get_current_log_file()
    if log_path is not None:
        logger.debug("Demo run writing logs to %s", log_path)

    transport = InMemoryTransport()

    echo_definition = ToolDefinition(
        name="echo",
        title="Echo",
        description="Echo arguments back as text output.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back from the server tool.",
                }
            },
            "required": ["message"],
        },
    )

    async def echo_handler(arguments: Dict[str, Any]) -> ToolCallResult:
        message = str(arguments.get("message", ""))
        logger.info("Echo tool invoked with message=%s", message)
        return ToolCallResult(
            content=[ContentBlock(type="text", text=f"ECHO: {message}")],
            is_error=False,
        )

    resource_text = (
        "# MCP Demo Notes\n\n"
        "- Demonstrates handshake, tools, resources, prompts, and sampling.\n"
        "- Resources are served from in-memory descriptors for clarity.\n"
        "- Sampling delegates to a local llama.cpp server when available.\n"
    )
    resource_descriptor = ResourceDescriptor(
        uri="memory:///guides/demo-notes",
        name="demo-notes.md",
        title="MCP Demo Notes",
        description="Overview of the MCP CLI demonstration.",
        mime_type="text/markdown",
    )
    resource_content = ResourceContent(
        uri=resource_descriptor.uri,
        name=resource_descriptor.name,
        title=resource_descriptor.title,
        mime_type=resource_descriptor.mime_type,
        text=resource_text,
    )

    checklist_text = (
        "## CLI Checklist\n\n"
        "1. Initialize handshake.\n"
        "2. List and call the echo tool.\n"
        "3. Enumerate resources and prompts.\n"
        "4. Delegate sampling to local LLM.\n"
    )
    checklist_descriptor = ResourceDescriptor(
        uri="memory:///guides/checklist",
        name="checklist.md",
        title="CLI Checklist",
        description="Steps executed during the demo run.",
        mime_type="text/markdown",
    )
    checklist_content = ResourceContent(
        uri=checklist_descriptor.uri,
        name=checklist_descriptor.name,
        title=checklist_descriptor.title,
        mime_type=checklist_descriptor.mime_type,
        text=checklist_text,
    )

    resource_map = {
        resource_descriptor.uri: resource_content,
        checklist_descriptor.uri: checklist_content,
    }

    summarize_prompt = PromptDefinition(
        name="summarize-resource",
        title="Summarize Resource",
        description="Creates a user message instructing the assistant to summarize a resource.",
        arguments=[
            PromptArgument(
                name="uri",
                description="Resource URI to summarize",
                required=True,
            )
        ],
    )

    async def summarize_prompt_handler(arguments: Dict[str, Any]) -> PromptRenderResult:
        uri = str(arguments.get("uri", resource_descriptor.uri))
        selected = resource_map.get(uri, resource_content)
        text = selected.text or selected.title or uri
        message_text = (
            "Please read the following resource and provide a concise summary:\n\n"
            f"Resource URI: {uri}\n\n{text}"
        )
        return PromptRenderResult(
            description="Summarize the referenced resource for quick context.",
            messages=[
                SamplingMessage(
                    role="user",
                    content=ContentBlock(type="text", text=message_text),
                )
            ],
        )

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
        tools=[(echo_definition, echo_handler)],
        resources=[
            (resource_descriptor, resource_content),
            (checklist_descriptor, checklist_content),
        ],
        prompts=[(summarize_prompt, summarize_prompt_handler)],
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
    client.set_sampling_provider(LocalLLMSamplingProvider())

    server_task = asyncio.create_task(server.serve(transport.server_endpoint()))

    try:
        await client.connect(transport.client_endpoint())
        handshake: HandshakeResult = await client.initialize()
        logger.info("Handshake complete; protocol=%s", handshake.protocol_version)

        tools = await client.list_tools()
        logger.info("Discovered %d tool(s).", len(tools))

        demo_message = "Hello from the MCP CLI demo!"
        call_result = None
        if tools:
            tool_name = tools[0].name
            logger.info("Calling demo tool '%s'.", tool_name)
            call_result = await client.call_tool(
                tool_name,
                {"message": demo_message},
            )
        else:
            logger.warning("No tools available to invoke.")

        resources = await client.list_resources()
        logger.info("Discovered %d resource(s).", len(resources))
        resource_snippets: Dict[str, str] = {}
        for descriptor in resources[:2]:  # limit to first two for demo output
            contents = await client.read_resource(descriptor.uri)
            snippet = ""
            if contents:
                snippet = (contents[0].text or "").strip()
            resource_snippets[descriptor.uri] = snippet

        prompts = await client.list_prompts()
        logger.info("Discovered %d prompt(s).", len(prompts))
        prompt_result = None
        if prompts:
            prompt_name = prompts[0].name
            prompt_args = {"uri": resources[0].uri if resources else resource_descriptor.uri}
            logger.info("Rendering prompt '%s'", prompt_name)
            prompt_result = await client.get_prompt(prompt_name, prompt_args)
        else:
            logger.warning("No prompts available to render.")

        sampling_result = None
        try:
            sampling_result = await server.request_sampling(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=ContentBlock(
                            type="text",
                            text=(
                                "Provide a single-sentence summary of this MCP demo "
                                "showing handshake, tool listing, and tool invocation."
                            ),
                        ),
                    )
                ],
                system_prompt=(
                    "You are an assistant running locally via MCP sampling delegation."
                ),
                max_tokens=128,
            )
            logger.info(
                "Received sampling response with stop_reason=%s",
                sampling_result.stop_reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sampling request failed: %s", exc)

        payload = {
            "protocolVersion": handshake.protocol_version,
            "client": handshake.client_info.to_payload(),
            "server": handshake.server_info.to_payload(),
            "clientCapabilities": handshake.client_capabilities.to_payload(),
            "serverCapabilities": handshake.server_capabilities.to_payload(),
            "tools": [tool.to_payload() for tool in tools],
            "resources": [descriptor.to_payload() for descriptor in resources],
        }
        if handshake.instructions:
            payload["instructions"] = handshake.instructions
        if call_result is not None:
            payload["toolCall"] = call_result.to_payload()
        if resource_snippets:
            payload["resourcePreview"] = resource_snippets
        if prompts:
            payload["prompts"] = [definition.to_payload() for definition in prompts]
        if prompt_result is not None:
            payload["prompt"] = prompt_result.to_payload()
        if sampling_result is not None:
            payload["sampling"] = sampling_result.to_payload()

        print("Handshake succeeded between ExampleClient and ExampleServer.")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if call_result is not None:
            text_blocks = [
                block.text for block in call_result.content if block.text
            ]
            if text_blocks:
                print("Sample tool output:")
                for block in text_blocks:
                    print(f"- {block}")
        if resource_snippets:
            print("Resource snippets:")
            for uri, snippet in resource_snippets.items():
                preview = snippet.splitlines()[0] if snippet else "(empty)"
                print(f"- {uri}: {preview}")
        if prompt_result is not None:
            print("Prompt preview:")
            for message in prompt_result.messages:
                if message.content.text:
                    print(f"- {message.role}: {message.content.text.splitlines()[0]}")
        if sampling_result is not None and sampling_result.content.text:
            print("Sampling output:")
            print(sampling_result.content.text)

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

    log_path = setup_logging()
    print(f"Debug log: {log_path}")
    arg_list = list(argv) if argv is not None else None
    logging.getLogger("mcp_cli.cli").debug("CLI invoked with args: %s", arg_list)

    if args.command == "demo":
        return asyncio.run(run_demo(instructions=args.instructions))

    parser.error("Unknown command.")
    return 2
