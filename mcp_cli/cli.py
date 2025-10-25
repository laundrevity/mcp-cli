from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    ResourceTemplate,
    RootDescriptor,
    SamplingMessage,
    SamplingResponse,
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
        help="Run a release-planning walkthrough showcasing MCP features.",
    )
    demo.add_argument(
        "--instructions",
        default="Coordinate with the client to draft a release plan and produce a summary.",
        help="Custom server instructions to include in the handshake response.",
    )

    parser.set_defaults(
        command="demo",
        instructions="Coordinate with the client to draft a release plan and produce a summary.",
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
        "2. List and exercise tools.\n"
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
    base_checklist_text = checklist_text

    def _normalize_tasks() -> List[str]:
        tasks: List[str] = []
        for line in base_checklist_text.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                _, task = line.split(".", 1)
                tasks.append(task.strip())
        if not tasks:
            tasks = [
                "Verify MCP handshake and capability negotiation",
                "Run tools/resources/prompts demo",
                "Capture demo transcript for release notes",
            ]
        return tasks

    plan_definition = ToolDefinition(
        name="draft_release_plan",
        title="Draft Release Plan",
        description="Generate a release plan derived from the checklist resource.",
        input_schema={
            "type": "object",
            "properties": {
                "milestone": {
                    "type": "string",
                    "description": "Milestone name to anchor the plan.",
                },
                "audience": {
                    "type": "string",
                    "description": "Primary audience for the release plan.",
                },
            },
            "required": ["milestone"],
        },
    )

    async def plan_handler(arguments: Dict[str, Any]) -> ToolCallResult:
        milestone = str(arguments.get("milestone", "MCP CLI Demo Release"))
        audience = str(arguments.get("audience", "contributors"))
        tasks = _normalize_tasks()
        plan_lines = [
            f"- Align {audience} on {tasks[0].lower()}",
        ]
        if len(tasks) > 1:
            plan_lines.append(f"- Execute: {tasks[1]}")
        if len(tasks) > 2:
            plan_lines.append(f"- Wrap up with: {tasks[2]}")
        plan_lines.append("- Publish release summary and next steps")
        plan_body = "\n".join(plan_lines)
        plan_text = f"Milestone: {milestone}\n{plan_body}"
        checklist_content.text = f"{base_checklist_text}\n\n### Release Plan\n{plan_text}"
        logger.info("Drafted release plan for milestone=%s", milestone)
        return ToolCallResult(
            content=[ContentBlock(type="text", text=plan_text)],
            is_error=False,
        )

    template = ResourceTemplate(
        uri_template="memory:///releases/{version}",
        name="release-notes",
        title="Release Notes Template",
        description="Generate release notes URI for a given version.",
        mime_type="text/markdown",
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
        tools=[
            (echo_definition, echo_handler),
            (plan_definition, plan_handler),
        ],
        resources=[
            (resource_descriptor, resource_content),
            (checklist_descriptor, checklist_content),
        ],
        prompts=[(summarize_prompt, summarize_prompt_handler)],
        resource_templates=[template],
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

    workspace_root = Path(__file__).resolve().parent.parent
    workspace_root_descriptor = RootDescriptor(
        uri=workspace_root.as_uri(),
        name=workspace_root.name or "workspace",
    )
    client.set_roots([workspace_root_descriptor])

    client_roots: List[RootDescriptor] = []

    resource_updates: List[Dict[str, Any]] = []
    list_change_events: Dict[str, int] = {
        "resources": 0,
        "tools": 0,
        "prompts": 0,
        "roots": 0,
    }
    log_messages: List[Dict[str, Any]] = []

    async def on_resource_update(params: Dict[str, Any]) -> None:
        resource_updates.append(params)
        logger.info("Resource update received for %s", params.get("uri"))

    client.register_notification_handler(
        "notifications/resources/updated",
        on_resource_update,
    )

    def _list_changed(kind: str) -> None:
        list_change_events[kind] += 1
        logger.info("%s list changed", kind.capitalize())

    client.register_notification_handler(
        "notifications/resources/list_changed",
        lambda params: _list_changed("resources"),
    )
    client.register_notification_handler(
        "notifications/tools/list_changed",
        lambda params: _list_changed("tools"),
    )
    client.register_notification_handler(
        "notifications/prompts/list_changed",
        lambda params: _list_changed("prompts"),
    )
    
    async def on_server_log(params: Dict[str, Any]) -> None:
        payload = params if isinstance(params, dict) else {}
        log_messages.append(payload)
        level = payload.get("level", "info")
        logger_name = payload.get("logger", "server")
        data = payload.get("data") or {}
        logger.debug(
            "Server log notification level=%s logger=%s data=%s",
            level,
            logger_name,
            data,
        )

    client.register_notification_handler(
        "notifications/message",
        on_server_log,
    )

    server_task = asyncio.create_task(server.serve(transport.server_endpoint()))

    try:
        await client.connect(transport.client_endpoint())
        handshake: HandshakeResult = await client.initialize()
        logger.info("Handshake complete; protocol=%s", handshake.protocol_version)

        await client.set_logging_level("debug")

        client_roots = await server.list_client_roots()
        logger.info("Server observed %d workspace root(s).", len(client_roots))

        logs_path = workspace_root / "logs"
        if logs_path.exists():
            log_root_descriptor = RootDescriptor(
                uri=logs_path.resolve().as_uri(),
                name=f"{workspace_root_descriptor.name} logs"
                if workspace_root_descriptor.name
                else "Logs",
            )
            if not any(root.uri == log_root_descriptor.uri for root in client_roots):
                updated_roots = list(client_roots) + [log_root_descriptor]
                client.set_roots(updated_roots)
                await client.notify_roots_list_changed()
                list_change_events["roots"] += 1
                client_roots = await server.list_client_roots()

        tools = await client.list_tools()
        logger.info("Discovered %d tool(s).", len(tools))
        tools_by_name = {tool.name: tool for tool in tools}

        resources = await client.list_resources()
        logger.info("Discovered %d resource(s).", len(resources))
        resource_snippets: Dict[str, str] = {}
        for descriptor in resources:
            await client.subscribe_resource(descriptor.uri)
        for descriptor in resources:
            contents = await client.read_resource(descriptor.uri)
            snippet = ""
            if contents:
                snippet = (contents[0].text or "").strip()
            resource_snippets[descriptor.uri] = snippet

        prompts = await client.list_prompts()
        logger.info("Discovered %d prompt(s).", len(prompts))

        resource_templates = await client.list_resource_templates()
        logger.info("Discovered %d resource template(s).", len(resource_templates))

        tool_call_result: Optional[ToolCallResult] = None
        plan_text = ""
        plan_tool_name = "draft_release_plan"
        if plan_tool_name in tools_by_name:
            logger.info("Calling demo tool '%s'.", plan_tool_name)
            plan_arguments = {"milestone": "MCP CLI Beta", "audience": "contributors"}
            tool_call_result = await client.call_tool(plan_tool_name, plan_arguments)
            plan_text = "\n".join(
                block.text for block in tool_call_result.content if block.text
            ).strip()
            await server.notify_resource_updated(
                checklist_descriptor.uri,
                title=checklist_descriptor.title,
            )
            await server.notify_tools_list_changed()
            await asyncio.sleep(0)
            updated_contents = await client.read_resource(checklist_descriptor.uri)
            if updated_contents:
                resource_snippets[checklist_descriptor.uri] = (
                    updated_contents[0].text or ""
                ).strip()
        elif tools:
            fallback_tool = tools[0]
            logger.info("Fallback tool '%s' invoked.", fallback_tool.name)
            tool_call_result = await client.call_tool(
                fallback_tool.name,
                {"message": "Fallback demo invocation."},
            )

        prompt_result = None
        if prompts:
            prompt_name = prompts[0].name
            prompt_args = {
                "uri": resources[0].uri if resources else resource_descriptor.uri
            }
            logger.info("Rendering prompt '%s'", prompt_name)
            prompt_result = await client.get_prompt(prompt_name, prompt_args)
        else:
            logger.warning("No prompts available to render.")

        summary_request_text = (
            "Produce an executive release summary combining the draft plan and "
            "resource insights.\n\n"
            f"Draft plan:\n{plan_text or 'Plan unavailable.'}\n\n"
            "Resource snapshots:\n"
            f"- Notes: {resource_snippets.get(resource_descriptor.uri, '')[:160]}\n"
            f"- Checklist: {resource_snippets.get(checklist_descriptor.uri, '')[:160]}"
        )

        sampling_result = None
        try:
            sampling_result = await server.request_sampling(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=ContentBlock(
                            type="text",
                            text=summary_request_text,
                        ),
                    )
                ],
                system_prompt="You are an assistant running locally via MCP sampling delegation.",
                max_tokens=160,
            )
            logger.info(
                "Received sampling response with stop_reason=%s",
                sampling_result.stop_reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sampling request failed: %s", exc)

        if (
            plan_text
            and (
                sampling_result is None
                or not sampling_result.content.text
                or sampling_result.content.text.startswith("[sampling error]")
            )
        ):
            sampling_result = SamplingResponse(
                role="assistant",
                content=ContentBlock(
                    type="text",
                    text=(
                        "Release brief for MCP CLI Beta:\n"
                        f"- Plan outline:\n{plan_text}\n"
                        "- Resources refreshed via MCP subscription notifications.\n"
                        "- Prompts available for ad-hoc summaries."
                    ),
                ),
                model="cli-fallback",
                stop_reason="synthetic",
            )

        await server.notify_resources_list_changed()
        await server.notify_prompts_list_changed()
        await asyncio.sleep(0)

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
        if client_roots:
            payload["roots"] = [root.to_payload() for root in client_roots]
        if tool_call_result is not None:
            payload["toolCall"] = tool_call_result.to_payload()
        if resource_snippets:
            payload["resourcePreview"] = resource_snippets
        if resource_templates:
            payload["resourceTemplates"] = [
                template.to_payload() for template in resource_templates
            ]
        if prompts:
            payload["prompts"] = [definition.to_payload() for definition in prompts]
        if prompt_result is not None:
            payload["prompt"] = prompt_result.to_payload()
        if resource_updates:
            payload["resourceUpdates"] = resource_updates
        if list_change_events:
            payload["listChanged"] = list_change_events
        if sampling_result is not None:
            payload["sampling"] = sampling_result.to_payload()
        if log_messages:
            payload["logs"] = log_messages

        print("Handshake succeeded between ExampleClient and ExampleServer.")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if client_roots:
            print("Workspace roots:")
            for root in client_roots:
                label = f" ({root.name})" if root.name else ""
                print(f"- {root.uri}{label}")
        if log_messages:
            print("Server logs:")
            for entry in log_messages:
                level = entry.get("level", "info")
                logger_name = entry.get("logger", "server")
                data = entry.get("data") or {}
                message = data.get("message") or data.get("event")
                if message:
                    print(f"- [{level}] {logger_name}: {message}")
                else:
                    print(f"- [{level}] {logger_name}: {data}")
        if tool_call_result is not None:
            text_blocks = [
                block.text for block in tool_call_result.content if block.text
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
        if resource_updates:
            print("Resource updates:")
            for update in resource_updates:
                uri = update.get("uri")
                title = update.get("title", "")
                print(f"- {uri} {title}".strip())
        if prompt_result is not None:
            print("Prompt preview:")
            for message in prompt_result.messages:
                if message.content.text:
                    print(f"- {message.role}: {message.content.text.splitlines()[0]}")
        if sampling_result is not None and sampling_result.content.text:
            print("Sampling output:")
            print(sampling_result.content.text)
        if list_change_events:
            print("List change notifications received:")
            for kind, count in list_change_events.items():
                print(f"- {kind}: {count}")

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
