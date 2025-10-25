from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .client import AsyncMCPClient
from .logging_utils import get_current_log_file, setup_logging
from .models import (
    ClientCapabilities,
    ClientInfo,
    ContentBlock,
    ElicitationRequest,
    ElicitationResponse,
    HandshakeResult,
    PromptArgument,
    PromptDefinition,
    PromptRenderResult,
    ResourceContent,
    ResourceDescriptor,
    ResourceTemplate,
    RootDescriptor,
    SamplingMessage,
    SamplingRequest,
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
        help="Run a capability-exploration walkthrough showcasing MCP features.",
    )
    demo.add_argument(
        "--instructions",
        default="Investigate negotiated MCP capabilities and produce a capability blueprint.",
        help="Custom server instructions to include in the handshake response.",
    )

    parser.set_defaults(
        command="demo",
        instructions="Investigate negotiated MCP capabilities and produce a capability blueprint.",
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
        "# MCP Capability Field Notes\n\n"
        "- Illustrates how handshake, tools, resources, prompts, and sampling interlock.\n"
        "- Highlights transport telemetry and logging for troubleshooting.\n"
        "- Showcases client-led capabilities such as roots, sampling, and elicitation.\n"
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
        "## Capability Checklist\n\n"
        "1. Inspect negotiated handshake data.\n"
        "2. Probe resource subscriptions and updates.\n"
        "3. Capture telemetry-driven insights.\n"
        "4. Compose a capability blueprint informed by elicitation.\n"
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
    elicited_context: Dict[str, str] = {
        "focus": "",
        "constraints": "",
        "verification": "",
    }

    journal_descriptor = ResourceDescriptor(
        uri="memory:///reports/capability-journal",
        name="capability-journal.md",
        title="Capability Journal",
        description="LLM-generated notes captured during the demo run.",
        mime_type="text/markdown",
    )
    journal_content = ResourceContent(
        uri=journal_descriptor.uri,
        name=journal_descriptor.name,
        title=journal_descriptor.title,
        mime_type=journal_descriptor.mime_type,
        text="# Capability Journal\n\n(Waiting for elicitation input.)",
    )
    journal_entries: List[str] = []
    journal_dirty = False

    def render_capability_notes() -> str:
        lines = ["### Capability Summary"]
        focus = elicited_context.get("focus") or "(auto) Context protocol refinement"
        constraints = elicited_context.get("constraints") or "(auto) Honor existing resource/security boundaries"
        verification = elicited_context.get("verification") or "(auto) Run agreed verification command"
        lines.append(f"- Focus: {focus}")
        lines.append(f"- Constraints: {constraints}")
        lines.append(f"- Verification: {verification}")
        return "\n".join(lines)

    checklist_content.text = (
        f"{base_checklist_text}\n\n{render_capability_notes()}"
    )

    # Echo tool stays minimal and demonstrates tool invocation without bespoke handlers.

    async def auto_complete_elicitation(
        request: ElicitationRequest,
        missing_keys: List[str],
        provided: Dict[str, Any],
    ) -> Dict[str, Any]:
        provider = getattr(client, "_sampling_provider", None)
        if provider is None or not missing_keys:
            logger.info("Sampling provider unavailable; using placeholder auto responses.")
            return {key: provided.get(key, f"auto-{key}") for key in missing_keys}

        prompt = textwrap.dedent(
            f"""You are assisting with Model Context Protocol elicitation.

Original request: {request.message or 'n/a'}
Fields needing values: {', '.join(missing_keys) or 'none'}
Fields already provided: {json.dumps(provided, ensure_ascii=False)}

Return a JSON object containing only the requested field names and string values.
"""
        )

        try:
            sampling_response = await provider.create_message(
                SamplingRequest(
                    messages=[
                        SamplingMessage(
                            role="user",
                            content=ContentBlock(type="text", text=prompt),
                        )
                    ],
                    system_prompt="Respond strictly with a JSON object of key/value pairs.",
                    max_tokens=160,
                )
            )
            raw_text = sampling_response.content.text or ""
            payload = json.loads(raw_text)
            logger.debug("Auto elicitation payload=%s", payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto elicitation failed to parse: %s", exc)
            payload = {}

        suggestions: Dict[str, Any] = {}
        for key in missing_keys:
            value = ""
            if isinstance(payload, dict):
                value = payload.get(key)
            if value:
                suggestions[key] = str(value)
            else:
                suggestions[key] = provided.get(key, f"auto-{key}")

        return suggestions

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
        journal_descriptor.uri: journal_content,
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

    async def handle_elicitation(request: ElicitationRequest) -> ElicitationResponse:
        if request.message:
            print(f"Elicitation request: {request.message}")

        prompts = [
            (
                "focus",
                "Describe the capability or behaviour we want to improve",
            ),
            (
                "constraints",
                "List any constraints, risks, or systems we must honor",
            ),
            (
                "verification",
                "Optional: command or test to verify success",
            ),
        ]

        manual_responses: Dict[str, Any] = {}
        remaining_keys: List[str] = []

        for key, description in prompts:
            prompt_text = (
                f"{description}\n"
                "Press Enter to let the local model suggest a value.\n> "
            )
            try:
                user_value = input(prompt_text).strip()
            except EOFError:
                user_value = ""
            if user_value:
                manual_responses[key] = user_value
            else:
                remaining_keys.append(key)

        auto_responses = await auto_complete_elicitation(
            request,
            remaining_keys,
            manual_responses,
        )

        combined = {**auto_responses, **manual_responses}
        if not combined:
            logger.info("Elicitation skipped; no responses captured.")
            return ElicitationResponse(action="decline")

        elicited_context.update({key: str(value) for key, value in combined.items()})
        checklist_content.text = (
            f"{base_checklist_text}\n\n{render_capability_notes()}"
        )

        logger.info(
            "Responding to elicitation request with action=accept fields=%s",
            list(combined.keys()),
        )
        return ElicitationResponse(action="accept", content=combined)

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
        ],
        resources=[
            (resource_descriptor, resource_content),
            (checklist_descriptor, checklist_content),
            (journal_descriptor, journal_content),
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
    client.set_elicitation_handler(handle_elicitation)

    workspace_root = Path(__file__).resolve().parent.parent
    workspace_root_descriptor = RootDescriptor(
        uri=workspace_root.as_uri(),
        name=workspace_root.name or "workspace",
    )
    client.set_roots([workspace_root_descriptor])

    client_roots: List[RootDescriptor] = []
    elicitation_result: Optional[ElicitationResponse] = None

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

        elicitation_result = await server.request_elicitation(
            message="Share the focus, constraints, and verification step for this MCP capability refinement.",
            requested_schema={
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What capability or behaviour do we want to improve?",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "List any constraints, risks, or systems that limit the change.",
                    },
                    "verification": {
                        "type": "string",
                        "description": "How would we verify that the improvement worked (command, test, metric)?",
                    },
                },
                "required": ["focus"],
            },
        )
        if (
            elicitation_result.action == "accept"
            and elicitation_result.content
        ):
            elicited_context.update(
                {k: str(v) for k, v in elicitation_result.content.items()}
            )
        else:
            logger.info(
                "Elicitation returned action=%s; using default context.",
                elicitation_result.action,
            )

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

        async def append_journal_entry(title: str, body: str) -> None:
            nonlocal journal_dirty
            cleaned = body.strip() or "(no content recorded)"
            filtered = [
                entry
                for entry in journal_entries
                if not entry.startswith(f"## {title}")
            ]
            filtered.append(f"## {title}\n{cleaned}")
            journal_entries[:] = filtered
            journal_content.text = "# Capability Journal\n\n" + "\n\n".join(journal_entries)
            resource_snippets[journal_descriptor.uri] = journal_content.text.strip()
            journal_dirty = True

        async def capture_sampling_note(
            title: str,
            prompt_text: str,
            *,
            fallback: str,
        ) -> SamplingResponse:
            logger.info("Requesting sampling note '%s'", title)
            try:
                result = await server.request_sampling(
                    messages=[
                        SamplingMessage(
                            role="user",
                            content=ContentBlock(type="text", text=prompt_text),
                        )
                    ],
                    system_prompt=(
                        "You are an MCP capability analyst summarizing negotiated "
                        "features and proposing follow-up explorations."
                    ),
                    max_tokens=220,
                )
                content_text = result.content.text or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sampling note '%s' failed: %s", title, exc)
                content_text = fallback
                result = SamplingResponse(
                    role="assistant",
                    content=ContentBlock(type="text", text=content_text),
                    model="cli-fallback",
                    stop_reason="synthetic",
                )
            await append_journal_entry(title, content_text)
            return result

        prompts = await client.list_prompts()
        logger.info("Discovered %d prompt(s).", len(prompts))

        resource_templates = await client.list_resource_templates()
        logger.info("Discovered %d resource template(s).", len(resource_templates))

        handshake_overview = textwrap.dedent(
            f"""Instructions: {handshake.instructions or 'None'}
Client capabilities: {json.dumps(handshake.client_capabilities.to_payload(), indent=2)}
Server capabilities: {json.dumps(handshake.server_capabilities.to_payload(), indent=2)}
Roots subscribed: {', '.join(root.uri for root in client_roots) if client_roots else 'None'}
"""
        )

        resource_catalogue = "\n".join(
            f"- {uri}: {(resource_snippets.get(uri, '').splitlines()[0])}"
            for uri in resource_snippets
        )
        recent_logs = json.dumps(log_messages[-5:], indent=2) if log_messages else "(no logs yet)"

        diagnostic_prompt = textwrap.dedent(
            f"""Provide a diagnostic summary of the current MCP session.

Focus: {elicited_context.get('focus') or '(auto)'}
Constraints: {elicited_context.get('constraints') or '(auto)'}
Verification idea: {elicited_context.get('verification') or '(auto)'}

Handshake snapshot:
{handshake_overview}

Resource catalogue:
{resource_catalogue}

Recent server logs (up to 5):
{recent_logs}
"""
        )

        iteration_results: List[SamplingResponse] = []
        iteration_results.append(
            await capture_sampling_note(
                "Diagnostic summary",
                diagnostic_prompt,
                fallback="Diagnostic summary unavailable; inspect handshake manually.",
            )
        )

        hypothesis_prompt = textwrap.dedent(
            f"""Using the diagnostic summary below, propose a concrete capability refinement hypothesis.

Diagnostic summary:
{iteration_results[-1].content.text if iteration_results[-1].content.text else '(empty)'}

Keep within the stated constraints: {elicited_context.get('constraints') or '(none)'}
"""
        )

        iteration_results.append(
            await capture_sampling_note(
                "Refinement hypothesis",
                hypothesis_prompt,
                fallback="Hypothesis deferred; capture manually.",
            )
        )

        action_prompt = textwrap.dedent(
            f"""Draft an actionable plan that operationalises the refinement hypothesis.

Hypothesis:
{iteration_results[-1].content.text if iteration_results[-1].content.text else '(empty)'}

Diagnostic context:
{iteration_results[-2].content.text if iteration_results[-2].content.text else '(empty)'}

Proposed verification step: {elicited_context.get('verification') or '(not specified)'}
"""
        )

        iteration_results.append(
            await capture_sampling_note(
                "Action plan",
                action_prompt,
                fallback="Plan pending; execute verification step manually.",
            )
        )

        sampling_result = iteration_results[-1] if iteration_results else None

        if journal_dirty:
            await server.notify_resource_updated(
                journal_descriptor.uri,
                title=journal_descriptor.title,
            )

        tool_call_result: Optional[ToolCallResult] = None
        if "echo" in tools_by_name:
            echo_message = " | ".join(
                f"{key}={value}"
                for key, value in elicited_context.items()
                if value
            ) or "No elicitation context captured"
            logger.info("Calling echo tool with elicited context snapshot.")
            tool_call_result = await client.call_tool(
                "echo",
                {"message": f"MCP capabilities anchored in: {echo_message}"},
            )

        await server.notify_resource_updated(
            checklist_descriptor.uri,
            title=checklist_descriptor.title,
        )
        await asyncio.sleep(0)
        updated_contents = await client.read_resource(checklist_descriptor.uri)
        if updated_contents:
            resource_snippets[checklist_descriptor.uri] = (
                updated_contents[0].text or ""
            ).strip()

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
        if elicitation_result is not None:
            payload["elicitation"] = elicitation_result.to_payload()

        print("Handshake succeeded between ExampleClient and ExampleServer.")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if client_roots:
            print("Workspace roots:")
            for root in client_roots:
                label = f" ({root.name})" if root.name else ""
                print(f"- {root.uri}{label}")
        def _print_summary() -> None:
            focus = elicited_context.get("focus") or "(auto) Context protocol refinement"
            constraints = elicited_context.get("constraints") or "(auto) Honor existing boundaries"
            verification = elicited_context.get("verification") or "(auto) Run verification command"
            print("Focus:", focus)
            print("Constraints:", constraints)
            print("Verification:", verification)
            if iteration_results:
                print("\nDiagnostic:")
                print(textwrap.shorten(iteration_results[0].content.text or "(empty)", width=240, placeholder="…"))
            if len(iteration_results) > 1:
                print("\nHypothesis:")
                print(textwrap.shorten(iteration_results[1].content.text or "(empty)", width=240, placeholder="…"))
            if sampling_result is not None and sampling_result.content.text:
                print("\nAction plan:")
                print(textwrap.shorten(sampling_result.content.text, width=240, placeholder="…"))
            if tool_call_result is not None:
                text_blocks = [
                    block.text for block in tool_call_result.content if block.text
                ]
                if text_blocks:
                    print("\nEcho tool output:")
                    print(text_blocks[0])

        _print_summary()

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
