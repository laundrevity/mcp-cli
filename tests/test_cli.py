import json

from mcp_cli import cli
from mcp_cli.models import ContentBlock, SamplingRequest, SamplingResponse


def _extract_json(output: str) -> dict:
    json_start = output.find("{")
    json_end = output.rfind("}")
    json_text = output[json_start : json_end + 1]
    return json.loads(json_text)


def _patch_sampling_provider(monkeypatch):
    class _StubProvider:
        async def create_message(self, request: SamplingRequest) -> SamplingResponse:
            return SamplingResponse(
                role="assistant",
                content=ContentBlock(
                    type="text",
                    text="Stubbed sampling output.",
                ),
                model="stub-cli",
                stop_reason="endTurn",
            )

    monkeypatch.setattr(cli, "LocalLLMSamplingProvider", lambda: _StubProvider())


def test_cli_demo_outputs_handshake_summary(monkeypatch, capsys):
    _patch_sampling_provider(monkeypatch)
    exit_code = cli.main(["demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Debug log:" in captured.out
    assert "Handshake succeeded" in captured.out
    data = _extract_json(captured.out)
    assert data["protocolVersion"] == "2025-06-18"
    tool_names = {tool["name"] for tool in data["tools"]}
    assert {"echo", "draft_release_plan"}.issubset(tool_names)
    assert data["toolCall"]["content"][0]["text"].startswith("Milestone:")
    assert len(data["resources"]) == 2
    assert "memory:///guides/demo-notes" in data["resourcePreview"]
    assert "Release Plan" in data["resourcePreview"]["memory:///guides/checklist"]
    assert data["resourceUpdates"]
    template_names = {tmpl["name"] for tmpl in data["resourceTemplates"]}
    assert "release-notes" in template_names
    assert data["listChanged"]["resources"] >= 1
    assert data["listChanged"]["tools"] >= 1
    assert data["listChanged"]["prompts"] >= 1
    assert data["listChanged"]["roots"] >= 1
    assert data["prompts"][0]["name"] == "summarize-resource"
    assert data["prompt"]["messages"][0]["content"]["text"].startswith("Please read")
    assert data["sampling"]["content"]["text"] == "Stubbed sampling output."
    assert data["logs"]
    first_log = data["logs"][0]
    assert first_log["level"] in {"notice", "info", "debug"}
    assert first_log.get("data", {}).get("event") == "log_level_set"
    assert "Sample tool output" in captured.out
    assert "Resource snippets:" in captured.out
    assert "Resource updates:" in captured.out
    assert "Prompt preview:" in captured.out
    assert "Sampling output:" in captured.out
    assert "Server logs:" in captured.out


def test_cli_default_invocation_runs_demo(monkeypatch, capsys):
    _patch_sampling_provider(monkeypatch)
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Handshake succeeded" in captured.out


def test_cli_creates_log_file(monkeypatch, tmp_path, capsys):
    _patch_sampling_provider(monkeypatch)
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("MCP_CLI_LOG_DIR", str(log_dir))

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Handshake succeeded" in captured.out
    assert log_dir.exists()
    log_files = sorted(log_dir.glob("mcp-cli-*.log"))
    assert len(log_files) == 1

    log_content = log_files[0].read_text(encoding="utf-8")
    assert "Initialized logging" in log_content
    assert "Handshake complete" in log_content
    assert "Discovered 2 tool(s)." in log_content
    assert "Discovered 2 resource(s)." in log_content
    assert "Discovered 1 prompt(s)." in log_content
    assert "Discovered 1 resource template(s)." in log_content
    assert "resources list changed" in log_content.lower()
    assert "Resource update received" in log_content
