from mcp_cli import cli


def test_cli_demo_outputs_handshake_summary(capsys):
    exit_code = cli.main(["demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Debug log:" in captured.out
    assert "Handshake succeeded" in captured.out
    assert '"protocolVersion": "2025-06-18"' in captured.out


def test_cli_default_invocation_runs_demo(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Handshake succeeded" in captured.out


def test_cli_creates_log_file(monkeypatch, tmp_path, capsys):
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
