from mcp_cli import cli


def test_cli_demo_outputs_handshake_summary(capsys):
    exit_code = cli.main(["demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Handshake succeeded" in captured.out
    assert '"protocolVersion": "2025-06-18"' in captured.out


def test_cli_default_invocation_runs_demo(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Handshake succeeded" in captured.out
