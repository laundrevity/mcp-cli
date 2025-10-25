# Repository Guidelines

## Project Structure & Module Organization
The CLI entrypoint delegates to `mcp_cli/cli.py`, which orchestrates an async client/server handshake demo by wiring together `mcp_cli/client.py`, `mcp_cli/server.py`, and `mcp_cli/transport.py`. Keep reusable protocol models in `mcp_cli/models.py`, and leave `main.py` as a thin passthrough into the package. Store protocol fixtures, JSON-RPC helpers, and shared test doubles inside `tests/utils/`, and stash sample transcripts under `tests/data/` for clarity.

## Build, Test, and Development Commands
- `uv sync` — install the locked toolchain from `pyproject.toml` and `uv.lock`.
- `uv run python main.py` — run the default handshake simulation to confirm the CLI still connects a client and server.
- `uv run python main.py --help` — inspect subcommands and flags.
- `uv run pytest` — execute the async test suite (pytest + pytest-asyncio) with default markers.
- `uv run pytest tests/test_client.py::test_client_initializes_with_jsonrpc_handshake` — re-run a focused spec during TDD.

## Logging & Diagnostics
Each CLI invocation creates `logs/mcp-cli-<timestamp>.log` with DEBUG-level messages from the client, server, and transport orchestration. Override the destination by exporting `MCP_CLI_LOG_DIR=/tmp/mcp-cli-logs` (tests do this automatically). Tail the latest file while iterating (`tail -f logs/mcp-cli-*.log`) to trace JSON-RPC exchanges and transport state transitions.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, descriptive `snake_case` for functions, and `PascalCase` for protocol dataclasses. Prefer explicit `async def` workflows, type hints on public APIs, and docstrings that cite relevant MCP sections (for example, “Spec §Server Features”). Run `uv run ruff check .` before opening a PR; use `ruff format` to auto-format when necessary.

## Testing Guidelines
House tests in `tests/` and mirror module names (e.g., `tests/test_server.py`). Start each behavior change with a failing async spec that uses `pytest.mark.asyncio`. Mock transport edges with in-memory streams, and assert JSON-RPC payloads against fixtures adapted from MCP_SPEC.md examples. Maintain coverage above 85%, and add regression tests whenever adjusting capability negotiation or sampling delegation.

## Commit & Pull Request Guidelines
Write commits in the imperative mood (for example, “Implement sampling delegation”). Reference issue IDs or MCP spec clauses in the body when relevant, and include before/after CLI transcripts for UX-facing updates. Pull requests must summarize testing evidence (`uv run pytest` output), describe any new tools or resources exposed, and highlight security considerations around resource access or delegated sampling.

## MCP Agent Roadmap
Ship one host-managed client and one server with full capabilities, allowing the client to delegate sampling to a local LLM when negotiated. Document any deviations from MCP_SPEC.md, keep capability exchange states explicit in tests, and note security decisions around consent, tools, and resource boundaries.
