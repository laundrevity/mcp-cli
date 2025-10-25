"""Executable helpers for wiring the CLI entrypoint to client/server stacks."""

from __future__ import annotations

from typing import Awaitable, Callable


async def run_async(main: Callable[[], Awaitable[int]]) -> int:
    """Execute an async main callable, returning its exit status."""
    return await main()
