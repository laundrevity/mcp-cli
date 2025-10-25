import asyncio
from typing import Callable


class ConditionTimeoutError(TimeoutError):
    """Raised when an async wait_for_condition call times out."""


async def wait_for_condition(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.0,
    interval: float = 0.01,
) -> None:
    """Poll predicate until true or timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() >= deadline:
            raise ConditionTimeoutError("Condition not met within timeout.")
        await asyncio.sleep(interval)
