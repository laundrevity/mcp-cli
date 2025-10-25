from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_events: List[Dict[str, object]] = []
_lock = threading.Lock()
_log_path: Optional[Path] = None


def initialize(log_dir: Path) -> None:
    """Reset the in-memory event store and prepare a JSONL log file."""
    global _events, _log_path
    with _lock:
        _events = []
        log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = (log_dir / "events.jsonl").resolve()
        _log_path.write_text("", encoding="utf-8")


def reset() -> None:
    """Clear the in-memory event buffer without touching the log file."""
    global _events
    with _lock:
        _events = []


def record_event(
    *,
    role: str,
    direction: str,
    payload: Dict[str, object],
    channel: Optional[str] = None,
) -> Dict[str, object]:
    """Store a JSON-RPC event in memory (and append to disk if configured)."""
    entry = {
        "id": None,
        "timestamp": time.time(),
        "role": role,
        "direction": direction,
        "channel": channel,
        "payload": payload,
    }
    with _lock:
        entry["id"] = len(_events)
        _events.append(entry)
        if _log_path is not None:
            with _log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
    return entry  # type: ignore[return-value]


def get_events(since: int = -1) -> List[Dict[str, object]]:
    """Return events whose id is greater than ``since``."""
    with _lock:
        if since < -1:
            since = -1
        return [event for event in _events if event["id"] > since]


def load_events_from_file(path: Path) -> List[Dict[str, object]]:
    """Load events from a JSONL file without mutating the in-memory store."""
    events: List[Dict[str, object]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def log_path() -> Optional[Path]:
    with _lock:
        return _log_path
