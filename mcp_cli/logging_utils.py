from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_ENV_VAR = "MCP_CLI_LOG_DIR"
LOG_FILE_PREFIX = "mcp-cli"
LOG_TIME_FORMAT = "%Y%m%d-%H%M%S%f"

_current_log_file: Optional[Path] = None


def resolve_log_directory() -> Path:
    """Resolve the directory where log files should be written."""
    override = os.getenv(LOG_ENV_VAR)
    if override:
        log_dir = Path(override).expanduser()
    else:
        log_dir = Path(__file__).resolve().parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir.resolve()


def setup_logging() -> Path:
    """Configure the package logger to write to a fresh timestamped file."""
    global _current_log_file

    log_dir = resolve_log_directory()
    timestamp = datetime.now().strftime(LOG_TIME_FORMAT)
    log_path = log_dir / f"{LOG_FILE_PREFIX}-{timestamp}.log"

    logger = logging.getLogger("mcp_cli")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Replace existing handlers so each run writes to a new file.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _current_log_file = log_path
    logger.debug("Initialized logging; writing to %s", log_path)
    return log_path


def get_current_log_file() -> Optional[Path]:
    """Return the log file initialized for the current run, if any."""
    return _current_log_file
