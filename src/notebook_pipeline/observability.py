"""Standard logging, with JSON records and no root-logger changes."""

import json
import logging
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ConsoleFormatter(logging.Formatter):
    def formatException(self, exc_info) -> str:
        # The notebook already displays the traceback; full details also live on disk.
        return ""


class JsonFormatter(logging.Formatter):
    def __init__(self, context: dict):
        super().__init__()
        self.context = context

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            **self.context,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            error_type, error, tb = record.exc_info
            payload["exception_type"] = error_type.__name__
            payload["exception_message"] = str(error)
            # Include chained causes and notes, but never capture local variables.
            payload["traceback"] = "".join(traceback.format_exception(error_type, error, tb))
        return json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)


def make_logger(log_dir: Path, name: str, context: dict) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    # A private logger avoids global handler registries and duplicate handlers on re-setup.
    logger = logging.Logger(name, level=logging.INFO)
    logger.propagate = False
    formatter = JsonFormatter(context)
    for suffix, level in (("events", logging.INFO), ("errors", logging.ERROR)):
        handler = logging.FileHandler(log_dir / f"{name}.{suffix}.jsonl", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console_formatter = ConsoleFormatter("%(asctime)sZ %(levelname)s %(message)s")
    console_formatter.converter = time.gmtime
    console.setFormatter(console_formatter)
    logger.addHandler(console)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
