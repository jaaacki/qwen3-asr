from __future__ import annotations
import contextvars
import json
import logging
import os
import sys
from loguru import logger

# Context variable for per-request tracing. Set by the FastAPI middleware.
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(req_id: str) -> contextvars.Token:
    """Set the requestId for the current async context. Returns a token to reset."""
    return _request_id_var.set(req_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Reset the requestId context variable using the token from set_request_id."""
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    return _request_id_var.get()


# Map loguru level names to the canonical set: fatal/error/warn/info/debug/trace
_LEVEL_MAP = {
    "critical": "fatal",
    "warning": "warn",
}


def _json_sink(message: "loguru.Message") -> None:
    """Custom JSON sink: emits structured log entries to stdout."""
    record = message.record
    raw_level = record["level"].name.lower()
    entry: dict = {
        "timestamp": record["time"].isoformat(),
        "level": _LEVEL_MAP.get(raw_level, raw_level),
        "message": record["message"],
        "service": "qwen3-asr",
    }
    req_id = _request_id_var.get()
    if req_id:
        entry["requestId"] = req_id
    # Merge any extra fields added via log.bind(key=value) or log.info(..., key=value)
    extra = record.get("extra", {})
    if extra:
        entry.update({k: v for k, v in extra.items()})
    if record["exception"]:
        entry["err"] = str(record["exception"].value)
    sys.stdout.write(json.dumps(entry) + "\n")
    sys.stdout.flush()


class InterceptHandler(logging.Handler):
    """Route stdlib logging -> Loguru so uvicorn/FastAPI logs are structured too."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Loguru levels that don't exist in stdlib logging.
# Map them to the nearest stdlib equivalent for logging.root.setLevel().
_STDLIB_LEVEL_MAP = {"TRACE": "DEBUG"}


def setup_logger() -> "loguru.Logger":
    log_level = os.getenv("LOG_LEVEL", "info").upper()

    logging.root.handlers = [InterceptHandler()]
    stdlib_level = _STDLIB_LEVEL_MAP.get(log_level, log_level)
    logging.root.setLevel(stdlib_level)

    logger.remove()
    # JSON to stdout — the only sink.
    logger.add(_json_sink, level=log_level, format="{message}", catch=False)

    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    for name in ["uvicorn.access", "uvicorn.error", "uvicorn"]:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    return logger


log = setup_logger()
