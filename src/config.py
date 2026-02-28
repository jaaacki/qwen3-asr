"""Startup configuration validation. Imported by server.py and gateway.py lifespans."""
import os
import sys
from logger import log

# --- Extracted config (formerly hardcoded) ---
TRANSLATE_TEMPERATURE = float(os.getenv("TRANSLATE_TEMPERATURE", "0.3"))
TRANSLATE_SRT_TEMPERATURE = float(os.getenv("TRANSLATE_SRT_TEMPERATURE", "0.1"))
SSE_CHUNK_SECONDS = int(os.getenv("SSE_CHUNK_SECONDS", "5"))
SSE_OVERLAP_SECONDS = int(os.getenv("SSE_OVERLAP_SECONDS", "1"))
SUBTITLE_MAX_DURATION = float(os.getenv("SUBTITLE_MAX_DURATION", "7.0"))
SUBTITLE_PAUSE_THRESHOLD = float(os.getenv("SUBTITLE_PAUSE_THRESHOLD", "0.5"))
SUBTITLE_MIN_DURATION = float(os.getenv("SUBTITLE_MIN_DURATION", "0.833"))
SUBTITLE_MIN_GAP = float(os.getenv("SUBTITLE_MIN_GAP", "0.083"))

_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"}
_VALID_QUANTIZE = {"", "int8", "fp8"}


def validate_env() -> None:
    """Validate critical environment variables at startup. Exits on invalid config."""
    errors = []

    # MODEL_ID
    model_id = os.getenv("MODEL_ID", "")
    if not model_id:
        errors.append("MODEL_ID is required but empty or unset")

    # REQUEST_TIMEOUT
    try:
        rt = int(os.getenv("REQUEST_TIMEOUT", "300"))
        if rt <= 0:
            errors.append(f"REQUEST_TIMEOUT must be positive, got {rt}")
    except ValueError as e:
        errors.append(f"REQUEST_TIMEOUT must be an integer: {e}")

    # IDLE_TIMEOUT
    try:
        it = int(os.getenv("IDLE_TIMEOUT", "120"))
        if it < 0:
            errors.append(f"IDLE_TIMEOUT must be non-negative, got {it}")
    except ValueError as e:
        errors.append(f"IDLE_TIMEOUT must be an integer: {e}")

    # LOG_LEVEL
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    if log_level not in _VALID_LOG_LEVELS:
        errors.append(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got '{log_level}'")

    # QUANTIZE
    quantize = os.getenv("QUANTIZE", "")
    if quantize not in _VALID_QUANTIZE:
        errors.append(f"QUANTIZE must be one of {_VALID_QUANTIZE}, got '{quantize}'")

    # WORKER_PORT (if gateway mode)
    if os.getenv("GATEWAY_MODE", "false").lower() == "true":
        try:
            wp = int(os.getenv("WORKER_PORT", "8001"))
            if not (1 <= wp <= 65535):
                errors.append(f"WORKER_PORT must be 1-65535, got {wp}")
        except ValueError as e:
            errors.append(f"WORKER_PORT must be an integer: {e}")

    # WS_WINDOW_MAX_S
    try:
        ws = float(os.getenv("WS_WINDOW_MAX_S", "6.0"))
        if ws <= 0:
            errors.append(f"WS_WINDOW_MAX_S must be positive, got {ws}")
    except ValueError as e:
        errors.append(f"WS_WINDOW_MAX_S must be a float: {e}")

    if errors:
        for err in errors:
            log.error("Config validation failed: {}", err)
        sys.exit(1)

    log.info("Config validation passed")
