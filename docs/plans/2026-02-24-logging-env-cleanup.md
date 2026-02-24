# Logging Overhaul, .env Cleanup & Translation Wiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add atomic-level loguru logging to every source file, clean up .env.example with proper Ollama Cloud defaults, and make LOG_LEVEL configurable.

**Architecture:** All logging uses the existing `src/logger.py` loguru setup. Every endpoint logs request entry (method, path, params, file size) and exit (duration, status). translator.py logs API calls. worker.py gets full logging parity with server.py. LOG_LEVEL env var controls verbosity.

**Tech Stack:** loguru (already installed), Python time module for duration tracking.

---

### Task 1: Make LOG_LEVEL configurable in logger.py

**Files:**
- Modify: `src/logger.py`

**Step 1: Read current logger.py**

Current `setup_logger()` hardcodes `level="INFO"` on line 38.

**Step 2: Add LOG_LEVEL env var support**

```python
import logging
import os
import sys
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and route them to Loguru.
    Code from loguru documentation.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logger():
    """Configure Loguru to intercept uvicorn and fastapi logs, and format them clearly."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(log_level)

    # Remove all loguru's default handlers
    logger.remove()

    # Configure our own handler to sys.stderr with nice formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Remove every other logger's handlers and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # Let uvicorn loggers intercept gracefully
    for name in ["uvicorn.access", "uvicorn.error", "uvicorn"]:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    return logger

log = setup_logger()
```

**Step 3: Verify no tests break**

Run: `pytest E2Etest/ -m "not slow and not performance" -v --timeout=300`
Expected: all pass (logger change is transparent)

**Step 4: Commit**

```bash
git add src/logger.py
git commit -m "feat: make LOG_LEVEL configurable via env var (default INFO)"
```

---

### Task 2: Add atomic logging to translator.py

**Files:**
- Modify: `src/translator.py`

**Step 1: Add loguru import and request-level logging**

```python
import os
import time
from typing import Optional

from logger import log


def _get_client():
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("The 'openai' python package is required for translation. Please run `pip install openai`.")

    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return AsyncOpenAI(**client_kwargs)


async def translate_text(text: str, target_lang: str) -> str:
    """Translate raw text to the specified target language using an OpenAI-compatible API."""
    if not text.strip():
        return text

    client = _get_client()
    model = os.getenv("TRANSLATE_MODEL", "gpt-3.5-turbo")

    if target_lang.lower() in ("en", "english"):
        lang_name = "English"
    elif target_lang.lower() in ("zh", "chinese"):
        lang_name = "Chinese"
    else:
        lang_name = target_lang

    log.info("Translation request | model={} target={} text_len={}", model, lang_name, len(text))
    t0 = time.time()

    prompt = (
        f"Translate the following spoken audio transcription into {lang_name}. "
        f"Preserve the original meaning and tone. Output ONLY the translated text required "
        f"without any introduction, markdown blocks, quotes, or commentary.\n\nText: {text}"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional and highly accurate translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
    except Exception as e:
        elapsed = time.time() - t0
        log.error("Translation API error | model={} target={} elapsed={:.2f}s error={}", model, lang_name, elapsed, e)
        raise

    elapsed = time.time() - t0

    if not response.choices:
        log.error("Translation returned no choices | model={} target={} elapsed={:.2f}s", model, lang_name, elapsed)
        raise ValueError("Translation returned no choices")

    result = response.choices[0].message.content.strip()
    log.info("Translation complete | model={} target={} in_len={} out_len={} elapsed={:.2f}s", model, lang_name, len(text), len(result), elapsed)
    return result


async def translate_srt(srt_content: str, target_lang: str) -> str:
    """Translate SRT subtitle content to the specified target language, preserving timestamps."""
    if not srt_content.strip():
        return srt_content

    client = _get_client()
    model = os.getenv("TRANSLATE_MODEL", "gpt-3.5-turbo")

    if target_lang.lower() in ("en", "english"):
        lang_name = "English"
    elif target_lang.lower() in ("zh", "chinese"):
        lang_name = "Chinese"
    else:
        lang_name = target_lang

    log.info("SRT translation request | model={} target={} srt_len={}", model, lang_name, len(srt_content))
    t0 = time.time()

    prompt = (
        f"Translate the following subtitle (SRT) content into {lang_name}. "
        f"Preserve the original SRT format and timing tags perfectly. "
        f"Output ONLY the valid translated SRT content without any introduction, markdown wrapping blocks (like ```srt), or commentary. "
        f"Do NOT change the SRT index numbers or timestamp lines.\n\nSRT Content:\n{srt_content}"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional subtitle translator. You MUST output ONLY valid SRT format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
    except Exception as e:
        elapsed = time.time() - t0
        log.error("SRT translation API error | model={} target={} elapsed={:.2f}s error={}", model, lang_name, elapsed, e)
        raise

    elapsed = time.time() - t0

    if not response.choices:
        log.error("SRT translation returned no choices | model={} target={} elapsed={:.2f}s", model, lang_name, elapsed)
        raise ValueError("Translation returned no choices")

    result = response.choices[0].message.content.strip()

    # Strip markdown block if model ignored instructions
    if result.startswith("```"):
        lines = result.split("\n")
        if lines[0].startswith("```"):
            lines.pop(0)
        if lines[-1].startswith("```"):
            lines.pop(-1)
        result = "\n".join(lines).strip()
        log.debug("Stripped markdown wrapper from SRT translation output")

    log.info("SRT translation complete | model={} target={} in_len={} out_len={} elapsed={:.2f}s", model, lang_name, len(srt_content), len(result), elapsed)
    return result
```

**Step 2: Commit**

```bash
git add src/translator.py
git commit -m "feat: add atomic loguru logging to translator.py"
```

---

### Task 3: Add atomic logging to server.py endpoints

**Files:**
- Modify: `src/server.py`

Add request-level logging to every HTTP endpoint and WebSocket handler. The model lifecycle logging (load/unload) already exists — this adds the missing endpoint-level logs.

**Endpoints to instrument:**
- `POST /v1/audio/transcriptions` (line ~583)
- `POST /v1/audio/translations` (line ~618)
- `POST /v1/audio/subtitles` (line ~700)
- `POST /v1/audio/transcriptions/stream` (SSE)
- `WS /ws/transcribe` (already has disconnect/error logs — add connect + chunk count)

**Pattern for each endpoint:**

```python
# At entry:
log.info("POST /v1/audio/transcriptions | file={} size={} language={}", file.filename, len(audio_bytes), language)
t0 = time.time()

# At exit (success):
log.info("POST /v1/audio/transcriptions | completed in {:.2f}s text_len={}", time.time() - t0, len(text))

# At exit (error):
log.error("POST /v1/audio/transcriptions | failed in {:.2f}s error={}", time.time() - t0, str(e))
```

**Step 1: Add logging to /v1/audio/transcriptions**

After `audio_bytes = await file.read()` (line 591), add:
```python
log.info("POST /v1/audio/transcriptions | file={} size={} language={}", file.filename, len(audio_bytes), language)
t0 = time.time()
```

Before the return (line 615), add:
```python
log.info("POST /v1/audio/transcriptions | completed in {:.2f}s text_len={} lang={}", time.time() - t0, len(text), language_code)
```

In the timeout except (line 606), add:
```python
log.warning("POST /v1/audio/transcriptions | timed out after {:.2f}s", time.time() - t0)
```

**Step 2: Add logging to /v1/audio/translations**

After `audio_bytes = await file.read()` (line 629), add:
```python
log.info("POST /v1/audio/translations | file={} size={} target={} format={}", file.filename, len(audio_bytes), target_lang, response_format)
t0 = time.time()
```

Before each return, add appropriate log.info/log.warning/log.error.

**Step 3: Add logging to /v1/audio/subtitles**

After `audio_bytes = await file.read()` (line 717), add:
```python
log.info("POST /v1/audio/subtitles | file={} size={} language={} mode={}", file.filename, len(audio_bytes), language, mode)
t0 = time.time()
```

Before the return (line 764), add:
```python
log.info("POST /v1/audio/subtitles | completed in {:.2f}s mode={} srt_len={}", time.time() - t0, mode, len(srt_content))
```

**Step 4: Add logging to SSE stream endpoint**

In the SSE generator/endpoint, add:
```python
log.info("POST /v1/audio/transcriptions/stream | file={} size={} language={}", file.filename, len(audio_bytes), language)
```

**Step 5: Add WebSocket connect logging**

At the top of the WebSocket handler, add:
```python
log.info("[WS] Client connected")
```

Add chunk counter and log it on disconnect:
```python
log.info("[WS] Client disconnected | chunks_processed={}", chunk_count)
```

**Step 6: Run tests**

Run: `pytest E2Etest/ -m "not slow and not performance" -v --timeout=300`
Expected: all pass

**Step 7: Commit**

```bash
git add src/server.py
git commit -m "feat: add atomic request-level logging to all server.py endpoints"
```

---

### Task 4: Add atomic logging to worker.py

**Files:**
- Modify: `src/worker.py`

**Step 1: Add logger import**

After the server imports (line 22), add:
```python
from logger import log
```

**Step 2: Add startup logging**

In `startup()` (line 35):
```python
@app.on_event("startup")
async def startup():
    """Start inference queue and load model eagerly on worker startup."""
    log.info("Worker starting up...")
    _infer_queue.start()
    await _ensure_model_loaded()
    log.info("Worker ready")
```

**Step 3: Add logging to all worker endpoints**

Same pattern as server.py — each endpoint gets entry/exit/error logging:
- `POST /transcribe`
- `POST /subtitles`
- `POST /translate`
- `POST /transcribe/stream`
- `WS /ws/transcribe`
- `GET /health`

Example for /transcribe:
```python
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
):
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    log.info("POST /transcribe | size={} language={}", len(audio_bytes), language)
    t0 = time.time()
    ...
    log.info("POST /transcribe | completed in {:.2f}s text_len={}", time.time() - t0, len(text))
    return {"text": text, "language": language_code}
```

**Step 4: Run tests**

Run: `pytest E2Etest/ -m "not slow and not performance" -v --timeout=300`
Expected: all pass

**Step 5: Commit**

```bash
git add src/worker.py
git commit -m "feat: add atomic loguru logging to worker.py"
```

---

### Task 5: Add atomic logging to gateway.py

**Files:**
- Modify: `src/gateway.py`

Gateway already has `from logger import log` and 4 log calls. Add request-level logging for every proxied endpoint.

**Step 1: Add logging to every proxy endpoint**

For each gateway endpoint (transcribe, translate, subtitles, stream, websocket, health):

```python
@app.post("/v1/audio/transcriptions", ...)
async def transcribe(...):
    audio_bytes = await file.read()
    log.info("Gateway POST /v1/audio/transcriptions | size={} language={}", len(audio_bytes), language)
    t0 = time.time()
    result = await _proxy_transcribe(audio_bytes, language, return_timestamps)
    log.info("Gateway POST /v1/audio/transcriptions | proxied in {:.2f}s", time.time() - t0)
    return result
```

Also log proxy errors:
```python
if resp.status != 200:
    body = await resp.text()
    log.error("Gateway proxy error | endpoint={} status={} body={}", url, resp.status, body[:200])
```

**Step 2: Add WebSocket proxy logging**

```python
log.info("[GW-WS] Client connected, proxying to worker")
# ... on close:
log.info("[GW-WS] Proxy session ended")
```

**Step 3: Run tests**

Run: `pytest E2Etest/ -m "not slow and not performance" -v --timeout=300`
Expected: all pass

**Step 4: Commit**

```bash
git add src/gateway.py
git commit -m "feat: add atomic request-level logging to gateway.py"
```

---

### Task 6: Add logging to subtitle.py

**Files:**
- Modify: `src/subtitle.py`

Already has `from logger import log` and 3 calls (aligner load/unload). Add generation-level logging.

**Step 1: Add logging to generate_srt_from_results**

At entry:
```python
log.info("SRT generation | mode={} segments={} audio_duration={:.1f}s max_chars={}",
         mode, len(results), len(audio) / sr, max_line_chars)
```

At exit:
```python
log.info("SRT generation complete | events={} srt_len={} elapsed={:.2f}s",
         event_count, len(srt_output), elapsed)
```

**Step 2: Commit**

```bash
git add src/subtitle.py
git commit -m "feat: add SRT generation logging to subtitle.py"
```

---

### Task 7: Clean up .env.example and .env

**Files:**
- Modify: `.env.example`
- Modify: `.env`

**Step 1: Rewrite .env.example with proper structure**

```ini
# ==========================================
# Qwen3-ASR Configuration
# ==========================================
# Copy to .env and adjust as needed.

# -----------------
# Host / Network
# -----------------
# External port exposed on the host machine
PORT=8100

# -----------------
# Core
# -----------------
MODEL_ID=Qwen/Qwen3-ASR-1.7B
IDLE_TIMEOUT=1800
REQUEST_TIMEOUT=300

# -----------------
# Logging
# -----------------
# Log verbosity: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# -----------------
# Subtitle Aligner
# -----------------
# Loaded on first /v1/audio/subtitles?mode=accurate call (~2GB VRAM)
FORCED_ALIGNER_ID=Qwen/Qwen3-ForcedAligner-0.6B

# -----------------
# WebSocket Streaming
# -----------------
# WS_BUFFER_SIZE=14400    # ~450ms at 16kHz mono 16-bit
# WS_OVERLAP_SIZE=4800    # ~150ms overlap between chunks
# WS_FLUSH_SILENCE_MS=600 # Silence padding on flush (ms)

# -----------------
# Translation (OpenAI-compatible API)
# -----------------
# Works with Ollama Cloud, local Ollama, vLLM, or OpenAI.
# Ollama Cloud: https://ollama.com/api
# Local Ollama: http://localhost:11434/v1
# OpenAI:       https://api.openai.com/v1
OPENAI_BASE_URL=https://ollama.com/api
OPENAI_API_KEY=your_api_key_here
TRANSLATE_MODEL=gemma3:12b

# -----------------
# Quantization
# -----------------
# QUANTIZE=int8   # bitsandbytes W8A8 (~50% VRAM reduction)
# QUANTIZE=fp8    # torchao (requires sm_89+ Hopper/Ada)
QUANTIZE=
USE_CUDA_GRAPHS=false

# -----------------
# Dual Model / Speculative
# -----------------
DUAL_MODEL=false
FAST_MODEL_ID=Qwen/Qwen3-ASR-0.6B
USE_SPECULATIVE=false

# -----------------
# Experimental Backends
# -----------------
USE_VLLM=false
USE_CAUSAL_ENCODER=false
ONNX_ENCODER_PATH=
TRT_ENCODER_PATH=

# -----------------
# Infrastructure
# -----------------
GATEWAY_MODE=true
# NUMA_NODE=0
USE_GRANIAN=false
WORKER_HOST=127.0.0.1
WORKER_PORT=8001

# -----------------
# Docker / CUDA Internals
# -----------------
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Step 2: Update .env to match (keep user's current values, add LOG_LEVEL and fix translation section)**

**Step 3: Commit**

```bash
git add .env.example .env
git commit -m "feat: clean up .env.example with LOG_LEVEL, Ollama Cloud defaults, proper docs"
```

---

### Task 8: Rebuild, test, and verify logs

**Step 1: Rebuild container**

```bash
docker compose up -d --build
```

**Step 2: Run E2E tests**

```bash
pytest E2Etest/ -m "not slow and not performance" -v --timeout=300
```
Expected: all 63 pass

**Step 3: Check logs show atomic entries**

```bash
docker compose logs --tail=50
```
Expected: log lines for each endpoint call with timing info

**Step 4: Commit test report**

```bash
git add E2Etest/reports/
git commit -m "test: verify logging overhaul passes all E2E tests"
```

---

### Task 9: Update living documentation

**Files:**
- Modify: `CHANGELOG.md` — add v0.11.0 entry
- Modify: `LEARNING_LOG.md` — add logging insights
- Modify: `CLAUDE.md` — add LOG_LEVEL to env vars table
- Modify: `README.md` — add LOG_LEVEL to configuration table, update translation section

Per `.agent-rules`, all 4 living docs must be updated after each completed issue.

**Step 1: Commit docs**

```bash
git add CHANGELOG.md LEARNING_LOG.md CLAUDE.md README.md
git commit -m "docs: update living docs for v0.11.0 (logging, env cleanup)"
```

**Step 2: Push**

```bash
git push
```
