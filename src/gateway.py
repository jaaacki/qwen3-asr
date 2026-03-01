"""
Gateway process: handles HTTP/WS routing and worker lifecycle.
Routes inference requests to the worker via HTTP on an internal port.

Usage: GATEWAY_MODE=true in compose.yaml environment.
The gateway starts on port 8000 (external) and spawns a worker on port 8001 (internal).
"""
from logger import log, set_request_id, reset_request_id, get_request_id
from errors import error_response

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid as _uuid_module
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

WORKER_HOST = os.getenv("WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

_worker_proc: subprocess.Popen | None = None
_last_used = time.time()
_worker_lock = asyncio.Lock()


def _check_vram_available(min_free_mb: int = 3500) -> tuple[bool, int]:
    """Return (ok, free_mb) — check GPU has enough free VRAM to load the model."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        free_mb = int(result.stdout.strip())
        return free_mb >= min_free_mb, free_mb
    except Exception:
        return True, -1  # can't check — optimistically proceed


async def _ensure_worker():
    """Start worker process if not running."""
    global _worker_proc, _last_used
    async with _worker_lock:
        if _worker_proc is None or _worker_proc.poll() is not None:
            ok, free_mb = _check_vram_available()
            if not ok:
                log.error("Not enough VRAM to start worker: {}MB free, need ~3500MB", free_mb)
                raise RuntimeError(f"Insufficient VRAM: {free_mb}MB free")
            log.info("Starting worker process... (VRAM free: {}MB)", free_mb)
            _worker_proc = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "worker:app",
                "--host", WORKER_HOST, "--port", str(WORKER_PORT),
                "--ws", "websockets",
            ])
            # Wait for worker to be ready
            for attempt in range(30):
                await asyncio.sleep(1)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://{WORKER_HOST}:{WORKER_PORT}/health"
                        ) as resp:
                            if resp.status == 200:
                                log.info("Worker process ready")
                                break
                except Exception:
                    continue
            else:
                log.error("Worker process failed to become ready after 30s")
        _last_used = time.time()


async def _kill_worker():
    """Kill worker process to free RAM/VRAM."""
    global _worker_proc
    async with _worker_lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            log.info("Killing worker process (idle timeout)...")
            _worker_proc.terminate()
            try:
                _worker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _worker_proc.kill()
            _worker_proc = None
            log.info("Worker process killed -- RAM reclaimed")


async def _idle_watchdog():
    """Kill worker after idle timeout to reclaim RAM."""
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0:
            continue
        if _worker_proc is not None and _worker_proc.poll() is None:
            if time.time() - _last_used > IDLE_TIMEOUT:
                await _kill_worker()


@asynccontextmanager
async def lifespan(app):
    from config import validate_env
    validate_env()
    asyncio.create_task(_idle_watchdog())
    if IDLE_TIMEOUT == 0:
        log.info("Always-on mode: pre-spawning worker at startup")
        try:
            await _ensure_worker()
        except Exception as e:
            log.warning("Worker pre-spawn failed (will retry on first request): {}", e)
    yield
    await _kill_worker()

from schemas import (
    HealthResponse, TranscriptionResponse, TranslationResponse,
    ErrorResponse, API_TAGS, API_DESCRIPTION,
)

app = FastAPI(
    title="Qwen3-ASR",
    version="0.14.0",
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Audio decode or validation error"},
        504: {"model": ErrorResponse, "description": "Inference timed out"},
    },
)


def _trace_headers() -> dict:
    """Build headers dict with current requestId for worker requests."""
    req_id = get_request_id()
    return {"X-Request-ID": req_id} if req_id else {}


async def _proxy_error_or_raise(resp: aiohttp.ClientResponse, url: str):
    """Handle non-200 response from worker. Returns an error JSONResponse."""
    body = await resp.text()
    log.error("Gateway proxy error | url={} status={}", url, resp.status)
    try:
        worker_error = json.loads(body)
        if "code" in worker_error:
            return JSONResponse(status_code=resp.status, content=worker_error)
    except (json.JSONDecodeError, KeyError):
        pass
    return error_response("WORKER_ERROR", body, resp.status)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Generate requestId for every incoming request and set in log context."""
    req_id = str(_uuid_module.uuid4())
    token = set_request_id(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        reset_request_id(token)


async def _proxy_transcribe(audio_bytes: bytes, language: str, return_timestamps: bool) -> dict:
    """Forward transcription request to worker via HTTP."""
    global _last_used
    await _ensure_worker()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/transcribe"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("return_timestamps", str(return_timestamps).lower())
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, headers=_trace_headers(), timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            if resp.status != 200:
                return await _proxy_error_or_raise(resp, url)
            return await resp.json()


@app.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="Transcribe audio file",
    description="Upload an audio file and get the transcribed text back. Supports WAV, FLAC, MP3, OGG, and other formats. Language is auto-detected by default.",
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (WAV, FLAC, MP3, OGG, AIFF, etc.)"),
    language: str = Form("auto", description="Language code (e.g. 'en', 'zh', 'ja'). Use 'auto' for detection."),
    return_timestamps: bool = Form(False, description="Include word-level timestamps in the output text"),
):
    audio_bytes = await file.read()
    log.info("Gateway POST /v1/audio/transcriptions | size={} language={}", len(audio_bytes), language)
    t0 = time.time()
    result = await _proxy_transcribe(audio_bytes, language, return_timestamps)
    log.info("Gateway POST /v1/audio/transcriptions | proxied in {:.2f}s", time.time() - t0)
    return result


@app.post(
    "/v1/audio/translations",
    tags=["Translation"],
    summary="Translate audio file",
    description="Transcribe audio and translate the text into English or Chinese using an external LLM. Returns JSON by default, or SRT subtitles with `response_format=srt`.",
)
async def translate(
    file: UploadFile = File(..., description="Audio file to transcribe and translate"),
    language: str = Form("en", description="Target language: 'en' (English) or 'zh' (Chinese)"),
    response_format: str = Form("json", description="Response format: 'json' (text) or 'srt' (subtitles)"),
):
    """Proxy translation request to worker."""
    from fastapi.responses import Response
    global _last_used
    await _ensure_worker()

    content = await file.read()
    log.info("Gateway POST /v1/audio/translations | size={} target={} format={}", len(content), language, response_format)
    t0 = time.time()

    # Restrict to en/zh only
    target_lang = "en" if language.lower() not in ["en", "zh"] else language.lower()

    url = f"http://{WORKER_HOST}:{WORKER_PORT}/translate"
    form = aiohttp.FormData()
    form.add_field("file", content, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", target_lang)
    form.add_field("response_format", response_format)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, headers=_trace_headers(), timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            if resp.status != 200:
                return await _proxy_error_or_raise(resp, url)

            log.info("Gateway POST /v1/audio/translations | proxied in {:.2f}s", time.time() - t0)
            if response_format.lower() == "srt":
                srt_content = await resp.text()
                return Response(
                    content=srt_content,
                    media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="translated_subtitles.srt"'},
                )
            else:
                return await resp.json()


@app.post(
    "/v1/audio/subtitles",
    tags=["Subtitles"],
    summary="Generate SRT subtitles",
    description="Generate SRT subtitle file from audio. **fast** mode uses heuristic timestamps (no extra model). **accurate** mode uses ForcedAligner for word-level timing (~33ms accuracy, requires ~6GB VRAM).",
    responses={200: {"content": {"text/plain": {}}, "description": "SRT subtitle file"}},
)
async def generate_subtitles(
    file: UploadFile = File(..., description="Audio file to generate subtitles from"),
    language: str = Form("auto", description="Language code (e.g. 'en', 'zh'). Use 'auto' for detection."),
    mode: str = Form("accurate", description="Subtitle mode: 'fast' (heuristic) or 'accurate' (ForcedAligner)"),
    max_line_chars: int = Form(42, description="Maximum characters per subtitle line before wrapping"),
):
    from fastapi.responses import Response

    global _last_used
    await _ensure_worker()
    audio_bytes = await file.read()
    log.info("Gateway POST /v1/audio/subtitles | size={} language={} mode={}", len(audio_bytes), language, mode)
    t0 = time.time()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/subtitles"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("mode", mode)
    form.add_field("max_line_chars", str(max_line_chars))
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, headers=_trace_headers(), timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            if resp.status != 200:
                return await _proxy_error_or_raise(resp, url)
            srt_content = await resp.text()
            log.info("Gateway POST /v1/audio/subtitles | proxied in {:.2f}s", time.time() - t0)
            return Response(
                content=srt_content,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
            )


@app.post(
    "/v1/audio/transcriptions/stream",
    tags=["Streaming"],
    summary="Stream transcription (SSE)",
    description="Upload a long audio file and receive transcription results as Server-Sent Events. Audio is split at silence boundaries and each chunk is transcribed progressively. Useful for real-time feedback on long files.",
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream of transcription chunks"}},
)
async def transcribe_stream(
    file: UploadFile = File(..., description="Audio file to transcribe in streaming mode"),
    language: str = Form("auto", description="Language code (e.g. 'en', 'zh'). Use 'auto' for detection."),
    return_timestamps: bool = Form(False, description="Include word-level timestamps in each chunk"),
):
    global _last_used
    await _ensure_worker()
    audio_bytes = await file.read()
    log.info("Gateway POST /v1/audio/transcriptions/stream | size={} language={}", len(audio_bytes), language)
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/transcribe/stream"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("return_timestamps", str(return_timestamps).lower())

    t0 = time.time()

    trace_hdrs = _trace_headers()

    async def stream_from_worker():
        chunk_count = 0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form, headers=trace_hdrs, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                    async for line in resp.content:
                        _last_used = time.time()
                        chunk_count += 1
                        yield line
            log.info("Gateway POST /v1/audio/transcriptions/stream | done chunks={} elapsed={:.2f}s", chunk_count, time.time() - t0)
        except Exception as e:
            log.error("Gateway POST /v1/audio/transcriptions/stream | error after {:.2f}s: {}", time.time() - t0, e)

    return StreamingResponse(
        stream_from_worker(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/transcribe")
async def websocket_proxy(websocket: WebSocket):
    """Proxy WebSocket transcription to worker."""
    global _last_used
    await websocket.accept()

    ws_req_id = str(_uuid_module.uuid4())
    token = set_request_id(ws_req_id)
    log.info("[GW-WS] Client connected, proxying to worker")

    try:
        await _ensure_worker()
    except Exception as e:
        await websocket.send_json({"code": "WORKER_STARTUP_FAILED", "message": f"Worker startup failed: {e}", "statusCode": 503})
        await websocket.close()
        reset_request_id(token)
        return

    # Forward query params to worker (e.g. use_server_vad, sample_rate)
    qs_parts = [f"request_id={ws_req_id}"]
    vad_param = websocket.query_params.get("use_server_vad")
    if vad_param is not None:
        qs_parts.append(f"use_server_vad={vad_param}")
    sr_param = websocket.query_params.get("sample_rate")
    if sr_param is not None:
        qs_parts.append(f"sample_rate={sr_param}")
    ws_url = f"ws://{WORKER_HOST}:{WORKER_PORT}/ws/transcribe?{'&'.join(qs_parts)}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as worker_ws:
                # Read the worker connection confirmation and forward it
                init_msg = await worker_ws.receive_json()
                await websocket.send_json(init_msg)

                async def client_to_worker():
                    """Forward client messages to worker."""
                    try:
                        while True:
                            data = await websocket.receive()
                            if data.get("type") == "websocket.disconnect":
                                await worker_ws.close()
                                break
                            _last_used = time.time()
                            if "text" in data:
                                await worker_ws.send_str(data["text"])
                            elif "bytes" in data:
                                await worker_ws.send_bytes(data["bytes"])
                    except WebSocketDisconnect:
                        await worker_ws.close()
                    except Exception as e:
                        log.warning("[GW-WS] client_to_worker error: {}", e)

                async def worker_to_client():
                    """Forward worker messages to client."""
                    try:
                        async for msg in worker_ws:
                            _last_used = time.time()
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await websocket.send_text(msg.data)
                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                await websocket.send_bytes(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                    except Exception as e:
                        log.warning("[GW-WS] worker_to_client error: {}", e)

                await asyncio.gather(client_to_worker(), worker_to_client(), return_exceptions=True)

    except Exception as e:
        try:
            await websocket.send_json({"code": "WORKER_CONNECTION_FAILED", "message": f"Worker connection failed: {e}", "statusCode": 502})
        except Exception:
            pass
    finally:
        log.info("[GW-WS] Proxy session ended")
        reset_request_id(token)
        try:
            await websocket.close()
        except Exception:
            pass


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns service status, model loading state, GPU info, and worker process status. Use this for load balancer health checks.",
)
async def health():
    worker_alive = _worker_proc is not None and _worker_proc.poll() is None
    log.debug("Gateway GET /health | worker_alive={}", worker_alive)
    info = {"status": "ok", "mode": "gateway", "worker_alive": worker_alive, "model_loaded": False, "model_id": None}
    if worker_alive:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{WORKER_HOST}:{WORKER_PORT}/health",
                    headers=_trace_headers(),
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        info.update(await resp.json())
        except Exception:
            pass
    return info
