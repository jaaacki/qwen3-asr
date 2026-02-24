"""
Worker process: inference-only, can be killed/restarted to free RAM.
Exposes a simple internal HTTP API on port 8001.

Not intended to be run directly -- started by gateway.py.
Imports core inference logic from server.py to stay in sync with optimizations.
"""
from server import (
    release_gpu_memory,
    _load_model_sync,
    _do_transcribe,
    _idle_watchdog,
    _ensure_model_loaded,
    _infer_queue,
    _transcribe_with_context,
    detect_and_fix_repetitions,
    TARGET_SR,
    REQUEST_TIMEOUT,
    WS_BUFFER_SIZE,
    WS_OVERLAP_SIZE,
)
import server as _srv
from logger import log
import time

import asyncio
import io
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Qwen3-ASR Worker")


@app.on_event("startup")
async def startup():
    """Start inference queue and load model eagerly on worker startup."""
    log.info("Worker starting up...")
    _infer_queue.start()
    await _ensure_model_loaded()
    log.info("Worker ready")


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
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    lang_code = None if language == "auto" else language

    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, return_timestamps),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("POST /transcribe | timed out after {:.2f}s", time.time() - t0)
        release_gpu_memory()
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = detect_and_fix_repetitions(results[0].text)
        language_code = results[0].language
        if return_timestamps and hasattr(results[0], "timestamps") and results[0].timestamps:
            log.info("POST /transcribe | completed in {:.2f}s text_len={}", time.time() - t0, len(text))
            return {"text": text, "language": language_code, "timestamps": results[0].timestamps}
    else:
        text = ""
        language_code = language

    log.info("POST /transcribe | completed in {:.2f}s text_len={}", time.time() - t0, len(text))
    return {"text": text, "language": language_code}


@app.post("/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("accurate"),
    max_line_chars: int = Form(42),
):
    """Subtitle generation -- worker-side endpoint."""
    from fastapi.responses import Response

    if mode not in ("accurate", "fast"):
        return JSONResponse(status_code=400, content={"error": f"Invalid mode: {mode!r}. Must be 'accurate' or 'fast'."})

    await _ensure_model_loaded()

    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    log.info("POST /subtitles | size={} language={} mode={}", len(audio_bytes), language, mode)
    t0 = time.time()

    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        log.error("POST /subtitles | audio decode failed: {}", e)
        return JSONResponse(status_code=422, content={"error": f"Could not decode audio: {e}"})

    lang_code = None if language == "auto" else language

    if mode == "accurate":
        from subtitle import load_aligner
        await asyncio.get_event_loop().run_in_executor(_srv._infer_executor, load_aligner)

    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, False),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("POST /subtitles | timed out after {:.2f}s", time.time() - t0)
        release_gpu_memory()
        return JSONResponse(status_code=504, content={"error": "Subtitle generation timed out"})

    if not results or len(results) == 0:
        return Response(
            content="",
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
        )

    for r in results:
        r.text = detect_and_fix_repetitions(r.text)

    from subtitle import generate_srt_from_results
    srt_content = await asyncio.get_event_loop().run_in_executor(
        _srv._infer_executor,
        lambda: generate_srt_from_results(
            results=results, audio=audio, sr=sr,
            mode=mode, max_line_chars=max_line_chars,
        ),
    )

    log.info("POST /subtitles | completed in {:.2f}s mode={} srt_len={}", time.time() - t0, mode, len(srt_content))
    return Response(
        content=srt_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
    )


@app.post("/translate")
async def translate(
    file: UploadFile = File(...),
    language: str = Form("en"),
    response_format: str = Form("json"),
):
    from fastapi.responses import Response
    from translator import translate_text, translate_srt

    await _ensure_model_loaded()

    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    log.info("POST /translate | size={} target={} format={}", len(audio_bytes), language, response_format)
    t0 = time.time()

    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        log.error("POST /translate | audio decode failed: {}", e)
        return JSONResponse(status_code=422, content={"error": f"Could not decode audio: {e}"})

    target_lang = "en" if language.lower() not in ["en", "zh"] else language.lower()

    if response_format.lower() == "srt":
        from subtitle import load_aligner
        # Lazy load aligner for accurate timing mode
        await asyncio.get_event_loop().run_in_executor(_srv._infer_executor, load_aligner)

        try:
            results = await asyncio.wait_for(
                _infer_queue.submit(
                    lambda: _do_transcribe(audio, sr, None, False),
                    priority=1,
                ),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("POST /translate | timed out after {:.2f}s", time.time() - t0)
            release_gpu_memory()
            return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

        if not results:
            return Response(content="", media_type="text/plain; charset=utf-8")

        for r in results:
            r.text = detect_and_fix_repetitions(r.text)

        from subtitle import generate_srt_from_results
        original_srt = await asyncio.get_event_loop().run_in_executor(
            _srv._infer_executor,
            lambda: generate_srt_from_results(results, audio, sr, mode="accurate", max_line_chars=42),
        )

        try:
            translated_srt = await translate_srt(original_srt, target_lang)
        except Exception as e:
            log.error("POST /translate | translation API failed: {}", e)
            return JSONResponse(status_code=502, content={"error": f"Translation API failed: {e}"})

        log.info("POST /translate | completed in {:.2f}s format={}", time.time() - t0, response_format)
        return Response(content=translated_srt, media_type="text/plain; charset=utf-8")

    else:
        try:
            results = await asyncio.wait_for(
                _infer_queue.submit(
                    lambda: _do_transcribe(audio, sr, None, False),
                    priority=1,
                ),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("POST /translate | timed out after {:.2f}s", time.time() - t0)
            release_gpu_memory()
            return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

        if results and len(results) > 0:
            text = detect_and_fix_repetitions(results[0].text)
        else:
            text = ""

        if text.strip():
            try:
                translated_text = await translate_text(text, target_lang)
            except Exception as e:
                log.error("POST /translate | translation API failed: {}", e)
                return JSONResponse(status_code=502, content={"error": f"Translation API failed: {e}"})
        else:
            translated_text = ""

        log.info("POST /translate | completed in {:.2f}s format={}", time.time() - t0, response_format)
        return {"text": translated_text, "language": target_lang}


@app.post("/transcribe/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
):
    """SSE streaming transcription -- worker-side endpoint."""
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    log.info("POST /transcribe/stream | size={} language={}", len(audio_bytes), language)
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    lang_code = None if language == "auto" else language

    from server import sse_transcribe_generator

    return StreamingResponse(
        sse_transcribe_generator(audio, sr, lang_code, return_timestamps),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket transcription -- worker-side, reuses server.py logic."""
    log.info("[Worker-WS] Proxying WebSocket transcription to server handler")
    from server import websocket_transcribe as _ws_handler
    await _ws_handler(websocket)


@app.get("/health")
async def health():
    log.debug("GET /health")
    return {
        "status": "ok",
        "mode": "worker",
        "model_loaded": _srv.model is not None,
        "model_id": _srv.loaded_model_id,
    }
