from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import torch
import soundfile as sf
import io
import os
import gc
import json
import re
import asyncio
import concurrent.futures
import time
import numpy as np
from qwen_asr import Qwen3ASRModel, parse_asr_output
from qwen_asr.inference.qwen3_asr import AutoProcessor

def _get_attn_implementation() -> str:
    """Select best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"

_ATTN_IMPL = _get_attn_implementation()

model = None
processor = None
loaded_model_id = None

# Semaphore to serialize GPU inference — prevents OOM with concurrent requests
_infer_semaphore = asyncio.Semaphore(1)

# Lock to prevent concurrent load/unload
_model_lock = asyncio.Lock()

# Dedicated single-threaded executor for GPU inference
# Ensures inference always runs on the same OS thread (better GPU context affinity)
_infer_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="qwen3-asr-infer",
)

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

# Idle unload timeout in seconds (0 = disabled)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))

# Track last request time
_last_used = 0.0

# Target sample rate expected by the model
TARGET_SR = 16000

# ── WebSocket streaming config ──────────────────────────────────────────────
# Buffer size: how much audio to accumulate before transcribing (~800ms default)
# At 16kHz 16-bit mono: 800ms = 25600 bytes
WS_BUFFER_SIZE = int(os.getenv("WS_BUFFER_SIZE", str(int(TARGET_SR * 2 * 0.8))))

# Overlap: how much of the previous chunk to prepend to the next (~300ms default)
# Prevents words from being split at chunk boundaries
WS_OVERLAP_SIZE = int(os.getenv("WS_OVERLAP_SIZE", str(int(TARGET_SR * 2 * 0.3))))

# Silence padding appended before final transcription on flush (~600ms)
# Gives the model trailing context to commit the last word
WS_FLUSH_SILENCE_MS = int(os.getenv("WS_FLUSH_SILENCE_MS", "600"))


def release_gpu_memory():
    """Force release of unused GPU memory back to the system."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def detect_and_fix_repetitions(text: str, max_repeats: int = 2) -> str:
    """Remove pathological repetitions from ASR output."""
    if not text or len(text) < 10:
        return text

    # Pattern 1: repeated single words (e.g. "um um um um")
    text = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', text)

    # Pattern 2: repeated short phrases (3-8 words, repeating 3+ times)
    words = text.split()
    for phrase_len in range(3, min(9, len(words) // 3 + 1)):
        i = 0
        result = []
        while i < len(words):
            phrase = words[i:i + phrase_len]
            count = 1
            j = i + phrase_len
            while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                count += 1
                j += phrase_len
            result.extend(phrase)
            if count > max_repeats:
                i = j  # skip the extra repeats
            else:
                i += phrase_len
        words = result

    return ' '.join(words)


def preprocess_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Preprocess audio for optimal inference:
    - Convert to mono
    - Convert to float32
    - Resample to 16kHz if needed
    - Normalize peak amplitude
    """
    # Convert to mono if stereo/multi-channel
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Resample to 16kHz if needed (torchaudio is bundled with PyTorch)
    if sr != TARGET_SR:
        import torchaudio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        audio = resampler(audio_tensor).squeeze(0).numpy()
        sr = TARGET_SR

    # Peak normalize to [-1, 1] — improves feature extraction on quiet audio
    peak = np.abs(audio).max()
    if peak > 0 and peak != 1.0:
        audio = audio / peak

    return audio, sr


def preprocess_audio_ws(audio: np.ndarray) -> np.ndarray:
    """Fast path for WebSocket PCM: already mono, float32, at 16kHz. Only normalize."""
    peak = np.abs(audio).max()
    if peak > 0 and peak != 1.0:
        audio = audio / peak
    return audio


def _load_model_sync():
    """Load model into GPU (blocking). Called from async context via lock."""
    global model, processor, loaded_model_id, _last_used

    if model is not None:
        return

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
    loaded_model_id = model_id

    print(f"Loading {model_id}...")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=_ATTN_IMPL,
    )
    model.eval()
    print(f"Attention implementation: {_ATTN_IMPL}")

    # Compile for faster repeated inference (first call will be slower due to compilation)
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile unavailable ({e}), using eager mode")

    # Warmup inference to trigger CUDA kernel caching
    if torch.cuda.is_available():
        print("Warming up GPU...")
        # Use low-amplitude noise (better than silence for CUDA kernel caching)
        rng = np.random.default_rng(seed=42)
        dummy = rng.standard_normal(TARGET_SR).astype(np.float32) * 0.01
        try:
            model.transcribe((dummy, TARGET_SR))
        except Exception:
            pass
        release_gpu_memory()

    _last_used = time.time()
    print(f"Model loaded! GPU memory after load:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


def _unload_model_sync():
    """Unload model from GPU to free VRAM."""
    global model, processor

    if model is None:
        return

    print("Unloading model (idle timeout)...")
    del model
    del processor
    model = None
    processor = None
    release_gpu_memory()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Model unloaded. GPU: Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


async def _ensure_model_loaded():
    """Load model if not already loaded. Thread-safe via lock."""
    global _last_used
    if model is not None:
        _last_used = time.time()
        return
    async with _model_lock:
        if model is not None:
            _last_used = time.time()
            return
        await asyncio.get_event_loop().run_in_executor(_infer_executor, _load_model_sync)
        _last_used = time.time()


async def _idle_watchdog():
    """Background task that unloads model after IDLE_TIMEOUT seconds of inactivity."""
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0 or model is None:
            continue
        if time.time() - _last_used > IDLE_TIMEOUT:
            async with _model_lock:
                if model is not None and time.time() - _last_used > IDLE_TIMEOUT:
                    await asyncio.get_event_loop().run_in_executor(_infer_executor, _unload_model_sync)


@asynccontextmanager
async def lifespan(the_app):
    """ASGI lifespan handler — compatible with both uvicorn and granian."""
    asyncio.create_task(_idle_watchdog())
    yield
    _infer_executor.shutdown(wait=False)


app = FastAPI(title="Qwen3-ASR API", lifespan=lifespan)


@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        }
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": torch.cuda.is_available(),
        **gpu_info,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    await _ensure_model_loaded()

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio, sr = preprocess_audio(audio, sr)

    lang_code = None if language == "auto" else language

    try:
        async with _infer_semaphore:
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    _infer_executor,
                    lambda: _do_transcribe(audio, sr, lang_code, return_timestamps)
                ),
                timeout=REQUEST_TIMEOUT
            )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = detect_and_fix_repetitions(results[0].text)
        language_code = results[0].language
        if return_timestamps and hasattr(results[0], 'timestamps') and results[0].timestamps:
            return {"text": text, "language": language_code, "timestamps": results[0].timestamps}
    else:
        text = ""
        language_code = language

    return {"text": text, "language": language_code}


def _do_transcribe(audio, sr, lang_code, return_timestamps):
    """Run inference in a thread pool."""
    with torch.inference_mode():
        results = model.transcribe(
            (audio, sr),
            language=lang_code,
            return_time_stamps=return_timestamps
        )
    return results


async def sse_transcribe_generator(audio, sr, lang_code, return_timestamps):
    """Generator for Server-Sent Events streaming transcription."""
    try:
        if hasattr(model, 'transcribe_stream') or hasattr(model, 'stream_transcribe'):
            stream_method = getattr(model, 'transcribe_stream', None) or getattr(model, 'stream_transcribe', None)
            try:
                for partial_result in stream_method(
                    (audio, sr),
                    language=lang_code,
                    return_time_stamps=return_timestamps
                ):
                    if hasattr(partial_result, 'text'):
                        data = {
                            "text": detect_and_fix_repetitions(partial_result.text),
                            "language": getattr(partial_result, 'language', lang_code or 'auto'),
                            "is_final": getattr(partial_result, 'is_final', False)
                        }
                        if return_timestamps and hasattr(partial_result, 'timestamps') and partial_result.timestamps:
                            data["timestamps"] = partial_result.timestamps
                        yield f"data: {json.dumps(data)}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            except (TypeError, AttributeError, NotImplementedError):
                pass

        # Fallback: run full transcription via thread pool
        results = await asyncio.get_event_loop().run_in_executor(
            _infer_executor,
            lambda: _do_transcribe(audio, sr, lang_code, return_timestamps)
        )

        if results and len(results) > 0:
            text = detect_and_fix_repetitions(results[0].text)
            language_code = results[0].language
            data = {
                "text": text,
                "language": language_code,
                "is_final": True
            }
            if return_timestamps and hasattr(results[0], 'timestamps') and results[0].timestamps:
                data["timestamps"] = results[0].timestamps
        else:
            data = {
                "text": "",
                "language": lang_code or "auto",
                "is_final": True
            }

        yield f"data: {json.dumps(data)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    """Streaming transcription endpoint using Server-Sent Events (SSE)."""
    await _ensure_model_loaded()

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio, sr = preprocess_audio(audio, sr)

    lang_code = None if language == "auto" else language

    return StreamingResponse(
        sse_transcribe_generator(audio, sr, lang_code, return_timestamps),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription.

    Accepts binary audio frames (PCM 16-bit, 16kHz mono).
    Buffers audio and transcribes in ~800ms windows with 300ms overlap
    between consecutive chunks to prevent word splits at boundaries.

    Client can send:
    - Binary audio data (raw PCM bytes)
    - JSON: {"action": "flush"} — transcribe remaining buffer with silence padding
    - JSON: {"action": "reset"} — clear buffer and overlap state
    """
    # WS compression disabled via uvicorn --ws websockets (see Dockerfile CMD)
    # per-message-deflate would add ~1ms CPU overhead per frame
    await websocket.accept()

    # Audio buffer for accumulating incoming chunks
    audio_buffer = bytearray()
    # Overlap: tail of previous chunk, prepended to next for acoustic context
    overlap_buffer = bytearray()
    # Language: None = auto-detect, or a code like "en", "zh"
    lang_code: str | None = None

    try:
        await _ensure_model_loaded()

        # Send connection confirmation with config
        await websocket.send_json({
            "status": "connected",
            "sample_rate": TARGET_SR,
            "format": "pcm_s16le",
            "buffer_size": WS_BUFFER_SIZE,
            "overlap_size": WS_OVERLAP_SIZE,
        })

        while True:
            try:
                data = await websocket.receive()

                # ── Control commands (JSON text) ────────────────────────
                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        action = msg.get("action", "")

                        if action == "flush" and len(audio_buffer) > 0:
                            text = await _transcribe_with_context(
                                audio_buffer, overlap_buffer, pad_silence=True,
                                lang_code=lang_code,
                            )
                            await websocket.send_json({
                                "text": text,
                                "is_partial": False,
                                "is_final": True,
                            })
                            audio_buffer.clear()
                            overlap_buffer.clear()

                        elif action == "flush" and len(audio_buffer) == 0:
                            # Nothing to flush — send empty final
                            await websocket.send_json({
                                "text": "",
                                "is_partial": False,
                                "is_final": True,
                            })

                        elif action == "reset":
                            audio_buffer.clear()
                            overlap_buffer.clear()
                            await websocket.send_json({
                                "status": "buffer_reset"
                            })

                        elif action == "config":
                            new_lang = msg.get("language")
                            if new_lang == "auto":
                                lang_code = None
                            elif new_lang:
                                lang_code = new_lang
                            await websocket.send_json({
                                "status": "configured",
                                "language": lang_code or "auto",
                            })

                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "error": "Invalid JSON command"
                        })

                # ── Binary audio data ───────────────────────────────────
                elif "bytes" in data:
                    audio_buffer.extend(data["bytes"])

                    # Process when buffer reaches target size
                    if len(audio_buffer) >= WS_BUFFER_SIZE:
                        # Take exactly WS_BUFFER_SIZE bytes (even-aligned for 16-bit)
                        chunk_size = (WS_BUFFER_SIZE // 2) * 2
                        process_chunk = bytes(audio_buffer[:chunk_size])
                        audio_buffer = audio_buffer[chunk_size:]

                        # Transcribe with overlap from previous chunk
                        text = await _transcribe_with_context(
                            process_chunk, overlap_buffer, pad_silence=False,
                            lang_code=lang_code,
                        )

                        # Save tail of this chunk as overlap for next
                        overlap_len = min(WS_OVERLAP_SIZE, len(process_chunk))
                        overlap_buffer = bytearray(process_chunk[-overlap_len:])

                        if text:
                            await websocket.send_json({
                                "text": text,
                                "is_partial": True,
                                "is_final": False,
                            })

            except WebSocketDisconnect:
                # Client disconnected — transcribe any remaining audio
                if len(audio_buffer) > 0:
                    try:
                        text = await _transcribe_with_context(
                            audio_buffer, overlap_buffer, pad_silence=True,
                            lang_code=lang_code,
                        )
                        if text:
                            print(f"[WS] Final transcription on disconnect: {text}")
                    except Exception:
                        pass
                print("WebSocket client disconnected")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _transcribe_with_context(
    audio_bytes: bytes | bytearray,
    overlap: bytes | bytearray,
    pad_silence: bool = False,
    lang_code: str | None = None,
) -> str:
    """
    Transcribe audio with optional overlap prefix and silence padding.

    Args:
        audio_bytes: Current chunk of PCM 16-bit audio
        overlap: Tail of the previous chunk (prepended for context)
        pad_silence: If True, append silence to help the model commit trailing words
        lang_code: Language code (e.g. "en", "zh") or None for auto-detect
    """
    try:
        # Build the full audio: [overlap] + [current chunk] + [optional silence]
        full_audio = bytearray()
        if overlap:
            full_audio.extend(overlap)
        full_audio.extend(audio_bytes)

        if pad_silence:
            silence_bytes = int((WS_FLUSH_SILENCE_MS / 1000) * TARGET_SR * 2)
            full_audio.extend(bytes(silence_bytes))

        if len(full_audio) == 0:
            return ""

        # Convert to numpy float32
        audio = np.frombuffer(full_audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

        # Fast path: WS audio is already mono, float32, at 16kHz
        audio = preprocess_audio_ws(audio)
        sr = TARGET_SR

        # Run inference
        async with _infer_semaphore:
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    _infer_executor,
                    lambda: _do_transcribe(audio, sr, lang_code, False)
                ),
                timeout=REQUEST_TIMEOUT,
            )

        if results and len(results) > 0:
            return detect_and_fix_repetitions(results[0].text)
        return ""

    except asyncio.TimeoutError:
        return "[timeout]"
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"[error: {e}]"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
