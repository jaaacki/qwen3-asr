from __future__ import annotations
from logger import log, set_request_id, reset_request_id
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import sys
import io
import os
import uuid as _uuid_module
import gc
import json
import re
import asyncio
import concurrent.futures
import time
import heapq
import dataclasses
import numpy as np
from scipy.signal import butter, sosfilt

model = None
_fast_model = None
loaded_model_id = None


def _telephony_bandpass(audio: np.ndarray, sr: int) -> np.ndarray:
    """300-3400 Hz bandpass to remove DC and resampling aliasing."""
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)

# Dedicated single-threaded executor for GPU inference
_infer_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="qwen3-asr-infer",
)


@dataclasses.dataclass(order=True)
class _InferJob:
    priority: int          # lower = higher priority (0=WS, 1=HTTP)
    submit_time: float     # tiebreaker
    future: asyncio.Future = dataclasses.field(compare=False)
    fn: object = dataclasses.field(compare=False)


class PriorityInferQueue:
    """Single-worker inference queue with priority scheduling.

    WebSocket requests (priority=0) preempt HTTP file uploads (priority=1)
    so real-time transcription is not blocked by slow batch uploads.
    """

    def __init__(self):
        self._heap: list[_InferJob] = []
        self._lock = asyncio.Lock()
        self._has_work = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

    async def _worker(self):
        loop = asyncio.get_event_loop()
        while True:
            await self._has_work.wait()
            async with self._lock:
                if not self._heap:
                    self._has_work.clear()
                    continue
                job = heapq.heappop(self._heap)
                if not self._heap:
                    self._has_work.clear()
            try:
                result = await loop.run_in_executor(_infer_executor, job.fn)
                job.future.set_result(result)
            except Exception as e:
                job.future.set_exception(e)

    async def submit(self, fn, priority: int = 1):
        """Submit an inference job. Returns the result when complete."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        job = _InferJob(priority=priority, submit_time=time.time(), future=future, fn=fn)
        async with self._lock:
            heapq.heappush(self._heap, job)
            self._has_work.set()
        return await future


_infer_queue = PriorityInferQueue()

# ONNX Runtime session for encoder (opt-in via ONNX_ENCODER_PATH)
_onnx_session = None

# Fast model for partial WS transcriptions (opt-in via DUAL_MODEL=true)
_fast_model = None
_fast_model_id = "Qwen/Qwen3-ASR-0.6B"

# Lock to prevent concurrent load/unload
_model_lock = asyncio.Lock()

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

# Idle unload timeout in seconds (0 = disabled)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))

# Track last request time
_last_used = 0.0

# Target sample rate expected by the model
TARGET_SR = 16000

# Pre-allocated pinned memory buffer for fast CPU→GPU audio transfer
# Sized for 30 seconds of audio at 16kHz float32 = 1.92 MB
_PINNED_AUDIO_BUFFER: torch.Tensor | None = None
_PINNED_BUFFER_SIZE = TARGET_SR * 30  # 480000 samples

# CUDA stream for inference — allows transfer/compute overlap
_cuda_stream: torch.cuda.Stream | None = None

# ── WebSocket streaming config ──────────────────────────────────────────────
# Buffer size: how much audio to accumulate before transcribing (~450ms default)
# At 16kHz 16-bit mono: 450ms = 14400 bytes
WS_BUFFER_SIZE = int(os.getenv("WS_BUFFER_SIZE", str(int(TARGET_SR * 2 * 0.45))))

# Overlap: how much of the previous chunk to prepend to the next (~150ms default)
# Prevents words from being split at chunk boundaries
WS_OVERLAP_SIZE = int(os.getenv("WS_OVERLAP_SIZE", str(int(TARGET_SR * 2 * 0.15))))

# Silence padding appended before final transcription on flush (~600ms)
# Gives the model trailing context to commit the last word
WS_FLUSH_SILENCE_MS = int(os.getenv("WS_FLUSH_SILENCE_MS", "600"))

# Sliding window: max seconds of audio to keep for re-transcription
# Larger = more context = better accuracy, but higher GPU cost per trigger
WS_WINDOW_MAX_S = float(os.getenv("WS_WINDOW_MAX_S", "6.0"))
WS_WINDOW_MAX_BYTES = int(WS_WINDOW_MAX_S * TARGET_SR * 2)  # 16-bit PCM

# Speculative decoding: use 0.6B as draft, 1.7B as verifier
USE_SPECULATIVE = os.getenv("USE_SPECULATIVE", "").lower() == "true"


# vLLM engine (opt-in via USE_VLLM=true)
_vllm_engine = None
USE_VLLM = os.getenv("USE_VLLM", "").lower() == "true"


def _load_vllm_engine(model_id: str):
    """Load model via vLLM engine (opt-in via USE_VLLM=true)."""
    global _vllm_engine
    try:
        from vllm import LLM
        _vllm_engine = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_num_seqs=4,
            enforce_eager=False,
        )
        log.info(f"vLLM engine loaded for {model_id}")
    except Exception as e:
        log.error(f"vLLM load failed: {e} -- falling back to native loader")
        _vllm_engine = None


def release_gpu_memory():
    """Force release of unused GPU memory back to the system."""
    import torch
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


# Silero VAD — loaded lazily at model startup
_vad_model = None


def _load_vad():
    """Load Silero VAD model (CPU, lightweight ~1MB)."""
    global _vad_model
    try:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
        _vad_model.eval()
        log.info("Silero VAD loaded")
    except ImportError:
        log.info("silero-vad not installed, VAD disabled")


def is_speech(audio_float32: np.ndarray, threshold: float = 0.5) -> bool:
    """Return True if audio contains speech (Silero VAD)."""
    if _vad_model is None:
        return True  # fallback: assume speech if VAD not available
    try:
        import torch
        tensor = torch.from_numpy(audio_float32).unsqueeze(0)
        with torch.no_grad():
            confidence = _vad_model(tensor, 16000).item()
        return confidence >= threshold
    except Exception:
        return True  # safe fallback


# TensorRT encoder (opt-in via TRT_ENCODER_PATH env var)
_trt_encoder = None


def _try_load_trt_encoder():
    """Load pre-built TensorRT encoder if available."""
    global _trt_encoder
    trt_path = os.getenv("TRT_ENCODER_PATH", "")
    if not trt_path or not os.path.exists(trt_path):
        return
    try:
        import torch
        _trt_encoder = torch.jit.load(trt_path)
        log.info(f"TensorRT encoder loaded from {trt_path}")
    except Exception as e:
        log.error(f"TRT encoder load failed: {e}")


# Encoder state cache for incremental encoding (session_id -> cached state)
# EXPERIMENTAL: reserved for future incremental streaming; not yet wired into WS handler
_encoder_state_cache: dict[str, object] = {}


def _patch_encoder_causal(model_obj):
    """
    Attempt to patch the Whisper encoder to use causal attention masks.
    This allows incremental encoding without full re-computation.

    EXPERIMENTAL: Requires model architecture to support causal encoder.
    Standard Whisper uses bidirectional attention in encoder.
    """
    if not os.getenv("USE_CAUSAL_ENCODER", "").lower() == "true":
        return model_obj

    try:
        encoder = getattr(model_obj, 'encoder', None) or getattr(
            getattr(model_obj, 'model', None), 'encoder', None
        )
        if encoder is None:
            return model_obj

        patched_count = 0
        for module in encoder.modules():
            if hasattr(module, 'is_causal'):
                module.is_causal = True
                patched_count += 1

        if patched_count > 0:
            log.info(f"Causal encoder patch applied (EXPERIMENTAL): {patched_count} attention modules patched")
        else:
            log.info("Causal encoder patch: no patchable attention modules found")
    except Exception as e:
        log.error(f"Causal encoder patch failed (non-critical): {e}")

    return model_obj


def _set_cpu_affinity():
    """Pin this process to CPUs on NUMA node 0 (collocated with GPU)."""
    numa_node = int(os.getenv("NUMA_NODE", "0"))
    try:
        import psutil
        proc = psutil.Process()
        cpus = proc.cpu_affinity()
        # Simple heuristic: NUMA node 0 = first half of CPUs
        half = max(1, len(cpus) // 2)
        node_cpus = cpus[:half] if numa_node == 0 else cpus[half:]
        if node_cpus:
            proc.cpu_affinity(node_cpus)
            log.info(f"CPU affinity set to NUMA node {numa_node}: {node_cpus}")
    except Exception as e:
        log.error(f"CPU affinity setting failed (non-critical): {e}")


def _load_model_sync():
    """Load model into GPU (blocking). Called from async context via lock."""
    import torch
    from qwen_asr import Qwen3ASRModel

    global model, loaded_model_id, _last_used, _fast_model

    if model is not None:
        return

    _set_cpu_affinity()

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
    loaded_model_id = model_id

    log.info(f"Loading {model_id}...")

    if USE_VLLM:
        _load_vllm_engine(model_id)
        if _vllm_engine is not None:
            _last_used = time.time()
            return

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select best available attention implementation
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    quantize_mode = os.getenv("QUANTIZE", "").lower()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    log.info(f"Attention implementation: {attn_impl}")

    if quantize_mode == "int8" and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["torch_dtype"] = torch.float16  # required for bitsandbytes
            log.info("INT8 quantization enabled (bitsandbytes)")
        except ImportError:
            log.info("bitsandbytes not available, using default precision")

    model = Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)

    # torch.compile investigation: GPU utilization during inference is only ~25%.
    # Bottleneck is Python overhead in HuggingFace generate() loop (~50ms/token × 70 tokens),
    # not GPU compute. torch.compile speeds up GPU kernels but leaves Python overhead untouched
    # → no measurable wall-clock improvement (4.2s compiled vs 3.4s eager).
    # Real speedup requires vLLM backend (USE_VLLM=true) for C++-level decode loop.

    # Load fast (draft) model for speculative decoding
    if USE_SPECULATIVE:
        fast_model_id = os.getenv("FAST_MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
        if fast_model_id != model_id:
            log.info(f"Loading fast model {fast_model_id} for speculative decoding...")
            _fast_model = Qwen3ASRModel.from_pretrained(
                fast_model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        else:
            log.info("Speculative decoding: main and fast model are the same, skipping dual load")

    model = _patch_encoder_causal(model)

    # Warmup inference to trigger CUDA kernel caching
    if torch.cuda.is_available():
        log.info("Warming up GPU...")
        # Use low-amplitude noise (better than silence for CUDA kernel caching)
        rng = np.random.default_rng(seed=42)
        dummy = rng.standard_normal(TARGET_SR).astype(np.float32) * 0.01
        try:
            model.transcribe((dummy, TARGET_SR))
        except Exception:
            pass
        release_gpu_memory()

    # FP8 post-training quantization (opt-in via QUANTIZE=fp8, requires sm_89+)
    # Done AFTER warmup so the from_pretrained() loading-peak (~1.7 GB transient) has
    # fully subsided and release_gpu_memory() has freed reserved pages. This gives
    # ~2.3 GB headroom vs the ~0.8 GB available immediately after loading.
    quantize_mode = os.getenv("QUANTIZE", "").lower()
    if quantize_mode == "fp8" and torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability >= (8, 9):  # sm_89+ (Ada/Hopper)
            try:
                from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
                before_alloc = torch.cuda.memory_allocated() / 1024**2
                before_res   = torch.cuda.memory_reserved()  / 1024**2
                quantize_(model.model, Float8DynamicActivationFloat8WeightConfig())
                after_alloc = torch.cuda.memory_allocated() / 1024**2
                after_res   = torch.cuda.memory_reserved()  / 1024**2
                log.info(f"FP8 quantization applied (torchao) — "
                         f"BF16 allocated={before_alloc:.0f}MB → FP8 allocated={after_alloc:.0f}MB "
                         f"(saved {before_alloc-after_alloc:.0f}MB)")
                # Second warmup: exercise the now-quantized model
                if torch.cuda.is_available():
                    log.info("Warming up FP8 model...")
                    rng2 = np.random.default_rng(seed=43)
                    dummy2 = rng2.standard_normal(TARGET_SR).astype(np.float32) * 0.01
                    try:
                        model.transcribe((dummy2, TARGET_SR))
                    except Exception:
                        pass
                    release_gpu_memory()
            except Exception as e:
                log.error(f"FP8 quantization failed: {e}")
        else:
            cc = f"sm_{compute_capability[0]}{compute_capability[1]}"
            log.info(f"FP8 requires sm_89+, current GPU is {cc} -- skipping")


    global _PINNED_AUDIO_BUFFER
    if torch.cuda.is_available():
        _PINNED_AUDIO_BUFFER = torch.zeros(
            _PINNED_BUFFER_SIZE, dtype=torch.float32
        ).pin_memory()
        log.info(f"Pinned memory buffer allocated: {_PINNED_BUFFER_SIZE * 4 / 1024:.0f} KB")

    global _cuda_stream
    if torch.cuda.is_available():
        _cuda_stream = torch.cuda.Stream()
        log.info("CUDA inference stream created")

    _try_build_cuda_graph()

    _try_load_onnx_encoder()

    _try_load_trt_encoder()

    if os.getenv("DUAL_MODEL", "").lower() == "true" and torch.cuda.is_available():
        try:
            log.info(f"Loading fast model ({_fast_model_id}) for partial transcriptions...")
            _fast_model = Qwen3ASRModel.from_pretrained(
                _fast_model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            _fast_model.eval()
            log.info("Dual-model strategy enabled")
        except Exception as e:
            log.error(f"Fast model load failed: {e}, using single model")

    _last_used = time.time()
    log.info(f"Model loaded! GPU memory after load:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        log.info(f"  Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")

    _load_vad()


def _try_build_cuda_graph():
    """
    CUDA kernel cache warming (opt-in via USE_CUDA_GRAPHS=true).

    Full CUDA graph capture requires fixed-size tensor inputs, but Qwen3-ASR
    uses variable-length audio.  Instead, this runs extra warmup passes so
    that CUDA JIT-compiles and caches the kernels used by the decoder,
    reducing latency on the first real request.
    """
    import torch
    if not torch.cuda.is_available():
        return
    if os.getenv("USE_CUDA_GRAPHS", "").lower() != "true":
        return
    try:
        dummy = np.random.randn(TARGET_SR).astype(np.float32) * 0.01
        for _ in range(3):
            model.transcribe((dummy, TARGET_SR))
        torch.cuda.synchronize()
        log.info("CUDA kernel cache warming complete (3 extra passes)")
    except Exception as e:
        log.error(f"CUDA kernel cache warming failed: {e}")


def _try_load_onnx_encoder():
    """Load ONNX encoder if path is configured (opt-in)."""
    global _onnx_session
    onnx_path = os.getenv("ONNX_ENCODER_PATH", "")
    if not onnx_path or not os.path.exists(onnx_path):
        return
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _onnx_session = ort.InferenceSession(onnx_path, opts, providers=providers)
        log.info(f"ONNX encoder loaded from {onnx_path}")
    except Exception as e:
        log.error(f"ONNX encoder load failed: {e}")


def _unload_model_sync():
    """Unload model from GPU to free VRAM."""
    import torch
    global model, _fast_model

    if model is None:
        return

    log.info("Unloading model (idle timeout)...")
    # Unload ForcedAligner if loaded
    from subtitle import unload_aligner
    unload_aligner()

    if _fast_model is not None:
        del _fast_model
        _fast_model = None
    del model
    model = None
    release_gpu_memory()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        log.info(f"Model unloaded. GPU: Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


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
    _infer_queue.start()
    asyncio.create_task(_idle_watchdog())
    yield
    _infer_executor.shutdown(wait=False)


app = FastAPI(title="Qwen3-ASR API", lifespan=lifespan)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(_uuid_module.uuid4())
    token = set_request_id(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = req_id
        return response
    finally:
        reset_request_id(token)


@app.get("/health")
async def health():
    gpu_info = {}
    # Only query GPU if torch is already imported (model loaded) — avoids
    # importing torch (~2.4GB) just for a health check from a load balancer.
    torch = sys.modules.get("torch")
    cuda_available = torch.cuda.is_available() if torch is not None else False
    if cuda_available and model is not None:
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        }
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": cuda_available,
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
    log.info("POST /v1/audio/transcriptions | file={} size={} language={}", file.filename, len(audio_bytes), language)
    t0 = time.time()
    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        log.error("POST /v1/audio/transcriptions | audio decode failed: {}", e)
        return JSONResponse(status_code=422, content={"error": f"Could not decode audio: {e}"})

    lang_code = None if language == "auto" else language

    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, return_timestamps),
                priority=1,  # HTTP = lower priority
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("POST /v1/audio/transcriptions | timed out after {:.2f}s", time.time() - t0)
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = detect_and_fix_repetitions(results[0].text)
        language_code = results[0].language
    else:
        text = ""
        language_code = lang_code or language

    log.info("POST /v1/audio/transcriptions | completed in {:.2f}s text_len={} lang={}", time.time() - t0, len(text), language_code)
    return {"text": text, "language": language_code}


@app.post("/v1/audio/translations")
async def translate_endpoint(
    file: UploadFile = File(...),
    language: str = Form("en"),
    response_format: str = Form("json"), # 'json' or 'srt'
):
    from fastapi.responses import Response
    from translator import translate_text, translate_srt

    await _ensure_model_loaded()

    audio_bytes = await file.read()
    log.info("POST /v1/audio/translations | file={} size={} target={} format={}", file.filename, len(audio_bytes), language, response_format)
    t0 = time.time()
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    target_lang = "en" if language.lower() not in ["en", "zh"] else language.lower()

    if response_format.lower() == "srt":
        # First generate English/Source SRT
        try:
            results = await asyncio.wait_for(
                _infer_queue.submit(
                    lambda: _do_transcribe(audio, sr, None, False),
                    priority=1,
                ),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("POST /v1/audio/translations | timed out after {:.2f}s", time.time() - t0)
            return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

        if not results:
            return Response(content="", media_type="text/plain; charset=utf-8")

        for r in results:
            r.text = detect_and_fix_repetitions(r.text)

        from subtitle import generate_srt_from_results
        original_srt = await asyncio.get_event_loop().run_in_executor(
            _infer_executor,
            lambda: generate_srt_from_results(results, audio, sr, mode="fast", max_line_chars=42),
        )

        try:
            translated_srt = await translate_srt(original_srt, target_lang)
        except Exception as e:
            log.error("POST /v1/audio/translations | translation API failed in {:.2f}s error={}", time.time() - t0, e)
            return JSONResponse(status_code=502, content={"error": f"Translation API failed: {e}"})

        log.info("POST /v1/audio/translations | completed in {:.2f}s format={}", time.time() - t0, response_format)
        return Response(
            content=translated_srt,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="translated_subtitles.srt"'},
        )

    else:
        # JSON standard text transcription -> translation
        try:
            results = await asyncio.wait_for(
                _infer_queue.submit(
                    lambda: _do_transcribe(audio, sr, None, False),
                    priority=1,
                ),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("POST /v1/audio/translations | timed out after {:.2f}s", time.time() - t0)
            return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

        if results and len(results) > 0:
            text = detect_and_fix_repetitions(results[0].text)
        else:
            text = ""

        if text.strip():
            try:
                translated_text = await translate_text(text, target_lang)
            except Exception as e:
                log.error("POST /v1/audio/translations | translation API failed in {:.2f}s error={}", time.time() - t0, e)
                return JSONResponse(status_code=502, content={"error": f"Translation API failed: {e}"})
        else:
            translated_text = ""

        log.info("POST /v1/audio/translations | completed in {:.2f}s format={}", time.time() - t0, response_format)
        return {"text": translated_text, "language": target_lang}


@app.post("/v1/audio/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("accurate"),
    max_line_chars: int = Form(42),
):
    """Generate SRT subtitles from audio file.

    Modes:
    - accurate: Uses ForcedAligner for word-level timestamps (~33ms accuracy)
    - fast: Heuristic estimation from segment boundaries (no aligner needed)
    """
    from fastapi.responses import Response

    await _ensure_model_loaded()

    audio_bytes = await file.read()
    log.info("POST /v1/audio/subtitles | file={} size={} language={} mode={}", file.filename, len(audio_bytes), language, mode)
    t0 = time.time()
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    lang_code = None if language == "auto" else language

    # Load aligner for accurate mode (lazy, first call only)
    if mode == "accurate":
        from subtitle import load_aligner
        await asyncio.get_event_loop().run_in_executor(_infer_executor, load_aligner)

    # Transcribe
    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, False),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("POST /v1/audio/subtitles | timed out after {:.2f}s", time.time() - t0)
        return JSONResponse(status_code=504, content={"error": "Subtitle generation timed out"})

    if not results or len(results) == 0:
        return Response(
            content="",
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
        )

    # Apply repetition detection
    for r in results:
        r.text = detect_and_fix_repetitions(r.text)

    # Generate SRT
    from subtitle import generate_srt_from_results
    srt_content = await asyncio.get_event_loop().run_in_executor(
        _infer_executor,
        lambda: generate_srt_from_results(
            results=results,
            audio=audio,
            sr=sr,
            mode=mode,
            max_line_chars=max_line_chars,
        ),
    )

    log.info("POST /v1/audio/subtitles | completed in {:.2f}s mode={} srt_len={}", time.time() - t0, mode, len(srt_content))
    return Response(
        content=srt_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
    )


def _do_transcribe_vllm(audio, sr, lang_code, return_timestamps):
    """Inference via vLLM engine (when USE_VLLM=true)."""
    from vllm import SamplingParams
    params = SamplingParams(temperature=0, max_tokens=448)
    audio_input = {"audio": (audio, sr)}
    if lang_code:
        audio_input["language"] = lang_code
    outputs = _vllm_engine.generate(audio_input, params)
    class _Result:
        def __init__(self, text, language):
            self.text = text
            self.language = language
    return [_Result(o.outputs[0].text, lang_code or "auto") for o in outputs]


def _do_transcribe_speculative(audio, sr, lang_code, return_timestamps):
    """
    Speculative decoding: draft with 0.6B, verify with 1.7B.
    Falls back to standard inference if dual models not loaded.
    """
    import torch
    if _fast_model is None or model is None:
        return _do_transcribe(audio, sr, lang_code, return_timestamps)

    with torch.inference_mode():
        draft_result = _fast_model.transcribe(
            (audio, sr), language=lang_code, return_time_stamps=return_timestamps
        )

    # Use draft result if confidence is high (heuristic: short text, no artifacts)
    draft_text = draft_result[0].text if draft_result else ""
    if len(draft_text) < 100 and "[" not in draft_text:
        return draft_result

    # Verify with full model for complex/uncertain outputs
    with torch.inference_mode():
        return model.transcribe(
            (audio, sr), language=lang_code, return_time_stamps=return_timestamps
        )


def _do_transcribe(audio, sr, lang_code, return_timestamps, use_fast=False):
    """Run inference in a thread pool, using ONNX encoder if available."""
    import torch
    if USE_VLLM and _vllm_engine is not None:
        return _do_transcribe_vllm(audio, sr, lang_code, return_timestamps)

    if USE_SPECULATIVE and _fast_model is not None:
        return _do_transcribe_speculative(audio, sr, lang_code, return_timestamps)

    # Use pinned memory buffer for faster CPU→GPU transfer if available.
    if _PINNED_AUDIO_BUFFER is not None and len(audio) <= _PINNED_BUFFER_SIZE:
        _PINNED_AUDIO_BUFFER[:len(audio)].copy_(torch.from_numpy(audio))
        audio = _PINNED_AUDIO_BUFFER[:len(audio)].numpy()

    m = (_fast_model if use_fast and _fast_model is not None else model)
    model_tag = "fast" if (use_fast and _fast_model is not None) else "full"
    audio_duration = len(audio) / sr
    t0 = time.time()
    log.debug("_do_transcribe | model={} audio={:.2f}s lang={} timestamps={}", model_tag, audio_duration, lang_code or "auto", return_timestamps)

    def _run_transcribe():
        return m.transcribe(
            (audio, sr), language=lang_code, return_time_stamps=return_timestamps
        )

    with torch.inference_mode():
        # Route through TRT encoder if loaded (opt-in via TRT_ENCODER_PATH).
        if _trt_encoder is not None and hasattr(m, 'encoder'):
            _orig_fwd = m.encoder.forward
            try:
                def _trt_encoder_fwd(*args, **kwargs):
                    inp = args[0] if args else kwargs.get('input_features')
                    if inp is None:
                        return _orig_fwd(*args, **kwargs)
                    try:
                        out = _trt_encoder(inp)
                        return (out,)
                    except Exception:
                        return _orig_fwd(*args, **kwargs)
                m.encoder.forward = _trt_encoder_fwd
                if _cuda_stream is not None:
                    with torch.cuda.stream(_cuda_stream):
                        results = _run_transcribe()
                    _cuda_stream.synchronize()
                else:
                    results = _run_transcribe()
            finally:
                m.encoder.forward = _orig_fwd
        # Route through ONNX encoder if loaded (opt-in via ONNX_ENCODER_PATH).
        elif _onnx_session is not None and hasattr(m, 'encoder'):
            _orig_fwd = m.encoder.forward
            try:
                def _onnx_fwd(*args, **kwargs):
                    inp = args[0] if args else kwargs.get('input_features')
                    if inp is None:
                        return _orig_fwd(*args, **kwargs)
                    ort_out = _onnx_session.run(
                        None, {"input_features": inp.cpu().numpy()}
                    )
                    return (torch.from_numpy(ort_out[0]).to(inp.device),)
                m.encoder.forward = _onnx_fwd
                if _cuda_stream is not None:
                    with torch.cuda.stream(_cuda_stream):
                        results = _run_transcribe()
                    _cuda_stream.synchronize()
                else:
                    results = _run_transcribe()
            finally:
                m.encoder.forward = _orig_fwd
        elif _cuda_stream is not None:
            with torch.cuda.stream(_cuda_stream):
                results = _run_transcribe()
            _cuda_stream.synchronize()
        else:
            results = _run_transcribe()
    text_len = sum(len(r.text) for r in results) if results else 0
    log.debug("_do_transcribe | done model={} audio={:.2f}s elapsed={:.2f}s text_len={}", model_tag, audio_duration, time.time() - t0, text_len)
    return results


async def sse_transcribe_generator(audio, sr, lang_code, return_timestamps):
    """Generator for Server-Sent Events streaming transcription."""
    audio_duration = len(audio) / sr
    t0 = time.time()
    chunk_count = 0
    log.info("SSE stream | audio={:.2f}s lang={}", audio_duration, lang_code or "auto")
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
                        chunk_count += 1
                        yield f"data: {json.dumps(data)}\n\n"
                log.info("SSE stream | done chunks={} elapsed={:.2f}s", chunk_count, time.time() - t0)
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            except (TypeError, AttributeError, NotImplementedError):
                pass

        # Chunked progressive transcription: yield results as each chunk is processed
        CHUNK_SAMPLES = TARGET_SR * 5  # 5-second chunks
        OVERLAP_SAMPLES = TARGET_SR    # 1-second overlap between chunks

        if len(audio) <= CHUNK_SAMPLES:
            # Short audio: single batch
            results = await _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, return_timestamps),
                priority=1,  # SSE = same as HTTP
            )
            if results and len(results) > 0:
                text = detect_and_fix_repetitions(results[0].text)
                data = {"text": text, "language": results[0].language, "is_final": True}
                if return_timestamps and hasattr(results[0], 'timestamps') and results[0].timestamps:
                    data["timestamps"] = results[0].timestamps
            else:
                data = {"text": "", "language": lang_code or "auto", "is_final": True}
            chunk_count += 1
            yield f"data: {json.dumps(data)}\n\n"
        else:
            # Long audio: process in 5s chunks, stream each result
            start = 0
            chunk_index = 0
            while start < len(audio):
                end = min(start + CHUNK_SAMPLES, len(audio))
                chunk = audio[start:end]
                is_last = (end >= len(audio))

                results = await _infer_queue.submit(
                    lambda c=chunk: _do_transcribe(c, sr, lang_code, return_timestamps),
                    priority=1,  # SSE = same as HTTP
                )

                if results and len(results) > 0:
                    text = detect_and_fix_repetitions(results[0].text)
                    data = {
                        "text": text,
                        "language": results[0].language,
                        "is_final": is_last,
                        "chunk_index": chunk_index,
                    }
                else:
                    data = {"text": "", "language": lang_code or "auto", "is_final": is_last, "chunk_index": chunk_index}

                chunk_count += 1
                yield f"data: {json.dumps(data)}\n\n"
                chunk_index += 1

                if is_last:
                    break
                start = end - OVERLAP_SAMPLES  # overlap for context

        log.info("SSE stream | done chunks={} elapsed={:.2f}s", chunk_count, time.time() - t0)
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        log.error("SSE stream | error after {:.2f}s: {}", time.time() - t0, e)
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
    log.info("POST /v1/audio/transcriptions/stream | file={} size={} language={}", file.filename, len(audio_bytes), language)
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))

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
    Uses an expanding sliding window (up to WS_WINDOW_MAX_S seconds)
    that re-transcribes accumulated audio each trigger for full context.
    Partials are cumulative transcripts — clients should replace, not append.

    Client can send:
    - Binary audio data (raw PCM bytes)
    - JSON: {"action": "flush"} — transcribe full window with silence padding
    - JSON: {"action": "reset"} — clear window and buffer state
    """
    # WS compression disabled via uvicorn --ws websockets (see Dockerfile CMD)
    # per-message-deflate would add ~1ms CPU overhead per frame
    await websocket.accept()
    log.info("[WS] Client connected")

    # Incoming audio accumulator (triggers inference at WS_BUFFER_SIZE)
    audio_buffer = bytearray()
    # Sliding window: all received audio up to WS_WINDOW_MAX_BYTES
    audio_window = bytearray()
    # Language: None = auto-detect, or a code like "en", "zh"
    lang_code: str | None = "English"   # default; overridden by {"action":"config"} message
    # Counter for transcribed windows
    chunk_count = 0
    _ws_prev_had_speech: bool = False

    try:
        await _ensure_model_loaded()

        # Send connection confirmation with config
        await websocket.send_json({
            "status": "connected",
            "sample_rate": TARGET_SR,
            "format": "pcm_s16le",
            "buffer_size": WS_BUFFER_SIZE,
            "window_max_s": WS_WINDOW_MAX_S,
        })

        while True:
            try:
                data = await websocket.receive()

                if data.get("type") == "websocket.disconnect":
                    break

                # ── Control commands (JSON text) ────────────────────────
                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        action = msg.get("action", "")

                        if action == "flush":
                            window_s = len(audio_window) / 2 / TARGET_SR
                            log.debug("[WS] flush | window={:.2f}s", window_s)
                            # Merge any pending audio into window
                            if audio_buffer:
                                audio_window.extend(audio_buffer)
                                audio_buffer.clear()

                            if len(audio_window) > 0:
                                text, _ = await _transcribe_with_context(
                                    bytes(audio_window), b"", pad_silence=True,
                                    lang_code=lang_code,
                                    encoder_cache=None,
                                )
                                chunk_count += 1
                                await websocket.send_json({
                                    "text": text,
                                    "is_partial": False,
                                    "is_final": True,
                                })
                            else:
                                await websocket.send_json({
                                    "text": "",
                                    "is_partial": False,
                                    "is_final": True,
                                })
                            audio_window.clear()

                        elif action == "reset":
                            log.debug("[WS] reset | clearing buffer and window")
                            audio_buffer.clear()
                            audio_window.clear()
                            await websocket.send_json({
                                "status": "buffer_reset"
                            })

                        elif action == "config":
                            new_lang = msg.get("language")
                            if new_lang == "auto":
                                lang_code = None
                            elif new_lang:
                                lang_code = new_lang
                            log.debug("[WS] config | language={}", lang_code or "auto")
                            await websocket.send_json({
                                "status": "configured",
                                "language": lang_code or "auto",
                            })

                        else:
                            log.warning("[WS] unknown action: {!r}", action)
                            await websocket.send_json({
                                "error": f"Unknown action: {action!r}"
                            })

                    except json.JSONDecodeError:
                        log.warning("[WS] invalid JSON command: {!r}", data["text"][:80])
                        await websocket.send_json({
                            "error": "Invalid JSON command"
                        })

                # ── Binary audio data ───────────────────────────────────
                elif "bytes" in data:
                    audio_buffer.extend(data["bytes"])

                    # Trigger when buffer accumulates WS_BUFFER_SIZE of new audio
                    if len(audio_buffer) >= WS_BUFFER_SIZE:
                        # Move new audio into the sliding window
                        audio_window.extend(audio_buffer)
                        audio_buffer.clear()

                        # Trim window if it exceeds the cap
                        if len(audio_window) > WS_WINDOW_MAX_BYTES:
                            trim = len(audio_window) - WS_WINDOW_MAX_BYTES
                            # Align to 2-byte boundary (16-bit PCM)
                            trim = (trim // 2) * 2
                            audio_window = audio_window[trim:]

                        # ── VAD auto-flush ──────────────────────────────────────────────
                        # Check if the tail of the current window has speech
                        _tail_bytes = bytes(audio_window[-WS_BUFFER_SIZE:]) if len(audio_window) >= WS_BUFFER_SIZE else bytes(audio_window)
                        _tail_float = np.frombuffer(_tail_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        _current_has_speech = is_speech(_tail_float)

                        if not _current_has_speech and _ws_prev_had_speech:
                            # End-of-utterance: flush full window with silence padding
                            _ws_prev_had_speech = False
                            text, _ = await _transcribe_with_context(
                                bytes(audio_window), b"", pad_silence=True,
                                lang_code=lang_code,
                                encoder_cache=None,
                            )
                            chunk_count += 1
                            if text:
                                await websocket.send_json({
                                    "text": text,
                                    "is_partial": False,
                                    "is_final": True,
                                })
                            audio_window.clear()
                        else:
                            _ws_prev_had_speech = _current_has_speech
                            # ── existing partial transcription ────
                            text, _ = await _transcribe_with_context(
                                bytes(audio_window), b"", pad_silence=False,
                                lang_code=lang_code,
                                encoder_cache=None,
                            )
                            chunk_count += 1
                            if text:
                                await websocket.send_json({
                                    "text": text,
                                    "is_partial": True,
                                    "is_final": False,
                                })

            except WebSocketDisconnect:
                # Client disconnected — transcribe any remaining audio
                if audio_buffer:
                    audio_window.extend(audio_buffer)
                if len(audio_window) > 0:
                    try:
                        text, _ = await _transcribe_with_context(
                            bytes(audio_window), b"", pad_silence=True,
                            lang_code=lang_code,
                            encoder_cache=None,
                        )
                        chunk_count += 1
                        if text:
                            log.info("[WS] Final transcription on disconnect: {}", text)
                    except Exception:
                        pass
                log.info("[WS] Client disconnected | chunks_processed={}", chunk_count)
                break

    except Exception as e:
        log.error(f"WebSocket error: {e}")
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
    encoder_cache=None,
) -> tuple[str, object]:
    """
    Transcribe audio with optional overlap prefix and silence padding.

    Args:
        audio_bytes: Current chunk of PCM 16-bit audio
        overlap: Tail of the previous chunk (prepended for context)
        pad_silence: If True, append silence to help the model commit trailing words
        lang_code: Language code (e.g. "en", "zh") or None for auto-detect
        encoder_cache: Previous encoder output for KV-cache reuse (or None)

    Returns:
        Tuple of (transcribed text, new encoder cache for next chunk)
    """
    audio_duration = len(audio_bytes) / 2 / TARGET_SR  # 16-bit PCM = 2 bytes per sample
    t0 = time.time()
    log.debug("_transcribe_with_context | audio={:.2f}s pad_silence={} lang={}", audio_duration, pad_silence, lang_code or "auto")
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
            return "", None

        # Convert to numpy float32
        audio = np.frombuffer(full_audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        # Telephony bandpass: remove DC offset and 4-8 kHz aliasing from 8->16 kHz resample
        audio = _telephony_bandpass(audio, TARGET_SR)

        sr = TARGET_SR

        # VAD gate: skip inference if no speech detected
        if not is_speech(audio):
            log.info("_transcribe_with_context | VAD: silence, skipping inference")
            return "", None

        # Run inference via priority queue (WS = priority 0)
        # Use fast model for partials, full model for flush
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, False, use_fast=not pad_silence),
                priority=0,  # WebSocket = higher priority
            ),
            timeout=REQUEST_TIMEOUT,
        )

        cache_out = None
        try:
            if hasattr(results, 'encoder_last_hidden_state'):
                cache_out = results.encoder_last_hidden_state
        except Exception:
            pass

        if results and len(results) > 0:
            text = detect_and_fix_repetitions(results[0].text)
            log.info("_transcribe_with_context | done elapsed={:.2f}s text_len={} text={!r}", time.time() - t0, len(text), text[:80])
            return text, cache_out
        return "", cache_out

    except asyncio.TimeoutError:
        log.warning("_transcribe_with_context | timed out after {:.2f}s audio={:.2f}s", time.time() - t0, audio_duration)
        return "[timeout]", None
    except Exception as e:
        log.error("_transcribe_with_context | error after {:.2f}s: {}", time.time() - t0, e)
        return f"[error: {e}]", None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
