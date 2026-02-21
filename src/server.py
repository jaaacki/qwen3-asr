from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import sys
import io
import os
import gc
import json
import re
import asyncio
import concurrent.futures
import time
import heapq
import dataclasses
import numpy as np

model = None
_fast_model = None
loaded_model_id = None

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
        print(f"vLLM engine loaded for {model_id}")
    except Exception as e:
        print(f"vLLM load failed: {e} -- falling back to native loader")
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
        print("Silero VAD loaded")
    except ImportError:
        print("silero-vad not installed, VAD disabled")


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
        print(f"TensorRT encoder loaded from {trt_path}")
    except Exception as e:
        print(f"TRT encoder load failed: {e}")


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
            print(f"Causal encoder patch applied (EXPERIMENTAL): {patched_count} attention modules patched")
        else:
            print("Causal encoder patch: no patchable attention modules found")
    except Exception as e:
        print(f"Causal encoder patch failed (non-critical): {e}")

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
            print(f"CPU affinity set to NUMA node {numa_node}: {node_cpus}")
    except Exception as e:
        print(f"CPU affinity setting failed (non-critical): {e}")


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

    print(f"Loading {model_id}...")

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
    print(f"Attention implementation: {attn_impl}")

    if quantize_mode == "int8" and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["torch_dtype"] = torch.float16  # required for bitsandbytes
            print("INT8 quantization enabled (bitsandbytes)")
        except ImportError:
            print("bitsandbytes not available, using default precision")

    model = Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)

    # FP8 post-training quantization (opt-in via QUANTIZE=fp8, requires sm_89+)
    quantize_mode = os.getenv("QUANTIZE", "").lower()
    if quantize_mode == "fp8" and torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability >= (8, 9):  # sm_89+ (Ada/Hopper)
            try:
                from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
                quantize_(model, Float8DynamicActivationFloat8WeightConfig())
                print("FP8 quantization applied (torchao)")
            except Exception as e:
                print(f"FP8 quantization failed: {e}")
        else:
            cc = f"sm_{compute_capability[0]}{compute_capability[1]}"
            print(f"FP8 requires sm_89+, current GPU is {cc} -- skipping")

    # Compile for faster repeated inference (first call will be slower due to compilation)
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile unavailable ({e}), using eager mode")

    # Load fast (draft) model for speculative decoding
    if USE_SPECULATIVE:
        fast_model_id = os.getenv("FAST_MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
        if fast_model_id != model_id:
            print(f"Loading fast model {fast_model_id} for speculative decoding...")
            _fast_model = Qwen3ASRModel.from_pretrained(
                fast_model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        else:
            print("Speculative decoding: main and fast model are the same, skipping dual load")

    model = _patch_encoder_causal(model)

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

    global _PINNED_AUDIO_BUFFER
    if torch.cuda.is_available():
        _PINNED_AUDIO_BUFFER = torch.zeros(
            _PINNED_BUFFER_SIZE, dtype=torch.float32
        ).pin_memory()
        print(f"Pinned memory buffer allocated: {_PINNED_BUFFER_SIZE * 4 / 1024:.0f} KB")

    global _cuda_stream
    if torch.cuda.is_available():
        _cuda_stream = torch.cuda.Stream()
        print("CUDA inference stream created")

    _try_build_cuda_graph()

    _try_load_onnx_encoder()

    _try_load_trt_encoder()

    if os.getenv("DUAL_MODEL", "").lower() == "true" and torch.cuda.is_available():
        try:
            print(f"Loading fast model ({_fast_model_id}) for partial transcriptions...")
            _fast_model = Qwen3ASRModel.from_pretrained(
                _fast_model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            _fast_model.eval()
            print("Dual-model strategy enabled")
        except Exception as e:
            print(f"Fast model load failed: {e}, using single model")

    _last_used = time.time()
    print(f"Model loaded! GPU memory after load:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")

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
        print("CUDA kernel cache warming complete (3 extra passes)")
    except Exception as e:
        print(f"CUDA kernel cache warming failed: {e}")


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
        print(f"ONNX encoder loaded from {onnx_path}")
    except Exception as e:
        print(f"ONNX encoder load failed: {e}")


def _unload_model_sync():
    """Unload model from GPU to free VRAM."""
    import torch
    global model, _fast_model

    if model is None:
        return

    print("Unloading model (idle timeout)...")
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
    _infer_queue.start()
    asyncio.create_task(_idle_watchdog())
    yield
    _infer_executor.shutdown(wait=False)


app = FastAPI(title="Qwen3-ASR API", lifespan=lifespan)


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
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))

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
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = detect_and_fix_repetitions(results[0].text)
        language_code = results[0].language
    else:
        text = ""
        language_code = lang_code or language

    return {"text": text, "language": language_code}


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

    return Response(
        content=srt_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
    )


def _do_transcribe_vllm(audio, sr, lang_code, return_timestamps):
    """Inference via vLLM engine (when USE_VLLM=true)."""
    from vllm import SamplingParams
    params = SamplingParams(temperature=0, max_tokens=448)
    outputs = _vllm_engine.generate({"audio": (audio, sr)}, params)
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

                yield f"data: {json.dumps(data)}\n\n"
                chunk_index += 1

                if is_last:
                    break
                start = end - OVERLAP_SAMPLES  # overlap for context

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
    Buffers audio and transcribes in ~450ms windows with 150ms overlap
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
    # KV-cache state for cross-chunk encoder reuse
    _prev_encoder_out = None

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
                            text, _prev_encoder_out = await _transcribe_with_context(
                                audio_buffer, overlap_buffer, pad_silence=True,
                                lang_code=lang_code,
                                encoder_cache=_prev_encoder_out,
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
                            _prev_encoder_out = None
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
                        text, _prev_encoder_out = await _transcribe_with_context(
                            process_chunk, overlap_buffer, pad_silence=False,
                            lang_code=lang_code,
                            encoder_cache=_prev_encoder_out,
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
                        text, _ = await _transcribe_with_context(
                            audio_buffer, overlap_buffer, pad_silence=True,
                            lang_code=lang_code,
                            encoder_cache=_prev_encoder_out,
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

        sr = TARGET_SR

        # VAD gate: skip inference if no speech detected
        if not is_speech(audio):
            return ""

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
            return detect_and_fix_repetitions(results[0].text), cache_out
        return "", cache_out

    except asyncio.TimeoutError:
        return "[timeout]", None
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"[error: {e}]", None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
