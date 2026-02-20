# Qwen3-ASR Performance & Latency Improvements

Based on a deep scan of the current `qwen3-asr` server architecture (`server.py`) and comparisons against the official framework, I have identified several critical bottlenecks causing poor STT (Speech-to-Text) quality and high latency. 

Here is a summary of the issues and the recommended architectural improvements:

| Improvement | Area / Aspect | Expected Gains |
|-------------|---------------|----------------|
| **Upgrade to Qwen3-ASR-1.7B Model** | STT Quality | ~35% improvement in multilingual accuracy, ~25% better English accuracy. |
| **Repetition Detection & Fixing** | STT Quality | Eliminates hallucinatory repeated phrases ("the the the") on noisy audio. |
| **Long Audio Chunking** | STT Quality & Stability | Prevents OOM crashes and degraded context on audio >30s. |
| **Enable `torch.compile`** | Latency / Speed | 20-30% faster inference, reducing WebSocket turnaround latency. |
| **Switch to Flash Attention 2** | Latency / Speed | ~20% faster attention computation. |
| **Enable TensorFloat-32 (TF32)** | Extreme Latency | 2x-3x faster Matrix Multiplications on NVIDIA Ampere+ GPUs (RTX 30xx/40xx/A100). |
| **Bypass `librosa` with `torchaudio`** | Extreme Latency | Drastically faster CPU audio resampling, saving ~10-30ms per real-time audio chunk. |
| **Integrate Silero VAD** | Real-Time Optimization | Skips inference on silent frames entirely. Blocks 100% of silence-induced hallucinations. |
| **Tune WebSocket Buffers** | Real-Time UX | Faster perceived TTFT by reducing chunk lengths from 800ms to 400ms-500ms. |
| **Adopt vLLM Engine backend** | Extreme Speed | 2x-4x throughput multiplier by utilizing PagedAttention and fused GPU kernels. |
| **Fix WebSocket Contention** | Latency / Reliability | Prevents active phone calls from stalling during heavy file processing. |
| **Limit Docker CPU Threads** | System Stability | Reduces CPU contention and speeds up GPU data feeding. |
| **Gateway + Worker Architecture** | System / Memory | Reclaims ~1.9GB of idle system RAM; drops idle footprint to ~30MB. |
| **Resolve "Fake" SSE Streaming** | Latency / UX | True Time-To-First-Token (TTFT) through incremental text yields. |

## 1. STT Quality Improvements

### Upgrade to Qwen3-ASR-1.7B Model
- **Current State**: The server defaults to the `0.6B` parameter model (`os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B")`).
- **Issue**: The `0.6B` model struggles significantly with multilingual content, noisy audio, and complex sentences compared to the `1.7B` model. 
- **Fix**: Update the `MODEL_ID` environment variable to `Qwen/Qwen3-ASR-1.7B`. The 1.7B model provides a ~35% improvement in multilingual accuracy (Fleurs benchmark) and ~25% better English accuracy. This is the ultimate fix for raw baseline quality.

### Implement Repetition Detection and Fixing
- **Current State**: The server returns the raw, unedited transcribed text directly from the model. 
- **Issue**: ASR models are highly prone to "hallucinating" repeated phrases (e.g., "the the the the") especially during periods of silence, noisy backgrounds, or phone call statics.
- **Fix**: Port the `detect_and_fix_repetitions()` utility from the official Qwen3-ASR SDK. Integrate it as a mandatory text post-processing step on all endpoint outputs before returning the HTTP/WebSocket response.

### Add Long Audio Chunking 
- **Current State**: Audio files are submitted entirely into the model in a single massive inference pass.
- **Issue**: The model's context window degrades quickly on audio longer than ~30 seconds. This leads to dropped sentences, hallucinations, or Out-Of-Memory (OOM) crashes on large files.
- **Fix**: Implement the official SDK's `split_audio_into_chunks()` method. This uses a sliding window energy estimation to safely partition long audio at natural silence boundaries without slicing words in half, processes the chunks sequentially, and concatenates the results.

## 2. Latency & Speed Improvements

### Enable `torch.compile`
- **Current State**: The model is loaded normally via lazy loading.
- **Issue**: Standard PyTorch execution overhead scales linearly and is suboptimal for recurring WebSocket chunks.
- **Fix**: Apply `torch.compile(model.model, mode="reduce-overhead", fullgraph=False)` immediately after loading the model. This compiles the PyTorch computation graph natively, yielding a **20-30% inference speedup**, drastically reducing WebSocket turnaround latency.

### Switch to Flash Attention 2
- **Current State**: The model is instantiated with `attn_implementation="sdpa"`.
- **Issue**: Scaled Dot-Product Attention (SDPA) is fast but slightly unoptimized compared to dedicated kernel implementations.
- **Fix**: Change it to `attn_implementation="flash_attention_2"` and ensure the `flash-attn` package is securely installed in the Docker container. This will provide up to **~20% faster attention computation**.

### Enable TensorFloat-32 (TF32) 
- **Current State**: Standard Float32 matrices are multiplying on CPU/GPU.
- **Issue**: If you run on an Nvidia Ampere (RTX 30xx/40xx or A100/H100) Tensor Cores are disabled by default for F32 operations in PyTorch.
- **Fix**: Add `torch.set_float32_matmul_precision('medium')` on application boot. This forces TF32 execution on Tensor Cores, offering **2x to 3x faster Matrix Multiplications (MM)** with negligible precision loss.

### Bypass `librosa` with `torchaudio`
- **Current State**: Audio is preprocessed in pure Python/NumPy using the `librosa` library.
- **Issue**: Real-time websocket audio streaming is heavily bottlenecked by CPU performance prior to hitting the GPU. `librosa.resample` blocks the main thread with slow computations. 
- **Fix**: Swap exclusively to `torchaudio.functional.resample` natively baked with the C++ PyTorch backend, or better yet, skip resampling altogether and strictly enforce 16kHz from your client devices. This shaves off `10-30ms` per buffer cycle.

### Integrate Silero VAD (Voice Activity Detection)
- **Current State**: Every single 800ms incoming audio buffer chunk is ruthlessly pushed to PyTorch for `.generate()` regardless of what it contains.
- **Issue**: Processing dead silence during a phone call wastes GPU cycles and forces the model to hallucinates text (mentioned above). 
- **Fix**: Insert a lightweight ONNX Silero VAD filter to check if someone actually spoke in the chunk. If Probability < 0.5, short-circuit and instantly return an empty string to the websocket `{"text": ""}`. 

### Scale Down WebSocket Buffer Size
- **Current State**: `WS_BUFFER_SIZE` defaults to ~800 milliseconds chunking.
- **Issue**: An 800ms chunk means the absolute fastest the system can output recognized text is 800ms (latency floor) + inference time.
- **Fix**: Drop the buffer size from 800ms to **400ms or 500ms** and dynamically scale the `WS_OVERLAP_SIZE` to 200ms. This speeds up perceived real-time UI by nearly ~40%, providing much more fluid token streams to users on the other end.

### Adopt the vLLM Engine backend
- **Current State**: We are manually orchestrating the HuggingFace `AutoModel` inside a Python thread loop queue.
- **Issue**: The server architecture entirely lacks PagedAttention, Continuous Batching, or fused decoding loops.
- **Fix**: The official Qwen3-ASR SDK actually supports vLLM. Wrap the entire `vLLM` async engine explicitly. Standard PyTorch execution scales horizontally extremely poorly. Leveraging vLLM PagedAttention handles massive traffic automatically giving us a **~2x to 4x multiplier on maximum real-time traffic** capacity.

### Fix WebSocket Semaphore Contention
- **Current State**: A single asyncio semaphore `_infer_semaphore = asyncio.Semaphore(1)` is globally shared between the `/v1/audio/transcriptions` HTTP API and the `/ws/transcribe` real-time WebSocket API.
- **Issue**: If a user uploads a heavy 5-minute audio file, the WebSocket stream for an active 2-way phone call will completely stall until the batch file finishes processing. 
- **Fix**: Implement a priority executor or priority semaphore logic. Real-time WebSocket chunks (which demand instant <1s latency) must pre-empt or have strict priority over background batch file HTTP requests.

### Limit Docker CPU Thread Contention
- **Current State**: PyTorch aggressively spawns CPU threads across all available cores.
- **Issue**: Unrestricted thread spawning slows down GPU data feeding and creates severe CPU contention with other containers running on the server.
- **Fix**: Apply `ENV OMP_NUM_THREADS=2` and `ENV MKL_NUM_THREADS=2` to the `Dockerfile` to strictly cap CPU over-subscription.

## 3. Memory & System Improvements

### Implement a Gateway + Worker Architecture
- **Current State**: The `server.py` implementation consumes ~1.9GB of idle resident system RAM even when the GPU model is "unloaded", due to the massive PyTorch CUDA runtime Python context persistently lingering in memory.
- **Issue**: This is a major waste of system RAM if the server operates in a resource-shared environment (e.g. Synology NAS).
- **Fix**: Refactor `server.py` into a lightweight 30MB pure-FastAPI "Gateway" router, which proxies requests to a heavier "Worker" PyTorch subprocess. The Gateway can gracefully completely kill the Worker OS process (dropping RAM usage to ~30MB) when idle, and spin it back up on-demand.

### Resolve "Fake" SSE Streaming
- **Current State**: The `/v1/audio/transcriptions/stream` endpoint mimics streaming but actually falls back to block-and-process the entire audio chunk at once due to missing base model streaming methods.
- **Issue**: Clients connected via SSE experience high Time-To-First-Token (TTFT) latency because no chunks are yielded iteratively.
- **Fix**: Implement manual progressive chunked streaming—split the incoming audio stream into ~2-second logical segments, trigger inference on each segment incrementally, and yield the text back to the client as each chunk completes.

---

## 4. Extreme Real-Time Optimizations

The improvements above address foundational gaps. This section goes further — tracing the exact hot path of a WebSocket audio chunk and eliminating every source of waste to achieve true real-time latency.

### WebSocket Critical Path Analysis

For every ~800ms audio chunk arriving on the WebSocket, the server currently executes:

```
Step                                          Est. Time
─────────────────────────────────────────────────────────
1.  WebSocket frame recv                      ~0.1ms
2.  bytearray.extend(data["bytes"])           ~0.01ms
3.  Buffer threshold check                    ~0ms
4.  Build full_audio = overlap + chunk        ~0.02ms
5.  bytes(full_audio) — UNNECESSARY COPY      ~0.05ms
6.  np.frombuffer → int16                     ~0.01ms
7.  .astype(float32) / 32768.0               ~0.1ms
8.  preprocess_audio():
    a. mono check (already mono → no-op)      ~0ms
    b. .astype(float32) — REDUNDANT           ~0.05ms
    c. import librosa check (cached)          ~0ms
    d. sr == 16000 → skip resample            ~0ms
    e. peak normalize                         ~0.05ms
9.  Acquire _infer_semaphore                  0–???ms ← blocks if HTTP active
10. run_in_executor dispatch to thread pool   ~0.3ms
11. model.transcribe() — THE ACTUAL WORK      ~50–200ms
12. release_gpu_memory():
    a. gc.collect()                           ~2–10ms ← HUGE WASTE
    b. torch.cuda.empty_cache()              ~1–5ms  ← defeats caching allocator
    c. torch.cuda.ipc_collect()              ~0.5ms
13. Return to event loop, send JSON           ~0.1ms
─────────────────────────────────────────────────────────
TOTAL                                         ~55–220ms+
```

Model inference (step 11) is ~50–200ms depending on chunk length and GPU. Everything else is overhead. The optimizations below target that overhead and the inference itself.

### Tier 0 — Free Latency (Zero Risk, Immediate Effect)

#### Stop Calling `release_gpu_memory()` After Every Inference
- **Current State**: `_do_transcribe()` calls `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` in a `finally` block after every single inference.
- **Issue**: `gc.collect()` triggers a full Python garbage collection sweep (2–10ms). `torch.cuda.empty_cache()` returns all cached blocks to CUDA, defeating the caching allocator's entire purpose (1–5ms). Combined, this burns **5–15ms of pure overhead per chunk** — that's 7–19% of the latency budget on an 800ms chunk.
- **Fix**: Remove `release_gpu_memory()` from `_do_transcribe()`. The caching allocator exists specifically to avoid `cudaMalloc`/`cudaFree` overhead. Keep `release_gpu_memory()` only in `_unload_model_sync()` and OOM error recovery paths.

#### Eliminate Redundant Preprocessing on WebSocket Path
- **Current State**: `_transcribe_with_context()` converts int16 → float32, then calls `preprocess_audio()` which calls `.astype(np.float32)` again (redundant), checks mono (always mono on WS), checks resampling (always 16kHz on WS). Only peak normalization is useful.
- **Issue**: The WebSocket path reuses the file-upload preprocessor. Every check is a no-op except peak normalization.
- **Fix**: Create a lean inline path for WebSocket audio. Skip `preprocess_audio()` entirely. Apply only peak normalization directly in `_transcribe_with_context()`.

#### Eliminate the `bytes()` Copy in `_transcribe_with_context()`
- **Current State**: Line 514: `np.frombuffer(bytes(full_audio), dtype=np.int16)` — `bytes(full_audio)` copies the entire bytearray.
- **Issue**: `np.frombuffer` can accept a `bytearray` directly via the buffer protocol. The `bytes()` call allocates and copies for no reason.
- **Fix**: `np.frombuffer(full_audio, dtype=np.int16).copy()` — the `.copy()` is needed only because `frombuffer` returns a read-only view, but still avoids the intermediate `bytes()` allocation.

#### Enable `cudnn.benchmark`
- **Current State**: Not set.
- **Issue**: If the encoder has convolutional layers (most ASR encoders use convolutional subsampling), cuDNN defaults to a generic algorithm.
- **Fix**: Add `torch.backends.cudnn.benchmark = True` at startup. cuDNN will auto-tune and cache the fastest convolution algorithm for the input size. **5–20% speedup on conv layers**.
- **Caveat**: Only helps when input sizes are consistent (they are — fixed chunk size).

#### Ensure `model.eval()` After Loading
- **Current State**: `torch.inference_mode()` is applied during transcription, but `model.eval()` may not be called after loading.
- **Issue**: `inference_mode()` disables autograd. `eval()` separately switches BatchNorm/Dropout to eval behavior. Both are needed for correct and fast inference.
- **Fix**: Add `model.eval()` immediately after `model = Qwen3ASRModel.from_pretrained(...)`.

#### Warmup with Representative Audio, Not Silence
- **Current State**: Warmup uses `np.zeros(TARGET_SR)` — one second of silence.
- **Issue**: CUDA kernels are JIT-compiled per code path. Silence triggers different encoder activations and possibly different attention patterns than speech. The first real request still pays a 100–500ms JIT compilation penalty.
- **Fix**: Warmup with random noise or a short synthetic speech-like signal at the typical chunk length (~800ms). This forces JIT compilation on the actual hot path.
  ```python
  warmup_audio = np.random.randn(int(TARGET_SR * 0.8)).astype(np.float32) * 0.1
  ```

#### Disable WebSocket `per-message-deflate` Compression
- **Current State**: The `websockets` library enables per-message-deflate by default.
- **Issue**: zlib compression adds CPU overhead (0.5–2ms per message) with near-zero compression ratio on raw PCM binary data. Also consumes ~300KB memory per connection.
- **Fix**: Disable in uvicorn: `uvicorn.run(app, ws_per_message_deflate=False)`. Or in the CMD: `--ws-per-message-deflate false`.

### Tier 1 — Low Effort, Significant Gains

#### Pre-Allocated Pinned Memory Buffers
- **Current State**: Each chunk creates new numpy arrays and torch tensors, triggering `malloc`/`cudaMalloc` per request. CPU → GPU transfer is synchronous pageable memory.
- **Issue**: Memory allocation overhead (~1–3ms per chunk) and synchronous PCIe transfer.
- **Fix**: Pre-allocate a page-locked (pinned) CPU buffer and a GPU buffer at startup. Reuse via `.copy_()` with `non_blocking=True`.
  ```python
  MAX_SAMPLES = int(TARGET_SR * 2)  # 2 seconds max
  _pinned_buf = torch.empty(MAX_SAMPLES, dtype=torch.float32, pin_memory=True)
  _gpu_buf = torch.empty(MAX_SAMPLES, dtype=torch.float32, device='cuda')

  # Per request: fill pinned buffer, async transfer to GPU
  _pinned_buf[:n].copy_(torch.from_numpy(audio))
  _gpu_buf[:n].copy_(_pinned_buf[:n], non_blocking=True)
  ```
- **Savings**: 1–3ms per chunk + enables overlapping transfer with computation.

#### Dedicated Inference ThreadPoolExecutor
- **Current State**: `run_in_executor(None, ...)` uses the default executor with unlimited threads.
- **Issue**: Each call may spawn a new thread (thread creation overhead, no CUDA context affinity, thread pool thrashing under concurrent load).
- **Fix**: Create a dedicated single-thread executor for inference:
  ```python
  from concurrent.futures import ThreadPoolExecutor
  _inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
  ```
  This guarantees CUDA context stays on one thread and eliminates thread creation/destruction overhead.
- **Savings**: Lower p99 latency by 10–30% under concurrent load.

#### CUDA Stream Pipelining
- **Current State**: Data transfer and compute happen sequentially on the default CUDA stream.
- **Issue**: While the GPU is processing chunk N, the PCIe bus sits idle. While chunk N+1 transfers, the GPU compute units sit idle.
- **Fix**: Use separate CUDA streams for transfer and compute to overlap them:
  ```python
  transfer_stream = torch.cuda.Stream()
  compute_stream = torch.cuda.Stream()

  with torch.cuda.stream(transfer_stream):
      next_gpu = pinned_next.to('cuda', non_blocking=True)
  with torch.cuda.stream(compute_stream):
      result = model(current_gpu)
  compute_stream.wait_stream(transfer_stream)
  ```
- **Savings**: Hides PCIe transfer latency entirely (~1–2ms per chunk).

### Tier 2 — Medium Effort, Large Gains

#### INT8 W8A8 Quantization
- **Current State**: Model runs in bfloat16.
- **Issue**: INT8 with SmoothQuant calibration can nearly double throughput on Ampere+ GPUs by leveraging INT8 Tensor Cores while maintaining near-identical accuracy.
- **Fix**: Use `llm-compressor`, `auto-gptq`, or CTranslate2's built-in INT8 quantization. Calibrate with ~100 representative audio samples.
- **Savings**: ~1.5–2x throughput improvement, ~50% memory reduction.
- **Caveat**: Requires calibration data for activation quantization. Weight-only INT8 (W8A16) is simpler but gives smaller throughput gains.

#### FP8 Quantization (Hopper/Ada GPUs Only)
- **Current State**: Model runs in bfloat16.
- **Issue**: FP8 (E4M3) halves memory bandwidth vs. BF16 and doubles Tensor Core throughput on H100/RTX 4090.
- **Fix**: Use `nvidia-ammo` or TensorRT-LLM for FP8 quantization with dynamic scaling.
- **Savings**: Up to 1.6x throughput, 2x memory reduction. For the memory-bandwidth-bound encoder, ~30–40% latency reduction.
- **Caveat**: Requires compute capability >= 8.9. Only H100, RTX 4090, and Ada-based GPUs.

#### CUDA Graphs for the Decoder Loop
- **Current State**: The decoder generates tokens autoregressively, launching many small CUDA kernels per token.
- **Issue**: Each kernel launch has ~5–15us of CPU dispatch overhead (Python → PyTorch dispatcher → CUDA driver). For a 20-token output, that's ~100–300us of pure dispatch overhead per token step.
- **Fix**: Capture the decoder forward pass as a CUDA Graph, then replay it with a single CPU dispatch:
  ```python
  g = torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
      static_output = model.decode(static_input)
  # Replay: just copy input and replay
  static_input.copy_(real_input)
  g.replay()
  ```
- **Savings**: 20–40% decoder latency reduction.
- **Caveat**: Requires fixed tensor shapes (pad to max decode length). No dynamic control flow or memory allocation inside the captured region. The Qwen3-ASR decoder may need refactoring for graph compatibility.

#### ONNX Runtime for the Encoder
- **Current State**: Encoder runs in PyTorch eager mode.
- **Issue**: PyTorch eager mode misses graph-level optimizations: operator fusion, constant folding, memory planning.
- **Fix**: Export the encoder to ONNX via `optimum-onnx`, run with ORT's CUDA execution provider:
  ```python
  from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
  model = ORTModelForSpeechSeq2Seq.from_pretrained(
      "Qwen/Qwen3-ASR-0.6B", export=True, provider="CUDAExecutionProvider"
  )
  ```
- **Savings**: 20–40% encoder speedup from graph optimizations.
- **Caveat**: Custom ops in Qwen3-ASR may not export cleanly. Validate output accuracy against PyTorch baseline.

#### KV-Cache Reuse Across WebSocket Chunks
- **Current State**: Every chunk is transcribed independently. The overlap audio (300ms) is re-encoded and re-decoded from scratch.
- **Issue**: The 300ms overlap is identical audio that was already processed in the previous chunk. Re-encoding it wastes GPU cycles.
- **Fix**: Cache the decoder's KV cache and the encoder's hidden states for the overlap portion. On the next chunk, warm-start the encoder/decoder from the cached state instead of re-computing.
- **Savings**: ~15–25% reduction in per-chunk encoder compute (proportional to overlap ratio).
- **Caveat**: Requires access to model internals (encoder hidden states, decoder KV cache). May need custom inference loop instead of `model.transcribe()`.

#### Dual-Model Strategy: 0.6B Partials, 1.7B Finals
- **Current State**: A single model handles both real-time partial results and final transcriptions.
- **Issue**: The 1.7B model is 35% more accurate but slower. The 0.6B is fast but less accurate.
- **Fix**: Keep the 0.6B model hot for real-time WebSocket partial results (speed matters most). On `flush`, run the accumulated audio through the 1.7B model for a high-accuracy final result. This gives the speed of 0.6B during the conversation and the accuracy of 1.7B for the transcript.
- **Savings**: Best of both worlds — real-time partials stay fast, final accuracy improves 35%.
- **Caveat**: Requires VRAM for two models (0.6B ~600MB + 1.7B ~1.7GB). May need the 0.6B in INT8 to fit both.

### Tier 3 — High Effort, Maximum Gains

#### TensorRT for the Encoder
- **Current State**: Encoder runs in PyTorch eager mode.
- **Issue**: Even ONNX Runtime leaves performance on the table compared to TensorRT's hardware-specific kernel auto-tuning and layer fusion.
- **Fix**: ONNX export → TensorRT engine conversion with shape profiling for the typical chunk size. Or use `torch-tensorrt` for inline conversion.
- **Savings**: **2–4x encoder speedup** over PyTorch eager mode. Often the lowest achievable latency on NVIDIA GPUs.
- **Caveat**: Engine is GPU-specific (must rebuild for different GPU). Fixed or profiled input shapes. Build time ~10–30 min.

#### Speculative Decoding for ASR
- **Current State**: The decoder generates tokens one at a time autoregressively.
- **Issue**: Each token requires a full forward pass through the decoder. Sequential bottleneck.
- **Fix**: Use a small draft model (0.6B as draft for 1.7B target) or n-gram token prediction for model-free speculation. Draft multiple tokens in parallel, verify in a single forward pass of the target model. Research (SpecASR, 2025) shows **3–4x decoder speedup** with zero accuracy loss.
- **Savings**: 3–4x faster decoding with draft model; 1.3–1.5x with n-gram model-free approach.
- **Caveat**: Draft model approach requires the 0.6B + 1.7B loaded simultaneously. N-gram approach needs a frequency table built from training data.

#### Cache-Aware Streaming Encoder (Nemotron-Style)
- **Current State**: The encoder re-processes the full overlap + chunk from scratch every time.
- **Issue**: The overlap represents redundant encoder computation. As overlap grows (to improve accuracy), the waste grows proportionally.
- **Fix**: Modify the encoder to use causal or limited-right-context attention with cached intermediate activations. Only new audio frames are encoded; cached frames provide context. This is the technique NVIDIA uses in Nemotron-speech-streaming.
- **Savings**: **3x improvement** in concurrent stream capacity. Near-flat latency as concurrency increases.
- **Caveat**: Requires rewriting the encoder's attention mechanism. May need retraining or fine-tuning the model. This is an architectural change, not a configuration tweak.

### Tier S — System-Level & Container

#### NUMA-Aware CPU Pinning
- **Current State**: Process runs without CPU affinity.
- **Issue**: On multi-socket servers, memory accesses may cross NUMA boundaries, adding latency to CPU-side operations (audio preprocessing, WebSocket handling, Python interpreter).
- **Fix**: Bind the process to the NUMA node closest to the GPU:
  ```bash
  nvidia-smi topo -m               # find GPU's NUMA node
  numactl --cpunodebind=0 --membind=0 uvicorn server:app ...
  ```
- **Savings**: 5–15% reduction in CPU-side latency.
- **Caveat**: Only helps on multi-socket servers. Single-socket machines have one NUMA node (no-op).

#### Switch ASGI Server to Granian
- **Current State**: Uvicorn (Python asyncio-based ASGI server).
- **Issue**: Uvicorn's Python event loop adds overhead compared to Rust-native alternatives.
- **Fix**: Replace Uvicorn with Granian, a Rust-based ASGI server. Drop-in replacement:
  ```bash
  pip install granian
  granian --interface asgi server:app --host 0.0.0.0 --port 8000
  ```
- **Savings**: ~10% higher throughput, lower tail latency, lower memory footprint.
- **Caveat**: Less mature ecosystem than Uvicorn. Test WebSocket behavior thoroughly.

#### Pre-Allocated Tensor Buffers (Eliminate Per-Request Allocation)
- **Current State**: Every chunk creates new numpy arrays and torch tensors.
- **Issue**: `malloc`, `cudaMalloc`, and Python object creation overhead accumulate per request.
- **Fix**: Pre-allocate fixed-size pinned CPU + GPU tensor buffers at startup. Reuse them for every chunk via `.copy_()`. Guard with the inference semaphore for thread safety.
- **Savings**: 1–3ms per chunk (eliminates allocation overhead entirely).

### Estimated Combined Impact

| Stage | Changes Applied | Est. Per-Chunk Latency | vs Baseline |
|-------|----------------|------------------------|-------------|
| **Baseline** | Current server | ~150ms | — |
| **+ Tier 0** | Remove gc/empty_cache, fix copies, cudnn.benchmark, warmup, disable WS compression | ~125ms | **-17%** |
| **+ Tier 1** | Pinned memory, dedicated executor, CUDA stream pipelining, TF32 | ~70–90ms | **-47%** |
| **+ Tier 2** | INT8 quantization, CUDA graphs, ONNX Runtime encoder | ~35–50ms | **-70%** |
| **+ Tier 3** | TensorRT encoder, speculative decoding | ~15–25ms | **-85%** |

With Tier 0+1 alone (all low-effort), combined with dropping the buffer from 800ms to 400ms:
**~500ms end-to-end latency** (400ms buffer + ~100ms inference) — genuinely real-time for phone calls.

With Tier 2 added: **~450ms** (400ms buffer + ~50ms inference). At this point, the buffer can drop to 200–300ms while maintaining stability.
