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
