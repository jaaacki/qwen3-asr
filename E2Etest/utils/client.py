"""HTTP and WebSocket client wrappers for E2E tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import httpx
import websockets
import numpy as np
import soundfile as sf


class ASRHTTPClient:
    """Synchronous HTTP client for Qwen3-ASR API."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> dict:
        """Check server health and model status."""
        response = self.client.get("/health")
        response.raise_for_status()
        return response.json()

    def transcribe(
        self,
        audio_path: Path,
        language: str = "auto",
        return_timestamps: bool = False
    ) -> dict:
        """Upload audio file for transcription."""
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"language": language}
            if return_timestamps:
                data["return_timestamps"] = "true"

            response = self.client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str = "auto",
        return_timestamps: bool = False
    ) -> dict:
        """Transcribe audio from bytes."""
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"language": language}
        if return_timestamps:
            data["return_timestamps"] = "true"

        response = self.client.post(
            "/v1/audio/transcriptions",
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()

    def transcribe_stream(
        self,
        audio_path: Path,
        language: str = "auto",
        return_timestamps: bool = False
    ) -> list[dict]:
        """Stream transcription via Server-Sent Events."""
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"language": language}
            if return_timestamps:
                data["return_timestamps"] = "true"

            response = self.client.post(
                "/v1/audio/transcriptions/stream",
                files=files,
                data=data,
                headers={"Accept": "text/event-stream"}
            )
            response.raise_for_status()

            # Parse SSE events
            events = []
            for line in response.text.strip().split("\n"):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip():
                        try:
                            events.append(json.loads(data_str))
                        except json.JSONDecodeError:
                            pass
            return events

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ASRAsyncHTTPClient:
    """Asynchronous HTTP client for Qwen3-ASR API."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def health(self) -> dict:
        """Check server health and model status."""
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()

    async def transcribe(
        self,
        audio_path: Path,
        language: str = "auto",
        return_timestamps: bool = False
    ) -> dict:
        """Upload audio file for transcription."""
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"language": language}
            if return_timestamps:
                data["return_timestamps"] = "true"

            response = await self.client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()


class ASRWebSocketClient:
    """WebSocket client for real-time streaming transcription."""

    def __init__(self, ws_url: str = "ws://localhost:8100/ws/transcribe"):
        self.ws_url = ws_url
        self.websocket = None
        self._connection_info: Optional[dict] = None

    async def connect(self) -> dict:
        """Connect to WebSocket endpoint and return connection info."""
        self.websocket = await websockets.connect(self.ws_url)
        # First message should be connection info
        raw_msg = await asyncio.wait_for(self.websocket.recv(), timeout=10)
        self._connection_info = json.loads(raw_msg)
        return self._connection_info

    async def send_audio(self, audio_bytes: bytes):
        """Send raw audio bytes (PCM 16-bit, 16kHz)."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(audio_bytes)

    async def send_audio_chunk(self, audio_array: np.ndarray, sr: int = 16000):
        """Send audio as numpy array (will convert to int16 PCM)."""
        # Ensure correct format
        if audio_array.dtype != np.int16:
            # Convert float to int16
            audio_array = (audio_array * 32767).astype(np.int16)
        await self.send_audio(audio_array.tobytes())

    async def flush(self) -> dict:
        """Send flush command and return final transcription.

        Drains any queued partial results and returns the final one.
        """
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(json.dumps({"action": "flush"}))
        # Drain messages until we get the final result
        while True:
            raw_msg = await asyncio.wait_for(self.websocket.recv(), timeout=60)
            msg = json.loads(raw_msg)
            if msg.get("is_final") or msg.get("status") == "buffer_reset":
                return msg
            # If it's a partial, keep reading

    async def reset(self) -> dict:
        """Send reset command to clear buffers."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(json.dumps({"action": "reset"}))
        raw_msg = await asyncio.wait_for(self.websocket.recv(), timeout=10)
        return json.loads(raw_msg)

    async def config(self, language: str) -> dict:
        """Send config command to set language."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(json.dumps({
            "action": "config",
            "language": language
        }))
        raw_msg = await asyncio.wait_for(self.websocket.recv(), timeout=10)
        return json.loads(raw_msg)

    async def receive(self, timeout: float = 30) -> dict:
        """Receive a message from the server."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        raw_msg = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
        return json.loads(raw_msg)

    async def stream_audio(
        self,
        audio_path: Path,
        chunk_duration_ms: float = 450,
        overlap_ms: float = 150
    ) -> AsyncGenerator[dict, None]:
        """Stream audio file and yield transcription results."""
        if not self.websocket:
            await self.connect()

        # Read audio file
        audio, sr = sf.read(str(audio_path), dtype="int16")

        # Resample if needed (simple downsample)
        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        # Calculate chunk sizes
        bytes_per_sample = 2  # int16
        samples_per_chunk = int(16000 * chunk_duration_ms / 1000)
        samples_overlap = int(16000 * overlap_ms / 1000)
        chunk_size_bytes = samples_per_chunk * bytes_per_sample

        # Stream chunks
        idx = 0
        while idx < len(audio):
            chunk = audio[idx:idx + samples_per_chunk]
            await self.send_audio(chunk.tobytes())

            # Try to receive any partial results (non-blocking with timeout)
            try:
                msg = await asyncio.wait_for(self.receive(), timeout=0.5)
                yield msg
            except asyncio.TimeoutError:
                pass

            idx += samples_per_chunk

        # Flush to get final result
        final = await self.flush()
        yield final

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio
        # Calculate new length
        new_length = int(len(audio) * target_sr / orig_sr)
        # Use numpy's interp for simple resampling
        old_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(new_indices, old_indices, audio).astype(audio.dtype)

    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    @property
    def connection_info(self) -> Optional[dict]:
        """Get connection info received on connect."""
        return self._connection_info

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def run_sync(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)
