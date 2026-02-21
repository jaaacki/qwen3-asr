"""Audio file generation and manipulation utilities."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional


def generate_test_tone(
    duration_s: float,
    frequency: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5
) -> np.ndarray:
    """Generate a sine wave test tone."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone.astype(np.float32)


def generate_silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence (zeros)."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def generate_noise(
    duration_s: float,
    sample_rate: int = 16000,
    amplitude: float = 0.1
) -> np.ndarray:
    """Generate white noise."""
    samples = int(sample_rate * duration_s)
    noise = amplitude * np.random.randn(samples)
    return noise.astype(np.float32)


def generate_speech_like_signal(
    duration_s: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """Generate a signal that looks like speech (modulated noise + tones)."""
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples, endpoint=False)

    # Base noise
    signal = 0.05 * np.random.randn(samples)

    # Add some tonal components (formant-like)
    for freq in [150, 300, 600, 1200]:
        signal += 0.1 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))

    # Add amplitude modulation (syllable-like, ~4 Hz)
    modulation = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 4 * t))
    signal *= modulation

    return signal.astype(np.float32)


def save_wav(
    audio: np.ndarray,
    path: Path,
    sample_rate: int = 16000,
    normalize: bool = True
) -> Path:
    """Save audio array to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = audio.copy()

    if normalize:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path


def generate_test_audio_files(output_dir: Path, force: bool = False) -> dict[Path, dict]:
    """Generate a set of test audio files.

    Returns dict mapping path to metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # 5-second test tone
    path_5s = output_dir / "short_5s.wav"
    if not path_5s.exists() or force:
        audio = generate_test_tone(5.0, frequency=440.0)
        save_wav(audio, path_5s)
    files[path_5s] = {"duration": 5.0, "type": "tone", "sample_rate": 16000}

    # 20-second speech-like signal
    path_20s = output_dir / "medium_20s.wav"
    if not path_20s.exists() or force:
        audio = generate_speech_like_signal(20.0)
        save_wav(audio, path_20s)
    files[path_20s] = {"duration": 20.0, "type": "speech-like", "sample_rate": 16000}

    # 60-second long audio
    path_60s = output_dir / "long_60s.wav"
    if not path_60s.exists() or force:
        audio = generate_speech_like_signal(60.0)
        save_wav(audio, path_60s)
    files[path_60s] = {"duration": 60.0, "type": "speech-like", "sample_rate": 16000}

    # Noisy sample
    path_noisy = output_dir / "noisy_sample.wav"
    if not path_noisy.exists() or force:
        # Mix speech-like with more noise
        audio = generate_speech_like_signal(10.0)
        noise = generate_noise(10.0, amplitude=0.3)
        noisy = audio + noise
        save_wav(noisy, path_noisy)
    files[path_noisy] = {"duration": 10.0, "type": "noisy", "sample_rate": 16000}

    # Silence
    path_silence = output_dir / "silence_5s.wav"
    if not path_silence.exists() or force:
        audio = generate_silence(5.0)
        save_wav(audio, path_silence)
    files[path_silence] = {"duration": 5.0, "type": "silence", "sample_rate": 16000}

    return files


def get_audio_duration(path: Path) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(str(path))
    return info.duration


def get_audio_info(path: Path) -> dict:
    """Get audio file information."""
    info = sf.info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
    }


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if orig_sr == target_sr:
        return audio

    new_length = int(len(audio) * target_sr / orig_sr)
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(new_indices, old_indices, audio)


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono."""
    if len(audio.shape) == 1:
        return audio
    return audio.mean(axis=1)


def convert_to_int16(audio: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convert float audio to int16 for WebSocket."""
    if audio.dtype == np.int16:
        return audio

    if normalize:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

    return (audio * 32767).astype(np.int16)


def create_pcm_16khz_mono(
    source_path: Path,
    output_path: Optional[Path] = None
) -> Path:
    """Convert any audio file to PCM 16-bit 16kHz mono for WebSocket testing.

    If output_path is None, overwrites the source (use with caution).
    """
    audio, sr = sf.read(str(source_path), dtype="float32")

    # Convert to mono
    audio = convert_to_mono(audio)

    # Resample to 16kHz if needed
    if sr != 16000:
        audio = resample_audio(audio, sr, 16000)

    # Convert to int16
    audio = convert_to_int16(audio)

    # Save
    if output_path is None:
        output_path = source_path

    sf.write(str(output_path), audio, 16000, subtype="PCM_16")
    return output_path
