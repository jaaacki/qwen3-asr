#!/usr/bin/env python3
"""Download real audio samples from FLEURS dataset for multilingual accuracy tests.

Usage:
    pip install datasets soundfile
    python E2Etest/download_test_audio.py

Downloads 1-2 clips per language and saves them as WAV files with reference
transcriptions. Audio files go to E2Etest/data/audio/real/, expected
transcriptions to E2Etest/data/expected/.

Total download size: ~5-10 MB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def download_fleurs_samples():
    """Download audio samples from the FLEURS dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with:")
        print("  pip install datasets")
        sys.exit(1)

    base_dir = Path(__file__).parent
    audio_dir = base_dir / "data" / "audio" / "real"
    expected_dir = base_dir / "data" / "expected"
    audio_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    # FLEURS language configs
    # (output_name, fleurs_config, num_samples)
    languages = [
        ("english", "en_us", 2),
        ("chinese", "cmn_hans_cn", 2),
        ("japanese", "ja_jp", 2),
        ("cantonese", "yue_hant_hk", 2),
        ("hindi", "hi_in", 2),
        ("thai", "th_th", 2),
    ]

    for lang_name, config_name, num_samples in languages:
        print(f"\nDownloading {lang_name} ({config_name})...")
        try:
            ds = load_dataset(
                "google/fleurs",
                config_name,
                split="test",
                streaming=True,
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= num_samples:
                    break

                suffix = f"_{count + 1:02d}" if num_samples > 1 else "_01"
                wav_path = audio_dir / f"{lang_name}{suffix}.wav"
                txt_path = expected_dir / f"{lang_name}{suffix}.txt"

                # Extract audio
                audio_data = sample["audio"]
                array = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]

                # Resample to 16kHz if needed
                if sr != 16000:
                    new_len = int(len(array) * 16000 / sr)
                    old_idx = np.linspace(0, len(array) - 1, len(array))
                    new_idx = np.linspace(0, len(array) - 1, new_len)
                    array = np.interp(new_idx, old_idx, array).astype(np.float32)
                    sr = 16000

                sf.write(str(wav_path), array, sr, subtype="PCM_16")

                # Save reference transcription
                transcription = sample.get("transcription") or sample.get("raw_transcription", "")
                txt_path.write_text(transcription.strip())

                duration = len(array) / sr
                print(f"  {wav_path.name} ({duration:.1f}s) -> {transcription[:60]}...")
                count += 1

            if count == 0:
                print(f"  WARNING: No samples found for {lang_name}")

        except Exception as e:
            print(f"  ERROR downloading {lang_name}: {e}")

    # Code-switching samples from CS-FLEURS (zh-en)
    print("\nDownloading code-switching samples (CS-FLEURS zh-en)...")
    try:
        ds = load_dataset(
            "GT-SALT/CS-FLEURS",
            "zh_en",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        for sample in ds:
            if count >= 2:
                break

            suffix = f"_{count + 1:02d}"
            wav_path = audio_dir / f"codeswitching{suffix}.wav"
            txt_path = expected_dir / f"codeswitching{suffix}.txt"

            audio_data = sample["audio"]
            array = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]

            if sr != 16000:
                new_len = int(len(array) * 16000 / sr)
                old_idx = np.linspace(0, len(array) - 1, len(array))
                new_idx = np.linspace(0, len(array) - 1, new_len)
                array = np.interp(new_idx, old_idx, array).astype(np.float32)
                sr = 16000

            sf.write(str(wav_path), array, sr, subtype="PCM_16")

            transcription = sample.get("transcription") or sample.get("raw_transcription", "")
            txt_path.write_text(transcription.strip())

            duration = len(array) / sr
            print(f"  {wav_path.name} ({duration:.1f}s) -> {transcription[:60]}...")
            count += 1

        if count == 0:
            print("  WARNING: No code-switching samples found")
            print("  Trying alternative: manual zh-en from FLEURS...")
            # Fallback: just note it wasn't available
            print("  Code-switching samples skipped (dataset not accessible)")

    except Exception as e:
        print(f"  ERROR downloading code-switching: {e}")
        print("  Code-switching tests will be skipped if audio not available")

    print("\nDone! Audio files saved to:", audio_dir)
    print("Transcriptions saved to:", expected_dir)

    # List downloaded files
    wav_files = sorted(audio_dir.glob("*.wav"))
    txt_files = sorted(expected_dir.glob("*.txt"))
    print(f"\n{len(wav_files)} audio files, {len(txt_files)} transcription files")
    for f in wav_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    download_fleurs_samples()
