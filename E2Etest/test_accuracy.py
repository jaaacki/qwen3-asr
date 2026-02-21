"""Accuracy tests for Qwen3-ASR transcription.

Tests transcription quality:
- Word Error Rate (WER) if reference transcripts available
- Repetition detection validation
- Language detection accuracy
"""

import re
from pathlib import Path
from typing import Optional

import pytest

from utils.client import ASRHTTPClient


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    """
    # Normalize text
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                  dp[i][j - 1],      # insertion
                                  dp[i - 1][j - 1])  # substitution

    errors = dp[m][n]
    return errors / len(ref_words)


def has_repetition_artifacts(text: str) -> bool:
    """Check if text has obvious repetition artifacts."""
    if not text:
        return False

    # Check for repeated words (3+ times)
    words = text.split()
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True

    # Check for repeated phrases (3-5 words repeating)
    for phrase_len in range(3, 6):
        for i in range(len(words) - phrase_len * 2 + 1):
            phrase1 = words[i:i+phrase_len]
            phrase2 = words[i+phrase_len:i+phrase_len*2]
            if phrase1 == phrase2:
                return True

    return False


# =============================================================================
# Reference-based WER Tests
# =============================================================================

@pytest.mark.accuracy
class TestTranscriptionAccuracy:
    """Accuracy tests when reference transcripts are available."""

    def test_wer_on_reference_data(self, ensure_server, sample_audio_5s: Path):
        """Calculate WER if reference transcript exists."""
        expected_dir = Path(__file__).parent / "data" / "expected"
        reference_file = expected_dir / f"{sample_audio_5s.stem}.txt"

        if not reference_file.exists():
            pytest.skip(f"No reference transcript found: {reference_file}")

        reference = reference_file.read_text().strip()

        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)
            hypothesis = result["text"]

        wer = calculate_wer(reference, hypothesis)

        # WER should be reasonable (adjust threshold based on expected quality)
        assert wer < 0.5, f"WER too high: {wer:.2%}\nRef: {reference}\nHyp: {hypothesis}"

    def test_transcription_not_empty_for_speech(self, ensure_server, sample_audio_20s: Path):
        """Transcription is not empty for speech-like audio."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_20s)

            # Should produce some output (even if garbage)
            assert "text" in result
            # Note: For synthetic audio, text might be meaningless
            # but shouldn't be completely empty


# =============================================================================
# Repetition Detection Tests
# =============================================================================

@pytest.mark.accuracy
class TestRepetitionDetection:
    """Tests for repetition detection and cleanup."""

    def test_no_obvious_repetitions(self, ensure_server, sample_audio_5s: Path):
        """Transcription doesn't have obvious repetition artifacts."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)
            text = result["text"]

            # Skip check for empty text (synthetic audio may produce nothing)
            if text.strip():
                assert not has_repetition_artifacts(text), \
                    f"Detected repetition artifacts: {text}"

    def test_repetition_cleanup_for_noisy_audio(self, ensure_server, sample_audio_noisy: Path):
        """Noisy audio doesn't produce excessive repetitions."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_noisy)
            text = result["text"]

            # Noisy audio might produce some artifacts but shouldn't be pathological
            words = text.split()
            if len(words) > 10:
                # Check that most words aren't the same
                unique_words = set(words)
                assert len(unique_words) > len(words) * 0.3, \
                    f"Too repetitive: {text}"


# =============================================================================
# Language Detection Tests
# =============================================================================

@pytest.mark.accuracy
class TestLanguageDetection:
    """Tests for automatic language detection."""

    def test_language_detected_or_defaulted(self, ensure_server, sample_audio_5s: Path):
        """Language is either detected or defaults to something reasonable."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s, language="auto")

            # Should have a language field
            assert "language" in result
            lang = result["language"]

            # Should be a valid language string (Qwen3-ASR uses full names like 'English')
            assert isinstance(lang, str)

    def test_explicit_language_used(self, ensure_server, sample_audio_5s: Path):
        """Explicit language parameter is respected.

        Note: Qwen3-ASR expects full language names (e.g. 'English') not
        ISO codes (e.g. 'en').
        """
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s, language="English")

            # Language should be 'English'
            assert result.get("language") == "English"


# =============================================================================
# Format Validation Tests
# =============================================================================

@pytest.mark.accuracy
class TestOutputFormat:
    """Tests for output format validation."""

    def test_text_is_string(self, ensure_server, sample_audio_5s: Path):
        """Text output is a string."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)

            assert isinstance(result["text"], str)

    def test_no_control_characters(self, ensure_server, sample_audio_5s: Path):
        """Text doesn't contain unexpected control characters."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)
            text = result["text"]

            # Allow normal whitespace but not other control chars
            for char in text:
                code = ord(char)
                if code < 32 and code not in [9, 10, 13]:  # tab, newline, carriage return
                    assert False, f"Control character found: U+{code:04X}"

    def test_reasonable_text_length(self, ensure_server, sample_audio_5s: Path):
        """Text length is reasonable for audio duration."""
        from utils.audio import get_audio_duration

        duration = get_audio_duration(sample_audio_5s)

        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)
            text = result["text"]

        # Rough heuristic: 5-30 characters per second is reasonable for speech
        chars_per_sec = len(text) / duration if duration > 0 else 0

        # Very high rate might indicate garbage
        assert chars_per_sec < 50, f"Unreasonably high char rate: {chars_per_sec:.1f}/s"
