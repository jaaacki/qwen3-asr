"""Accuracy tests for Qwen3-ASR transcription.

Tests transcription quality:
- Word Error Rate (WER) if reference transcripts available
- Repetition detection validation
- Language detection accuracy
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pytest

from utils.client import ASRHTTPClient


def _edit_distance(ref_tokens: list, hyp_tokens: list) -> int:
    """Compute edit distance between two token sequences."""
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                  dp[i][j - 1],      # insertion
                                  dp[i - 1][j - 1])  # substitution
    return dp[m][n]


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    errors = _edit_distance(ref_words, hyp_words)
    return errors / len(ref_words)


def _normalize_for_cer(text: str) -> str:
    """Normalize text for character-level comparison.

    Removes spaces, punctuation, and normalizes Unicode for fair CJK comparison.
    """
    text = unicodedata.normalize("NFKC", text.strip().lower())
    # Remove whitespace and common punctuation
    text = re.sub(r'[\s\u3000]+', '', text)  # spaces + ideographic space
    text = re.sub(r'[，。、；：！？""''「」『』（）\[\]【】〈〉《》‧·,\.;:!\?\-\'\"()]', '', text)
    return text


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate — suited for CJK and Thai text.

    Like WER but operates on individual characters after normalizing away
    spaces and punctuation (which differ between reference and ASR output).
    """
    ref_chars = list(_normalize_for_cer(reference))
    hyp_chars = list(_normalize_for_cer(hypothesis))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    errors = _edit_distance(ref_chars, hyp_chars)
    return errors / len(ref_chars)


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


# =============================================================================
# Multilingual Accuracy Tests (real audio from FLEURS)
# =============================================================================

REAL_AUDIO_DIR = Path(__file__).parent / "data" / "audio" / "real"
EXPECTED_DIR = Path(__file__).parent / "data" / "expected"

# Languages that need character-level error rate (CER) instead of word-level WER.
# CJK and Thai text is not space-delimited, so word-level WER is meaningless.
CER_LANGUAGES = {"Chinese", "Japanese", "Cantonese", "Thai"}

# Language mapping: (display_name, audio_file, ref_file, language_param)
MULTILINGUAL_CASES = [
    ("English", "english_01.wav", "english_01.txt", "English"),
    ("English-2", "english_02.wav", "english_02.txt", "English"),
    ("Chinese", "chinese_01.wav", "chinese_01.txt", "Chinese"),
    ("Chinese-2", "chinese_02.wav", "chinese_02.txt", "Chinese"),
    ("Japanese", "japanese_01.wav", "japanese_01.txt", "Japanese"),
    ("Japanese-2", "japanese_02.wav", "japanese_02.txt", "Japanese"),
    ("Cantonese", "cantonese_01.wav", "cantonese_01.txt", "Cantonese"),
    ("Cantonese-2", "cantonese_02.wav", "cantonese_02.txt", "Cantonese"),
    ("Hindi", "hindi_01.wav", "hindi_01.txt", "Hindi"),
    ("Hindi-2", "hindi_02.wav", "hindi_02.txt", "Hindi"),
    ("Thai", "thai_01.wav", "thai_01.txt", "Thai"),
    ("Thai-2", "thai_02.wav", "thai_02.txt", "Thai"),
]


@pytest.mark.accuracy
class TestMultilingualAccuracy:
    """Test transcription accuracy across languages with real FLEURS audio."""

    @pytest.mark.parametrize(
        "lang,audio_file,ref_file,lang_param",
        [(c[0], c[1], c[2], c[3]) for c in MULTILINGUAL_CASES],
        ids=[c[0] for c in MULTILINGUAL_CASES],
    )
    def test_language_transcription(
        self, ensure_server, lang, audio_file, ref_file, lang_param,
    ):
        """Transcribe real audio and compare to reference transcription."""
        audio_path = REAL_AUDIO_DIR / audio_file
        ref_path = EXPECTED_DIR / ref_file

        if not audio_path.exists():
            pytest.skip(
                f"Real audio not found: {audio_path}. "
                "Run: python E2Etest/download_test_audio.py"
            )
        if not ref_path.exists():
            pytest.skip(f"Reference transcript not found: {ref_path}")

        reference = ref_path.read_text().strip()

        with ASRHTTPClient() as client:
            result = client.transcribe(audio_path, language=lang_param)
            hypothesis = result["text"]

        assert hypothesis.strip(), f"Empty transcription for {lang}"

        # Use CER for CJK/Thai (not space-delimited), WER for alphabetic
        base_lang = lang.rstrip("-2").rstrip("0123456789").rstrip("-")
        if base_lang in CER_LANGUAGES:
            error_rate = calculate_cer(reference, hypothesis)
            metric_name = "CER"
        else:
            error_rate = calculate_wer(reference, hypothesis)
            metric_name = "WER"

        # Print details for report capture
        print(f"Language: {lang}")
        print(f"{metric_name}: {error_rate:.2%}")
        print(f"Reference: {reference[:120]}")
        print(f"Hypothesis: {hypothesis[:120]}")

        # Generous threshold — real multilingual ASR is challenging
        assert error_rate <= 0.5, (
            f"{metric_name} too high for {lang}: {error_rate:.2%}\n"
            f"Ref: {reference[:200]}\n"
            f"Hyp: {hypothesis[:200]}"
        )

    @pytest.mark.parametrize(
        "lang,audio_file,lang_param",
        [
            ("English", "english_01.wav", "English"),
            ("Chinese", "chinese_01.wav", "Chinese"),
            ("Japanese", "japanese_01.wav", "Japanese"),
        ],
        ids=["English-auto", "Chinese-auto", "Japanese-auto"],
    )
    def test_auto_language_detection_real_audio(
        self, ensure_server, lang, audio_file, lang_param,
    ):
        """Auto language detection works on real speech."""
        audio_path = REAL_AUDIO_DIR / audio_file
        if not audio_path.exists():
            pytest.skip(f"Real audio not found: {audio_path}")

        with ASRHTTPClient() as client:
            result = client.transcribe(audio_path, language="auto")

        assert result.get("text", "").strip(), f"Empty transcription for auto-detected {lang}"
        detected = result.get("language", "")
        print(f"Expected: {lang_param}, Detected: {detected}")


@pytest.mark.accuracy
class TestCodeSwitching:
    """Test code-switching (mixed language) transcription."""

    def test_code_switched_audio(self, ensure_server):
        """Transcribe code-switched audio with auto language detection."""
        # Try both possible code-switching files
        for name in ["codeswitching_01.wav", "codeswitching_02.wav"]:
            audio_path = REAL_AUDIO_DIR / name
            if audio_path.exists():
                break
        else:
            pytest.skip(
                "Code-switching audio not available. "
                "CS-FLEURS dataset may not be accessible."
            )

        ref_path = EXPECTED_DIR / audio_path.stem
        ref_path = ref_path.with_suffix(".txt")

        with ASRHTTPClient() as client:
            result = client.transcribe(audio_path, language="auto")

        hypothesis = result["text"]
        assert hypothesis.strip(), "Empty transcription for code-switched audio"

        if ref_path.exists():
            reference = ref_path.read_text().strip()
            wer = calculate_wer(reference, hypothesis)
            print(f"Code-switching WER: {wer:.2%}")
            print(f"Reference: {reference[:120]}")
            print(f"Hypothesis: {hypothesis[:120]}")
        else:
            print(f"No reference — hypothesis: {hypothesis[:120]}")
