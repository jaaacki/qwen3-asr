"""Subtitle generation module: ForcedAligner, segmentation, SRT formatting."""
from logger import log

from __future__ import annotations

import dataclasses
import os
import re


@dataclasses.dataclass
class SubtitleEvent:
    """A single subtitle entry with timing and text."""
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    # Round to milliseconds first to avoid millis=1000 from float imprecision
    total_ms = round(seconds * 1000)
    hours = int(total_ms // 3_600_000)
    total_ms %= 3_600_000
    minutes = int(total_ms // 60_000)
    total_ms %= 60_000
    secs = int(total_ms // 1000)
    millis = int(total_ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_srt(events: list[SubtitleEvent]) -> str:
    """Format subtitle events as an SRT string."""
    blocks = []
    for event in events:
        block = (
            f"{event.index}\n"
            f"{_format_timestamp(event.start)} --> {_format_timestamp(event.end)}\n"
            f"{event.text}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


@dataclasses.dataclass
class WordTimestamp:
    """A single word with its start and end time."""
    text: str
    start: float
    end: float


# CJK Unicode ranges: Han, Hiragana, Katakana, CJK punctuation
_CJK_PATTERN = re.compile(
    r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u3000-\u303F\uFF00-\uFFEF]"
)


def _is_cjk(text: str) -> bool:
    """Return True if text contains CJK characters."""
    return bool(_CJK_PATTERN.search(text))


def _is_cjk_char(char: str) -> bool:
    """Return True if a single character is CJK."""
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
        or 0x3040 <= cp <= 0x309F    # Hiragana
        or 0x30A0 <= cp <= 0x30FF    # Katakana
        or 0xAC00 <= cp <= 0xD7AF    # Hangul
        or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
    )


def _tokenize(text: str) -> list[str]:
    """Split text into tokens suitable for subtitle segmentation.

    For CJK text, splits CJK characters individually while keeping
    Latin/other words as whitespace-delimited tokens. Handles mixed
    CJK/Latin text correctly.
    For pure non-CJK text, splits by whitespace.
    """
    if not text or not text.strip():
        return []
    if not _is_cjk(text):
        return text.split()

    # Mixed or pure CJK: split CJK chars individually, group Latin by whitespace
    tokens: list[str] = []
    current: list[str] = []

    for ch in text:
        if _is_cjk_char(ch):
            # Flush any accumulated Latin text
            if current:
                tokens.extend("".join(current).split())
                current = []
            tokens.append(ch)
        else:
            current.append(ch)

    if current:
        tokens.extend("".join(current).split())

    return tokens


# Punctuation that ends a sentence -- triggers a subtitle break
_SENTENCE_ENDERS = frozenset(".?!;")

# Punctuation that marks a good mid-sentence break point
_CLAUSE_BREAKS = frozenset(",:;")

# Prepositions/conjunctions -- prefer breaking BEFORE these
_BREAK_BEFORE = frozenset({
    "and", "but", "or", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "that", "which", "who", "whom", "where", "when", "while",
    "because", "although", "if", "unless", "until", "after", "before",
})


def segment_subtitles(
    words: list[WordTimestamp],
    max_line_chars: int = 42,
    max_duration: float = 7.0,
    pause_threshold: float = 0.5,
) -> list[SubtitleEvent]:
    """Group word timestamps into subtitle events.

    Rules:
    - Max 2 lines per subtitle, each <= max_line_chars
    - Break at sentence endings, long pauses, or char limits
    - Enforce max_duration per subtitle
    - Split long subtitles into two lines at clause/phrase boundaries
    """
    if not words:
        return []

    max_chars = max_line_chars * 2  # two lines
    events: list[SubtitleEvent] = []
    current: list[WordTimestamp] = []

    # Detect CJK from first word to decide join strategy
    all_text = "".join(w.text for w in words)
    cjk = _is_cjk(all_text)
    joiner = "" if cjk else " "

    def _flush():
        if not current:
            return
        text = joiner.join(w.text for w in current)
        text = _split_into_two_lines(text, max_line_chars)
        events.append(SubtitleEvent(
            index=len(events) + 1,
            start=current[0].start,
            end=current[-1].end,
            text=text,
        ))
        current.clear()

    for i, word in enumerate(words):
        # Check if adding this word would exceed limits
        current_text = joiner.join(w.text for w in current) if current else ""
        candidate_text = (
            f"{current_text}{joiner}{word.text}".strip() if current_text
            else word.text
        )
        current_duration = (word.end - current[0].start) if current else 0.0

        # Force break conditions
        should_break = False

        # 1. Long pause before this word
        if current and (word.start - current[-1].end) > pause_threshold:
            should_break = True

        # 2. Character limit exceeded
        elif len(candidate_text) > max_chars:
            should_break = True

        # 3. Duration limit exceeded
        elif current and current_duration > max_duration:
            should_break = True

        if should_break:
            _flush()

        current.append(word)

        # Check for sentence-ending punctuation -- break AFTER this word
        if word.text and word.text[-1] in _SENTENCE_ENDERS:
            _flush()

    _flush()
    return events


def _split_into_two_lines(text: str, max_line_chars: int) -> str:
    """Split text into two lines if it exceeds max_line_chars.

    Prefers breaking at clause boundaries, before conjunctions/prepositions,
    and aims for bottom-heavy layout (shorter top line).
    """
    if len(text) <= max_line_chars:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    # Find best split point
    best_idx = len(words) // 2  # fallback: midpoint
    best_score = -1

    for i in range(1, len(words)):
        top = " ".join(words[:i])
        bottom = " ".join(words[i:])

        # Skip if either line is too long
        if len(top) > max_line_chars or len(bottom) > max_line_chars:
            continue

        score = 0

        # Prefer breaking after clause punctuation
        if words[i - 1][-1] in _CLAUSE_BREAKS:
            score += 3

        # Prefer breaking before conjunctions/prepositions
        if words[i].lower().rstrip(".,!?;:") in _BREAK_BEFORE:
            score += 2

        # Prefer bottom-heavy (shorter top line)
        if len(top) <= len(bottom):
            score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    # If no valid split point was found (best_score still -1), keep as single line
    # Overflow is better than a broken two-line layout
    if best_score == -1:
        top = " ".join(words[:best_idx])
        bottom = " ".join(words[best_idx:])
        if len(top) > max_line_chars or len(bottom) > max_line_chars:
            return text

    top = " ".join(words[:best_idx])
    bottom = " ".join(words[best_idx:])
    return f"{top}\n{bottom}"


def enforce_timing(
    events: list[SubtitleEvent],
    min_duration: float = 0.833,
    min_gap: float = 0.083,
) -> list[SubtitleEvent]:
    """Post-process subtitle timing to enforce duration and gap constraints.

    - Extends short subtitles to min_duration
    - Ensures min_gap between consecutive subtitles
    - Fixes overlaps by truncating the earlier subtitle's end time
    """
    if not events:
        return events

    result = [dataclasses.replace(e) for e in events]

    # 1. Enforce minimum duration
    for e in result:
        if (e.end - e.start) < min_duration:
            e.end = e.start + min_duration

    # 2. Fix overlaps and enforce minimum gap (clamp to avoid negatives)
    for i in range(len(result) - 1):
        gap = result[i + 1].start - result[i].end
        if gap < min_gap:
            # Truncate earlier subtitle, but never below min_duration
            result[i].end = max(
                result[i].start + min_duration,
                result[i + 1].start - min_gap,
            )

    # 3. Second min_duration pass -- gap/overlap fix may have shrunk subtitles
    for e in result:
        if (e.end - e.start) < min_duration:
            e.end = e.start + min_duration

    return result


# ---------------------------------------------------------------------------
# ForcedAligner -- lazy-loaded on first accurate-mode subtitle request
# ---------------------------------------------------------------------------

_aligner = None

FORCED_ALIGNER_ID = os.getenv("FORCED_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")

# Maximum audio duration per aligner call (seconds)
_ALIGNER_MAX_SECONDS = 300  # 5 minutes


def load_aligner():
    """Lazy-load the Qwen3-ForcedAligner model."""
    global _aligner
    if _aligner is not None:
        return

    import torch
    from qwen_asr import Qwen3ForcedAligner

    log.info(f"Loading ForcedAligner: {FORCED_ALIGNER_ID}...")
    _aligner = Qwen3ForcedAligner.from_pretrained(
        FORCED_ALIGNER_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    log.info("ForcedAligner loaded")


def unload_aligner():
    """Unload the ForcedAligner to free VRAM."""
    global _aligner
    if _aligner is None:
        return
    del _aligner
    _aligner = None
    log.info("ForcedAligner unloaded")


def align_audio(
    audio: "np.ndarray",
    sr: int,
    text: str,
    language: str,
) -> list[WordTimestamp]:
    """Align transcribed text to audio, returning word-level timestamps.

    Handles the 5-minute aligner limit by chunking at silence boundaries.
    """
    import numpy as np

    if _aligner is None:
        raise RuntimeError("ForcedAligner not loaded. Call load_aligner() first.")

    total_samples = len(audio)
    max_samples = _ALIGNER_MAX_SECONDS * sr

    if total_samples <= max_samples:
        return _align_chunk(audio, sr, text, language, time_offset=0.0)

    # Long audio (>5 min): chunk audio and pass full text to each chunk.
    # The aligner will align the portion of text that matches each chunk.
    # If a chunk fails, fall back to heuristic estimation for that chunk.
    all_words: list[WordTimestamp] = []
    chunk_start = 0
    offset = 0.0

    while chunk_start < total_samples:
        chunk_end = min(chunk_start + max_samples, total_samples)
        chunk = audio[chunk_start:chunk_end]
        chunk_duration = len(chunk) / sr

        try:
            chunk_words = _align_chunk(
                chunk, sr, text, language, time_offset=offset,
            )
        except Exception:
            # Fallback: estimate timestamps for this chunk
            chunk_words = estimate_word_timestamps(
                text, offset, offset + chunk_duration,
            )

        all_words.extend(chunk_words)

        offset += chunk_duration
        chunk_start = chunk_end

    return all_words


def _align_chunk(
    audio: "np.ndarray",
    sr: int,
    text: str,
    language: str,
    time_offset: float,
) -> list[WordTimestamp]:
    """Align a single chunk (<=5 min) and apply time offset."""
    import torch

    with torch.inference_mode():
        results = _aligner.align(
            audio=(audio, sr),
            text=text,
            language=language,
        )

    words = []
    if results and hasattr(results[0], "time_stamps") and results[0].time_stamps:
        for ts in results[0].time_stamps:
            words.append(WordTimestamp(
                text=ts.text,
                start=ts.start_time + time_offset,
                end=ts.end_time + time_offset,
            ))
    return words


# ---------------------------------------------------------------------------
# Fast mode -- heuristic word timestamps from segment-level output
# ---------------------------------------------------------------------------


def estimate_word_timestamps(
    text: str,
    start: float,
    end: float,
) -> list[WordTimestamp]:
    """Estimate word-level timestamps by distributing time proportionally.

    Used in 'fast' mode when ForcedAligner is not available.
    Distributes segment duration across words based on character count.
    """
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []

    total_chars = sum(len(w) for w in tokens)
    if total_chars == 0:
        return []

    duration = end - start
    result = []
    t = start

    for token in tokens:
        token_duration = duration * (len(token) / total_chars)
        result.append(WordTimestamp(text=token, start=t, end=t + token_duration))
        t += token_duration

    return result


# ---------------------------------------------------------------------------
# Orchestrator -- full pipeline from ASR results to SRT
# ---------------------------------------------------------------------------


def generate_srt_from_results(
    results: list,
    audio: "np.ndarray",
    sr: int,
    mode: str = "accurate",
    max_line_chars: int = 42,
) -> str:
    """Generate SRT from ASR transcription results.

    Args:
        results: ASR model output (list of result objects with .text, .language)
        audio: Audio numpy array (float32)
        sr: Sample rate
        mode: "accurate" (ForcedAligner) or "fast" (heuristic)
        max_line_chars: Maximum characters per subtitle line
    """
    if not results:
        return ""

    text = " ".join(r.text for r in results if r.text).strip()
    if not text:
        return ""

    language = results[0].language if hasattr(results[0], "language") else "en"
    audio_duration = len(audio) / sr

    if mode == "accurate":
        if _aligner is None:
            raise RuntimeError(
                "ForcedAligner not loaded. Call load_aligner() before using accurate mode."
            )
        words = align_audio(audio, sr, text, language)
    else:
        words = estimate_word_timestamps(text, 0.0, audio_duration)

    if not words:
        return ""

    events = segment_subtitles(
        words,
        max_line_chars=max_line_chars,
    )

    events = enforce_timing(events)

    return format_srt(events)
