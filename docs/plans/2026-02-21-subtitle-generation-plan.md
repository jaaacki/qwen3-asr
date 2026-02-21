# Subtitle Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SRT subtitle generation via a new `/v1/audio/subtitles` endpoint with accurate (ForcedAligner) and fast (heuristic) modes, without affecting real-time WebSocket transcription.

**Architecture:** New `src/subtitle.py` module owns aligner loading, subtitle segmentation, and SRT formatting. `server.py` adds the endpoint and wires it to the existing `PriorityInferQueue`. The ForcedAligner is lazy-loaded on first accurate-mode request and unloaded alongside the main model on idle timeout.

**Tech Stack:** Qwen3-ForcedAligner-0.6B (via `qwen_asr`), Silero VAD (existing), FastAPI, numpy

---

## Task 1: SRT Formatter

The simplest, most testable piece. Pure string formatting with zero dependencies on models or audio.

**Files:**
- Create: `src/subtitle.py`
- Create: `src/subtitle_test.py`

**Step 1: Write the failing test for SRT formatting**

In `src/subtitle_test.py`:

```python
"""Unit tests for subtitle generation module."""
from subtitle import format_srt, SubtitleEvent


def test_format_srt_single_event():
    """Single subtitle event formats correctly."""
    events = [
        SubtitleEvent(index=1, start=1.2, end=4.5, text="Hello world")
    ]
    result = format_srt(events)
    expected = "1\n00:00:01,200 --> 00:00:04,500\nHello world\n"
    assert result == expected


def test_format_srt_multiple_events():
    """Multiple events are separated by blank lines."""
    events = [
        SubtitleEvent(index=1, start=0.0, end=2.0, text="First line"),
        SubtitleEvent(index=2, start=3.0, end=5.5, text="Second line"),
    ]
    result = format_srt(events)
    lines = result.strip().split("\n")
    # Should have: 1, timestamp, text, blank, 2, timestamp, text
    assert lines[0] == "1"
    assert lines[3] == ""
    assert lines[4] == "2"


def test_format_srt_two_line_subtitle():
    """Subtitle with two lines preserves line break."""
    events = [
        SubtitleEvent(index=1, start=1.0, end=4.0, text="Top line\nBottom line")
    ]
    result = format_srt(events)
    assert "Top line\nBottom line" in result


def test_format_srt_timestamp_precision():
    """Timestamps have millisecond precision with comma separator."""
    events = [
        SubtitleEvent(index=1, start=3661.123, end=3665.456, text="Text")
    ]
    result = format_srt(events)
    assert "01:01:01,123 --> 01:01:05,456" in result
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_format_srt_single_event -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'subtitle'`

**Step 3: Write the SRT formatter**

In `src/subtitle.py`:

```python
"""Subtitle generation module: ForcedAligner, segmentation, SRT formatting."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SubtitleEvent:
    """A single subtitle entry with timing and text."""
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/subtitle.py src/subtitle_test.py
git commit -m "feat: add SRT formatter with SubtitleEvent dataclass"
```

---

## Task 2: Subtitle Segmentation Engine

Groups word-level timestamps into subtitle blocks respecting CPS, line length, and duration constraints. Pure logic, no model dependencies.

**Files:**
- Modify: `src/subtitle.py`
- Modify: `src/subtitle_test.py`

**Step 1: Write failing tests for segmentation**

Append to `src/subtitle_test.py`:

```python
from subtitle import segment_subtitles, WordTimestamp


def test_segment_short_sentence():
    """Short sentence becomes a single subtitle."""
    words = [
        WordTimestamp("Hello", 0.0, 0.5),
        WordTimestamp("world.", 0.6, 1.0),
    ]
    events = segment_subtitles(words)
    assert len(events) == 1
    assert events[0].text == "Hello world."
    assert events[0].start == 0.0
    assert events[0].end == 1.0


def test_segment_splits_at_sentence_boundary():
    """Splits at sentence-ending punctuation."""
    words = [
        WordTimestamp("Hello.", 0.0, 0.5),
        WordTimestamp("How", 1.0, 1.3),
        WordTimestamp("are", 1.3, 1.5),
        WordTimestamp("you?", 1.5, 2.0),
    ]
    events = segment_subtitles(words)
    assert len(events) == 2
    assert events[0].text == "Hello."
    assert events[1].text == "How are you?"


def test_segment_splits_at_long_pause():
    """Splits when gap between words exceeds threshold."""
    words = [
        WordTimestamp("First", 0.0, 0.5),
        WordTimestamp("part.", 0.5, 1.0),
        WordTimestamp("Second", 2.5, 3.0),  # 1.5s gap
        WordTimestamp("part.", 3.0, 3.5),
    ]
    events = segment_subtitles(words)
    assert len(events) == 2


def test_segment_respects_max_chars():
    """Splits when character count would exceed max_line_chars * 2."""
    # Create a long string of words that exceeds 84 chars (42 * 2)
    words = []
    t = 0.0
    for i in range(20):
        w = f"word{i:02d}"
        words.append(WordTimestamp(w, t, t + 0.2))
        t += 0.3
    events = segment_subtitles(words, max_line_chars=42)
    # Should have more than 1 event
    assert len(events) > 1
    # Each event text should be <= 84 chars (2 lines of 42)
    for e in events:
        for line in e.text.split("\n"):
            assert len(line) <= 42


def test_segment_enforces_max_duration():
    """No subtitle block exceeds 7 seconds."""
    words = []
    t = 0.0
    for i in range(30):
        words.append(WordTimestamp(f"w{i}", t, t + 0.2))
        t += 0.3
    events = segment_subtitles(words)
    for e in events:
        assert (e.end - e.start) <= 7.0


def test_segment_two_line_split():
    """Long subtitle is split into two lines at a good break point."""
    words = [
        WordTimestamp("The", 0.0, 0.1),
        WordTimestamp("quick", 0.1, 0.3),
        WordTimestamp("brown", 0.3, 0.5),
        WordTimestamp("fox", 0.5, 0.7),
        WordTimestamp("jumps", 0.7, 0.9),
        WordTimestamp("over", 0.9, 1.1),
        WordTimestamp("the", 1.1, 1.2),
        WordTimestamp("lazy", 1.2, 1.4),
        WordTimestamp("dog.", 1.4, 1.6),
    ]
    events = segment_subtitles(words, max_line_chars=25)
    # Should fit in one event but split across two lines
    assert len(events) == 1
    assert "\n" in events[0].text
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_segment_short_sentence -v`
Expected: FAIL — `ImportError: cannot import name 'segment_subtitles'`

**Step 3: Write the segmentation engine**

Add to `src/subtitle.py`:

```python
@dataclasses.dataclass
class WordTimestamp:
    """A single word with its start and end time."""
    text: str
    start: float
    end: float


# Punctuation that ends a sentence — triggers a subtitle break
_SENTENCE_ENDERS = frozenset(".?!;")

# Punctuation that marks a good mid-sentence break point
_CLAUSE_BREAKS = frozenset(",:;")

# Prepositions/conjunctions — prefer breaking BEFORE these
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
    max_cps: int = 20,
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

    def _flush():
        if not current:
            return
        text = " ".join(w.text for w in current)
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
        current_text = " ".join(w.text for w in current) if current else ""
        candidate_text = f"{current_text} {word.text}".strip() if current_text else word.text
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

        # Check for sentence-ending punctuation — break AFTER this word
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

    top = " ".join(words[:best_idx])
    bottom = " ".join(words[best_idx:])
    return f"{top}\n{bottom}"
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/subtitle.py src/subtitle_test.py
git commit -m "feat: add subtitle segmentation engine with line-breaking"
```

---

## Task 3: Timing Enforcement

Post-processes subtitle events to enforce min/max duration and minimum gaps. Pure logic.

**Files:**
- Modify: `src/subtitle.py`
- Modify: `src/subtitle_test.py`

**Step 1: Write failing tests for timing enforcement**

Append to `src/subtitle_test.py`:

```python
from subtitle import enforce_timing


def test_enforce_min_duration():
    """Short subtitles are extended to minimum duration."""
    events = [
        SubtitleEvent(index=1, start=1.0, end=1.3, text="Hi"),  # 300ms, too short
    ]
    result = enforce_timing(events)
    assert (result[0].end - result[0].start) >= 0.833


def test_enforce_min_gap():
    """Gap between consecutive subtitles is at least 83ms."""
    events = [
        SubtitleEvent(index=1, start=1.0, end=2.0, text="First"),
        SubtitleEvent(index=2, start=2.02, end=3.0, text="Second"),  # 20ms gap
    ]
    result = enforce_timing(events)
    gap = result[1].start - result[0].end
    assert gap >= 0.083


def test_enforce_no_overlap():
    """Overlapping subtitles are fixed by truncating the first."""
    events = [
        SubtitleEvent(index=1, start=1.0, end=3.0, text="First"),
        SubtitleEvent(index=2, start=2.5, end=4.0, text="Second"),  # overlaps
    ]
    result = enforce_timing(events)
    assert result[0].end <= result[1].start


def test_enforce_timing_preserves_order():
    """Events remain in chronological order."""
    events = [
        SubtitleEvent(index=1, start=0.0, end=2.0, text="A"),
        SubtitleEvent(index=2, start=3.0, end=5.0, text="B"),
        SubtitleEvent(index=3, start=6.0, end=8.0, text="C"),
    ]
    result = enforce_timing(events)
    for i in range(len(result) - 1):
        assert result[i].end <= result[i + 1].start
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_enforce_min_duration -v`
Expected: FAIL — `ImportError: cannot import name 'enforce_timing'`

**Step 3: Write timing enforcement**

Add to `src/subtitle.py`:

```python
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

    # 2. Fix overlaps and enforce minimum gap
    for i in range(len(result) - 1):
        gap = result[i + 1].start - result[i].end
        if gap < min_gap:
            # Truncate earlier subtitle to create the gap
            result[i].end = result[i + 1].start - min_gap

    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/subtitle.py src/subtitle_test.py
git commit -m "feat: add subtitle timing enforcement (min duration, gaps, overlaps)"
```

---

## Task 4: ForcedAligner Loading and Alignment

Lazy-loads the Qwen3-ForcedAligner model and provides word-level timestamp alignment. Handles the 5-minute chunking constraint.

**Files:**
- Modify: `src/subtitle.py`
- Modify: `src/subtitle_test.py`

**Step 1: Write failing tests for aligner**

Append to `src/subtitle_test.py`:

```python
import os
from unittest.mock import patch, MagicMock
from subtitle import load_aligner, unload_aligner, align_audio, _aligner


def test_load_aligner_sets_global():
    """load_aligner sets the module-level _aligner global."""
    # Mock the actual model loading
    mock_model = MagicMock()
    with patch("subtitle.Qwen3ForcedAligner") as mock_cls:
        mock_cls.from_pretrained.return_value = mock_model
        load_aligner()
        mock_cls.from_pretrained.assert_called_once()


def test_unload_aligner_clears_global():
    """unload_aligner sets _aligner to None."""
    import subtitle
    subtitle._aligner = MagicMock()
    unload_aligner()
    assert subtitle._aligner is None


def test_align_audio_returns_word_timestamps():
    """align_audio returns list of WordTimestamp objects."""
    import subtitle

    # Mock aligner with fake alignment result
    mock_result = MagicMock()
    mock_result.time_stamps = [
        MagicMock(text="Hello", start_time=0.1, end_time=0.5),
        MagicMock(text="world", start_time=0.6, end_time=1.0),
    ]
    mock_aligner = MagicMock()
    mock_aligner.align.return_value = [mock_result]
    subtitle._aligner = mock_aligner

    import numpy as np
    audio = np.zeros(16000, dtype=np.float32)
    words = align_audio(audio, 16000, "Hello world", "en")

    assert len(words) == 2
    assert words[0].text == "Hello"
    assert words[0].start == 0.1
    assert words[1].text == "world"

    # Cleanup
    subtitle._aligner = None
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_load_aligner_sets_global -v`
Expected: FAIL — `ImportError: cannot import name 'load_aligner'`

**Step 3: Write aligner loading and alignment**

Add to `src/subtitle.py`:

```python
import os
import numpy as np

# ForcedAligner — lazy-loaded on first accurate-mode subtitle request
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

    print(f"Loading ForcedAligner: {FORCED_ALIGNER_ID}...")
    _aligner = Qwen3ForcedAligner.from_pretrained(
        FORCED_ALIGNER_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    _aligner.eval()
    print("ForcedAligner loaded")


def unload_aligner():
    """Unload the ForcedAligner to free VRAM."""
    global _aligner
    if _aligner is None:
        return
    del _aligner
    _aligner = None
    print("ForcedAligner unloaded")


def align_audio(
    audio: np.ndarray,
    sr: int,
    text: str,
    language: str,
) -> list[WordTimestamp]:
    """Align transcribed text to audio, returning word-level timestamps.

    Handles the 5-minute aligner limit by chunking at silence boundaries.
    """
    if _aligner is None:
        raise RuntimeError("ForcedAligner not loaded. Call load_aligner() first.")

    import torch

    total_samples = len(audio)
    max_samples = _ALIGNER_MAX_SECONDS * sr

    if total_samples <= max_samples:
        return _align_chunk(audio, sr, text, language, time_offset=0.0)

    # Split at 5-min boundaries for long audio
    # Use simple chunking — split text proportionally
    all_words: list[WordTimestamp] = []
    chunk_start = 0
    offset = 0.0

    while chunk_start < total_samples:
        chunk_end = min(chunk_start + max_samples, total_samples)
        chunk = audio[chunk_start:chunk_end]
        chunk_duration = len(chunk) / sr

        # Proportionally split text for this chunk
        ratio_start = chunk_start / total_samples
        ratio_end = chunk_end / total_samples
        text_words = text.split()
        word_start = int(ratio_start * len(text_words))
        word_end = int(ratio_end * len(text_words))
        if chunk_end >= total_samples:
            word_end = len(text_words)
        chunk_text = " ".join(text_words[word_start:word_end])

        if chunk_text.strip():
            chunk_words = _align_chunk(chunk, sr, chunk_text, language, time_offset=offset)
            all_words.extend(chunk_words)

        offset += chunk_duration
        chunk_start = chunk_end

    return all_words


def _align_chunk(
    audio: np.ndarray,
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/subtitle.py src/subtitle_test.py
git commit -m "feat: add ForcedAligner loading and word-level alignment"
```

---

## Task 5: Fast Mode (Heuristic Timestamps)

Generates approximate word timestamps from segment-level ASR output by distributing time proportionally across words by character count. No aligner needed.

**Files:**
- Modify: `src/subtitle.py`
- Modify: `src/subtitle_test.py`

**Step 1: Write failing tests for fast mode**

Append to `src/subtitle_test.py`:

```python
from subtitle import estimate_word_timestamps


def test_estimate_timestamps_basic():
    """Distributes time across words proportionally by char count."""
    words = estimate_word_timestamps("Hello world", 0.0, 2.0)
    assert len(words) == 2
    assert words[0].text == "Hello"
    assert words[1].text == "world"
    # Both words are 5 chars, so should get equal time
    duration_0 = words[0].end - words[0].start
    duration_1 = words[1].end - words[1].start
    assert abs(duration_0 - duration_1) < 0.01


def test_estimate_timestamps_proportional():
    """Longer words get more time."""
    words = estimate_word_timestamps("I wonderful", 0.0, 3.0)
    assert len(words) == 2
    # "wonderful" (9 chars) should get more time than "I" (1 char)
    d_short = words[0].end - words[0].start
    d_long = words[1].end - words[1].start
    assert d_long > d_short


def test_estimate_timestamps_single_word():
    """Single word gets full duration."""
    words = estimate_word_timestamps("Hello", 1.0, 3.0)
    assert len(words) == 1
    assert words[0].start == 1.0
    assert words[0].end == 3.0


def test_estimate_timestamps_empty():
    """Empty text returns empty list."""
    words = estimate_word_timestamps("", 0.0, 2.0)
    assert words == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_estimate_timestamps_basic -v`
Expected: FAIL — `ImportError: cannot import name 'estimate_word_timestamps'`

**Step 3: Write fast mode timestamp estimation**

Add to `src/subtitle.py`:

```python
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

    words = text.split()
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    duration = end - start
    result = []
    t = start

    for word in words:
        word_duration = duration * (len(word) / total_chars)
        result.append(WordTimestamp(text=word, start=t, end=t + word_duration))
        t += word_duration

    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/subtitle.py src/subtitle_test.py
git commit -m "feat: add fast-mode heuristic word timestamp estimation"
```

---

## Task 6: Top-Level `generate_srt()` Function

Orchestrates the full pipeline: ASR results → alignment (or estimation) → segmentation → timing → SRT.

**Files:**
- Modify: `src/subtitle.py`
- Modify: `src/subtitle_test.py`

**Step 1: Write failing tests**

Append to `src/subtitle_test.py`:

```python
from subtitle import generate_srt_from_results


def test_generate_srt_fast_mode():
    """Fast mode produces valid SRT from mock ASR results."""
    # Simulate ASR result object
    mock_result = MagicMock()
    mock_result.text = "Hello world. How are you?"
    mock_result.language = "en"

    audio = np.zeros(48000, dtype=np.float32)  # 3 seconds
    srt = generate_srt_from_results(
        results=[mock_result],
        audio=audio,
        sr=16000,
        mode="fast",
    )

    # Should be valid SRT
    assert "1\n" in srt
    assert "-->" in srt
    assert "Hello" in srt


def test_generate_srt_empty_results():
    """Empty ASR results produce empty SRT."""
    audio = np.zeros(16000, dtype=np.float32)
    srt = generate_srt_from_results(results=[], audio=audio, sr=16000, mode="fast")
    assert srt == ""
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_generate_srt_fast_mode -v`
Expected: FAIL — `ImportError: cannot import name 'generate_srt_from_results'`

**Step 3: Write the orchestrator**

Add to `src/subtitle.py`:

```python
def generate_srt_from_results(
    results: list,
    audio: np.ndarray,
    sr: int,
    mode: str = "accurate",
    max_cps: int = 20,
    max_line_chars: int = 42,
) -> str:
    """Generate SRT from ASR transcription results.

    Args:
        results: ASR model output (list of result objects with .text, .language)
        audio: Audio numpy array (float32)
        sr: Sample rate
        mode: "accurate" (ForcedAligner) or "fast" (heuristic)
        max_cps: Maximum characters per second
        max_line_chars: Maximum characters per subtitle line
    """
    if not results:
        return ""

    text = " ".join(r.text for r in results if r.text).strip()
    if not text:
        return ""

    language = results[0].language if hasattr(results[0], "language") else "en"
    audio_duration = len(audio) / sr

    if mode == "accurate" and _aligner is not None:
        words = align_audio(audio, sr, text, language)
    else:
        words = estimate_word_timestamps(text, 0.0, audio_duration)

    if not words:
        return ""

    events = segment_subtitles(
        words,
        max_line_chars=max_line_chars,
        max_cps=max_cps,
    )

    events = enforce_timing(events)

    return format_srt(events)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/subtitle.py src/subtitle_test.py
git commit -m "feat: add generate_srt_from_results orchestrator"
```

---

## Task 7: Server Endpoint and Integration

Wire the subtitle module into `server.py` with a new endpoint. Integrate aligner unload into idle lifecycle.

**Files:**
- Modify: `src/server.py` (lines ~496-515 for unload, add endpoint after line ~611)
- Modify: `src/subtitle_test.py` (add integration test)

**Step 1: Write failing test for the endpoint**

Append to `src/subtitle_test.py`:

```python
from fastapi.testclient import TestClient


def test_subtitle_endpoint_exists():
    """The /v1/audio/subtitles endpoint is registered."""
    # Import server app — will fail until endpoint is added
    import server
    client = TestClient(server.app)
    # OPTIONS or invalid request just to confirm route exists
    # We send a GET which should return 405 (Method Not Allowed), not 404
    response = client.get("/v1/audio/subtitles")
    assert response.status_code == 405  # POST only
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py::test_subtitle_endpoint_exists -v`
Expected: FAIL — 404 (route doesn't exist)

**Step 3: Add the endpoint to server.py**

After the existing `transcribe()` endpoint (around line 611), add:

```python
@app.post("/v1/audio/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("accurate"),
    max_cps: int = Form(20),
    max_line_chars: int = Form(42),
):
    """Generate SRT subtitles from audio file.

    Modes:
    - accurate: Uses ForcedAligner for word-level timestamps (~33ms accuracy)
    - fast: Heuristic estimation from segment boundaries (no aligner needed)
    """
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
        return JSONResponse(
            content="",
            media_type="text/plain",
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
            max_cps=max_cps,
            max_line_chars=max_line_chars,
        ),
    )

    return JSONResponse(
        content=srt_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
    )
```

Also modify `_unload_model_sync()` at line ~496 to unload the aligner:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/server.py src/subtitle.py src/subtitle_test.py
git commit -m "feat: add /v1/audio/subtitles endpoint with aligner lifecycle"
```

---

## Task 8: Gateway and Worker Proxy

Add subtitle endpoint forwarding to gateway and worker for GATEWAY_MODE support.

**Files:**
- Modify: `src/gateway.py` (add proxy endpoint after line ~114)
- Modify: `src/worker.py` (add worker endpoint)

**Step 1: Add proxy to gateway.py**

After the existing `transcribe()` handler (line ~114), add:

```python
@app.post("/v1/audio/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("accurate"),
    max_cps: int = Form(20),
    max_line_chars: int = Form(42),
):
    """Proxy subtitle generation to worker."""
    global _last_used
    await _ensure_worker()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/subtitles"
    form = aiohttp.FormData()
    form.add_field("file", await file.read(), filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("mode", mode)
    form.add_field("max_cps", str(max_cps))
    form.add_field("max_line_chars", str(max_line_chars))
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            srt_content = await resp.text()
            return JSONResponse(
                content=srt_content,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
            )
```

**Step 2: Add endpoint to worker.py**

After the existing `transcribe()` handler, add:

```python
@app.post("/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("accurate"),
    max_cps: int = Form(20),
    max_line_chars: int = Form(42),
):
    """Subtitle generation -- worker-side endpoint."""
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    lang_code = None if language == "auto" else language

    if mode == "accurate":
        from subtitle import load_aligner
        load_aligner()

    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, False),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        release_gpu_memory()
        return JSONResponse(status_code=504, content={"error": "Subtitle generation timed out"})

    if not results or len(results) == 0:
        return JSONResponse(content="", media_type="text/plain")

    for r in results:
        r.text = detect_and_fix_repetitions(r.text)

    from subtitle import generate_srt_from_results
    srt_content = generate_srt_from_results(
        results=results, audio=audio, sr=sr,
        mode=mode, max_cps=max_cps, max_line_chars=max_line_chars,
    )

    return JSONResponse(
        content=srt_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
    )
```

**Step 3: Run existing tests to verify no regressions**

Run: `cd /Users/noonoon/Dev/qwen3-asr/src && python -m pytest subtitle_test.py server_test.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/gateway.py src/worker.py
git commit -m "feat: add subtitle proxy in gateway and worker"
```

---

## Task 9: E2E Tests

End-to-end tests that run against the live container.

**Files:**
- Create: `E2Etest/test_subtitle.py`
- Modify: `E2Etest/conftest.py` (add `subtitle` marker)
- Modify: `E2Etest/utils/client.py` (add `subtitle()` method)

**Step 1: Add subtitle method to test client**

In `E2Etest/utils/client.py`, add to `ASRHTTPClient`:

```python
def subtitle(
    self,
    audio_path: Path,
    language: str = "auto",
    mode: str = "accurate",
    max_cps: int = 20,
    max_line_chars: int = 42,
) -> str:
    """Generate SRT subtitles from audio file."""
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {
            "language": language,
            "mode": mode,
            "max_cps": str(max_cps),
            "max_line_chars": str(max_line_chars),
        }
        response = self.client.post(
            "/v1/audio/subtitles",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return response.text
```

**Step 2: Add conftest marker**

In `E2Etest/conftest.py`, add to `pytest_configure()`:

```python
config.addinivalue_line("markers", "subtitle: marks tests as subtitle tests")
```

And add to `pytest_collection_modifyitems()`:

```python
if "subtitle" in item.nodeid:
    item.add_marker(pytest.mark.subtitle)
```

Also add to the `_categorize` mapping:

```python
"test_subtitle": "Subtitle",
```

**Step 3: Create E2E test file**

Create `E2Etest/test_subtitle.py`:

```python
"""Subtitle generation E2E tests.

Tests the POST /v1/audio/subtitles endpoint.
Requires the server to be running on port 8100.
"""
from pathlib import Path

import pytest

from utils.client import ASRHTTPClient


@pytest.mark.smoke
@pytest.mark.subtitle
class TestSubtitleBasic:
    """Basic subtitle generation tests."""

    def test_subtitle_fast_mode(self, ensure_server, sample_audio_5s: Path):
        """Fast mode returns valid SRT."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_5s, mode="fast")

            # Should be valid SRT format
            assert "-->" in srt
            lines = srt.strip().split("\n")
            assert lines[0].strip() == "1"

    def test_subtitle_accurate_mode(self, ensure_server, sample_audio_5s: Path):
        """Accurate mode returns valid SRT (requires ForcedAligner)."""
        with ASRHTTPClient() as client:
            try:
                srt = client.subtitle(sample_audio_5s, mode="accurate")
                assert "-->" in srt
            except Exception:
                pytest.skip("ForcedAligner not available")

    def test_subtitle_with_language(self, ensure_server, sample_audio_5s: Path):
        """Subtitle generation accepts language parameter."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_5s, language="English", mode="fast")
            assert "-->" in srt

    def test_subtitle_content_disposition(self, ensure_server, sample_audio_5s: Path):
        """Response has Content-Disposition header for file download."""
        import httpx
        with httpx.Client(base_url="http://localhost:8100", timeout=300) as client:
            with open(sample_audio_5s, "rb") as f:
                response = client.post(
                    "/v1/audio/subtitles",
                    files={"file": (sample_audio_5s.name, f, "audio/wav")},
                    data={"mode": "fast"},
                )
                assert response.status_code == 200
                assert "subtitles.srt" in response.headers.get("content-disposition", "")


@pytest.mark.slow
@pytest.mark.subtitle
class TestSubtitleAdvanced:
    """Advanced subtitle tests for longer audio."""

    def test_subtitle_long_audio(self, ensure_server, sample_audio_long: Path):
        """Long audio (60s) produces multiple subtitle events."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_long, mode="fast")
            # Count subtitle events (numbered entries)
            import re
            event_count = len(re.findall(r"^\d+$", srt, re.MULTILINE))
            # 60s of audio should have multiple events
            assert event_count >= 2

    def test_subtitle_line_length(self, ensure_server, sample_audio_20s: Path):
        """No subtitle line exceeds max_line_chars."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_20s, mode="fast", max_line_chars=42)
            for line in srt.split("\n"):
                # Skip index lines, timestamp lines, and blank lines
                if line.strip() and "-->" not in line and not line.strip().isdigit():
                    assert len(line) <= 42, f"Line too long: {len(line)} chars: '{line}'"

    def test_subtitle_no_overlaps(self, ensure_server, sample_audio_20s: Path):
        """No subtitle events overlap in time."""
        import re
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_20s, mode="fast")
            # Parse timestamps
            timestamps = re.findall(
                r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                srt,
            )
            for i in range(len(timestamps) - 1):
                end_current = timestamps[i][1]
                start_next = timestamps[i + 1][0]
                assert end_current <= start_next, f"Overlap: {end_current} > {start_next}"
```

**Step 4: Run E2E tests (server must be running)**

Run: `pytest E2Etest/test_subtitle.py -v -k fast_mode`
Expected: PASS (if server is running)

**Step 5: Commit**

```bash
git add E2Etest/test_subtitle.py E2Etest/utils/client.py E2Etest/conftest.py
git commit -m "test: add E2E tests for subtitle generation endpoint"
```

---

## Task 10: Documentation Updates

Update CLAUDE.md, CHANGELOG.md, README.md, and LEARNING_LOG.md per project rules.

**Files:**
- Modify: `CLAUDE.md` — add subtitle endpoint to API table, add FORCED_ALIGNER_ID to env vars
- Modify: `CHANGELOG.md` — add entry
- Modify: `LEARNING_LOG.md` — add design decisions
- Modify: `README.md` — add subtitle usage example

**Step 1: Update CLAUDE.md**

Add to the API Endpoints table:

```
| `/v1/audio/subtitles` | POST | SRT subtitle generation (accurate + fast modes) |
```

Add to Key Environment Variables table:

```
| `FORCED_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace model for word-level alignment |
```

Add to Common Commands:

```bash
# Generate subtitles (fast mode)
curl -X POST http://localhost:8100/v1/audio/subtitles -F "file=@audio.wav" -F "mode=fast" -o subtitles.srt

# Generate subtitles (accurate mode, requires ForcedAligner)
curl -X POST http://localhost:8100/v1/audio/subtitles -F "file=@audio.wav" -F "mode=accurate" -o subtitles.srt
```

**Step 2: Update CHANGELOG.md**

Add new version entry at the top.

**Step 3: Update LEARNING_LOG.md**

Add entry explaining why subtitle generation is a separate module, the two-mode design decision, and ForcedAligner lifecycle integration.

**Step 4: Update README.md**

Add subtitle section with usage example and API parameters.

**Step 5: Commit**

```bash
git add CLAUDE.md CHANGELOG.md LEARNING_LOG.md README.md
git commit -m "docs: add subtitle generation documentation"
```
