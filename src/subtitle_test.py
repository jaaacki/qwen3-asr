"""Unit tests for subtitle generation module."""
from unittest.mock import patch, MagicMock

import numpy as np

from subtitle import (
    format_srt, SubtitleEvent, segment_subtitles, WordTimestamp,
    enforce_timing, load_aligner, unload_aligner, align_audio,
    estimate_word_timestamps, generate_srt_from_results,
    _tokenize, _is_cjk,
)


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


def test_format_timestamp_boundary_millis():
    """Timestamp handles values near whole seconds (millis=1000 edge case)."""
    from subtitle import _format_timestamp
    # 2.9999997 should round to 00:00:03,000 not 00:00:02,1000
    assert _format_timestamp(2.9999997) == "00:00:03,000"
    # 59.9999 should carry over to 01:00:00,000 area
    assert _format_timestamp(59.9999) == "00:01:00,000"
    # 3599.9999 should carry over to 01:00:00,000
    assert _format_timestamp(3599.9999) == "01:00:00,000"
    # Exact values should work fine
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(1.0) == "00:00:01,000"


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


def test_tokenize_english():
    """English text splits by whitespace."""
    assert _tokenize("Hello world") == ["Hello", "world"]


def test_tokenize_chinese():
    """Chinese text splits by character."""
    tokens = _tokenize("你好世界")
    assert tokens == ["你", "好", "世", "界"]


def test_tokenize_japanese():
    """Japanese text splits by character."""
    tokens = _tokenize("こんにちは")
    assert tokens == ["こ", "ん", "に", "ち", "は"]


def test_tokenize_mixed_cjk_latin():
    """Mixed CJK/Latin text splits CJK chars individually, keeps Latin words."""
    tokens = _tokenize("Hello你好world世界")
    assert tokens == ["Hello", "你", "好", "world", "世", "界"]


def test_is_cjk_detection():
    """CJK detection identifies Chinese, Japanese, and non-CJK text."""
    assert _is_cjk("你好世界")
    assert _is_cjk("こんにちは")
    assert _is_cjk("カタカナ")
    assert not _is_cjk("Hello world")
    assert not _is_cjk("Bonjour le monde")


def test_segment_cjk_text():
    """CJK characters are segmented without spaces."""
    words = [
        WordTimestamp("你", 0.0, 0.2),
        WordTimestamp("好", 0.2, 0.4),
        WordTimestamp("世", 0.4, 0.6),
        WordTimestamp("界", 0.6, 0.8),
        WordTimestamp("。", 0.8, 1.0),
    ]
    events = segment_subtitles(words)
    assert len(events) == 1
    assert events[0].text == "你好世界。"
    assert " " not in events[0].text


def test_estimate_timestamps_cjk():
    """CJK text splits by character for timestamp estimation."""
    words = estimate_word_timestamps("你好世界", 0.0, 2.0)
    assert len(words) == 4
    assert words[0].text == "你"
    assert words[3].text == "界"
    # Each char should get equal time (all 1 char)
    for w in words:
        assert abs((w.end - w.start) - 0.5) < 0.01


def test_segment_empty_word_text():
    """Empty-string WordTimestamp does not crash segment_subtitles."""
    words = [
        WordTimestamp("Hello", 0.0, 0.5),
        WordTimestamp("", 0.5, 0.6),  # empty text
        WordTimestamp("world.", 0.6, 1.0),
    ]
    events = segment_subtitles(words)
    assert len(events) >= 1


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


def test_split_two_lines_no_valid_split():
    """When no split keeps both lines within limit, keep as single line."""
    from subtitle import _split_into_two_lines
    # A single very long word that exceeds max_line_chars
    long_text = "Superlongwordthatcannotbesplit anotherverylongwordhere"
    result = _split_into_two_lines(long_text, max_line_chars=10)
    # Both parts exceed 10 chars, so should keep as single line
    assert "\n" not in result


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
    assert gap >= 0.083 - 1e-9


def test_enforce_no_overlap():
    """Overlapping subtitles are fixed by truncating the first."""
    events = [
        SubtitleEvent(index=1, start=1.0, end=3.0, text="First"),
        SubtitleEvent(index=2, start=2.5, end=4.0, text="Second"),  # overlaps
    ]
    result = enforce_timing(events)
    assert result[0].end <= result[1].start


def test_enforce_timing_no_negative_duration():
    """Gap/overlap fix never produces negative-duration subtitles."""
    events = [
        SubtitleEvent(index=1, start=1.0, end=1.8, text="A"),  # short
        SubtitleEvent(index=2, start=1.05, end=2.0, text="B"),  # nearly overlapping
    ]
    result = enforce_timing(events)
    for e in result:
        assert (e.end - e.start) >= 0.833 - 1e-9, (
            f"Subtitle {e.index} has negative or sub-minimum duration: "
            f"{e.end - e.start:.3f}s"
        )


def test_enforce_timing_tightly_packed():
    """Tightly packed events (50ms apart) all maintain min duration."""
    events = [
        SubtitleEvent(index=1, start=0.0, end=0.5, text="A"),
        SubtitleEvent(index=2, start=0.05, end=0.55, text="B"),  # 50ms apart
        SubtitleEvent(index=3, start=0.10, end=0.60, text="C"),  # 50ms apart
    ]
    result = enforce_timing(events)
    for e in result:
        duration = e.end - e.start
        assert duration >= 0.833 - 1e-9, (
            f"Subtitle {e.index} duration {duration:.3f}s < min 0.833s"
        )


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


def test_load_aligner_sets_global():
    """load_aligner sets the module-level _aligner global."""
    import subtitle
    subtitle._aligner = None  # ensure clean state

    mock_model = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_qwen_asr = MagicMock()
    mock_qwen_asr.Qwen3ForcedAligner = mock_cls

    with patch.dict("sys.modules", {"torch": mock_torch, "qwen_asr": mock_qwen_asr}):
        load_aligner()
        mock_cls.from_pretrained.assert_called_once()

    # Cleanup
    subtitle._aligner = None


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

    mock_torch = MagicMock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        audio = np.zeros(16000, dtype=np.float32)
        words = align_audio(audio, 16000, "Hello world", "en")

    assert len(words) == 2
    assert words[0].text == "Hello"
    assert words[0].start == 0.1
    assert words[1].text == "world"

    # Cleanup
    subtitle._aligner = None


def test_align_audio_long_audio_fallback():
    """Long audio (>5 min) falls back to heuristic when aligner fails on chunks."""
    import subtitle

    mock_aligner = MagicMock()
    mock_aligner.align.side_effect = Exception("chunk alignment failed")
    subtitle._aligner = mock_aligner

    mock_torch = MagicMock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        # 6 minutes of audio at 16kHz
        audio = np.zeros(16000 * 360, dtype=np.float32)
        words = align_audio(audio, 16000, "some text here", "en")

    # Should fall back to estimate_word_timestamps instead of raising
    assert len(words) > 0
    assert all(isinstance(w, WordTimestamp) for w in words)

    subtitle._aligner = None


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


def test_generate_srt_fast_mode():
    """Fast mode produces valid SRT from mock ASR results."""
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


def test_generate_srt_accurate_happy_path():
    """Accurate mode with mocked aligner produces valid SRT."""
    import subtitle

    # Mock aligner with word timestamps
    mock_ts = [
        MagicMock(text="Hello", start_time=0.1, end_time=0.5),
        MagicMock(text="world.", start_time=0.6, end_time=1.0),
    ]
    mock_align_result = MagicMock()
    mock_align_result.time_stamps = mock_ts
    mock_aligner = MagicMock()
    mock_aligner.align.return_value = [mock_align_result]
    subtitle._aligner = mock_aligner

    mock_result = MagicMock()
    mock_result.text = "Hello world."
    mock_result.language = "en"

    mock_torch = MagicMock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        audio = np.zeros(16000, dtype=np.float32)
        srt = generate_srt_from_results(
            results=[mock_result], audio=audio, sr=16000, mode="accurate",
        )

    assert "1\n" in srt
    assert "-->" in srt
    assert "Hello world." in srt

    # Cleanup
    subtitle._aligner = None


def test_generate_srt_accurate_without_aligner_raises():
    """Accurate mode raises RuntimeError when aligner is not loaded."""
    import pytest
    import subtitle
    subtitle._aligner = None

    mock_result = MagicMock()
    mock_result.text = "Hello world."
    mock_result.language = "en"

    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match="ForcedAligner not loaded"):
        generate_srt_from_results(
            results=[mock_result], audio=audio, sr=16000, mode="accurate",
        )


def test_subtitle_endpoint_exists():
    """The /v1/audio/subtitles endpoint is registered in server.py."""
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    import server
    routes = [r.path for r in server.app.routes]
    assert "/v1/audio/subtitles" in routes


def test_subtitle_endpoint_in_gateway():
    """The /v1/audio/subtitles endpoint is registered in gateway.py."""
    pytest = __import__("pytest")
    pytest.importorskip("aiohttp")
    import gateway
    routes = [r.path for r in gateway.app.routes]
    assert "/v1/audio/subtitles" in routes


def test_subtitle_endpoint_in_worker():
    """The /subtitles endpoint is registered in worker.py."""
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    import worker
    routes = [r.path for r in worker.app.routes]
    assert "/subtitles" in routes
