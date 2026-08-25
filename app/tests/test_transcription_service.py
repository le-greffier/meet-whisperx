"""Tests for the root of what the transcription pipeline returns."""

from unittest.mock import MagicMock

from services import transcription

# Measured: alignment rebuilds the text of a segment from its words, and only the
# first one keeps the leading space the transcription pass wrote.
_SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": " Hello world.", "speaker": "SPEAKER_00"},
    {"start": 1.6, "end": 2.4, "text": "Goodbye.", "speaker": "SPEAKER_01"},
]


def _pipeline(monkeypatch):
    """Stand in for the three passes, which need weights and a device."""
    monkeypatch.setattr(
        transcription,
        "_transcribe_audio",
        lambda audio, settings, language: {"segments": _SEGMENTS, "language": "en"},
    )
    monkeypatch.setattr(
        transcription,
        "_align_transcription",
        # Alignment returns segments and words, and no language.
        lambda audio, result, settings: {"segments": _SEGMENTS, "word_segments": []},
    )
    monkeypatch.setattr(
        transcription,
        "_diarize_and_assign_speakers",
        lambda audio, result, settings: result,
    )


def test_the_root_carries_the_language_the_pass_detected(monkeypatch):
    """Alignment drops it, so the language is put back from the pass that found it."""
    _pipeline(monkeypatch)

    result = transcription.transcribe([0] * 16000, MagicMock())

    assert result["language"] == "en"


def test_the_length_is_the_audio_and_not_the_last_word(monkeypatch):
    """Two seconds of samples, whatever the last segment ends at."""
    _pipeline(monkeypatch)

    result = transcription.transcribe([0] * 32000, MagicMock())

    assert result["duration"] == 2.0


def test_the_text_is_the_segments_joined_and_nothing_else(monkeypatch):
    """A separator where none exists, and the leading space the model wrote kept."""
    _pipeline(monkeypatch)

    result = transcription.transcribe([0] * 16000, MagicMock())

    assert result["text"] == " Hello world. Goodbye."


def test_a_space_a_segment_already_carries_is_not_doubled(monkeypatch):
    """The separator fills a gap; it never normalises what is there."""
    _pipeline(monkeypatch)
    monkeypatch.setattr(
        transcription,
        "_diarize_and_assign_speakers",
        lambda audio, result, settings: {
            "segments": [{"text": "Hello world."}, {"text": " Goodbye."}],
            "word_segments": [],
        },
    )

    result = transcription.transcribe([0] * 16000, MagicMock())

    assert result["text"] == "Hello world. Goodbye."


def test_what_the_pipeline_already_returned_is_untouched(monkeypatch):
    """The three fields are added beside the segments, never in their place."""
    _pipeline(monkeypatch)

    result = transcription.transcribe([0] * 16000, MagicMock())

    assert result["segments"] == _SEGMENTS
    assert result["word_segments"] == []
