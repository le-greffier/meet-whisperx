import logging

import numpy as np
import whisperx

from utils.config import Settings, get_device
from utils.lifespan import pipelines

logger = logging.getLogger("api")


def transcribe(
    audio: np.ndarray,
    settings: Settings,
    language: str | None = None,
) -> dict:
    """Run the full transcription pipeline: transcribe, align, and diarize."""
    transcription = _transcribe_audio(audio, settings, language)
    detected = transcription["language"]
    result = _align_transcription(audio, transcription, settings)
    result = _diarize_and_assign_speakers(audio, result, settings)

    # Alignment returns segments and words only, dropping the language the
    # transcription pass detected. The length and the full text are known here and
    # nowhere downstream: the audio is not kept, and a reader adding up segment
    # ends would be reading the last spoken word, not the end of the recording.
    result["language"] = detected
    result["duration"] = len(audio) / whisperx.audio.SAMPLE_RATE
    result["text"] = _joined_text(result["segments"])
    return result


def _joined_text(segments: list[dict]) -> str:
    """Join the segments into the transcript they spell.

    Alignment rebuilds the text of a segment from its words, and only the first one
    keeps the leading space the transcription pass wrote. Joining them raw welds the
    last word of a segment to the first of the next. The separator is the assembly,
    not a correction: no segment is altered, and an existing space is never doubled.
    """
    joined = ""
    for segment in segments:
        text = segment["text"]
        if joined and not joined[-1].isspace() and not text[:1].isspace():
            joined += " "
        joined += text
    return joined


def _transcribe_audio(
    audio: np.ndarray,
    settings: Settings,
    language: str | None,
) -> dict:
    """Run whisperx transcription on audio."""
    logger.info("Starting transcription …")
    result = pipelines.transcribe_model.transcribe(
        audio, batch_size=settings.batch_size, language=language
    )
    logger.info("Transcription done.")
    return result


def _align_transcription(
    audio: np.ndarray,
    transcription_result: dict,
    settings: Settings,
) -> dict:
    """Align transcription segments with audio."""
    logger.info("Aligning transcription …")
    device = get_device()

    language = transcription_result["language"]
    if language in pipelines.align_models:
        align_model, metadata = pipelines.align_models[language]
    else:
        # Weights are downloaded and cached but model does not stay loaded
        align_model, metadata = whisperx.load_align_model(
            language_code=transcription_result["language"], device=device
        )

    aligned = whisperx.align(
        transcription_result["segments"],
        align_model,
        metadata,
        audio,
        device,
        interpolate_method=settings.interpolate_method,
        return_char_alignments=settings.return_char_alignments,
    )
    logger.info("Alignment done.")
    return aligned


def _diarize_and_assign_speakers(
    audio: np.ndarray,
    alignment_result: dict,
    settings: Settings,
) -> dict:
    """Run diarization and assign speakers to words."""
    logger.info("Diarization …")
    diarize_segments = pipelines.diarize_model(audio)
    result = whisperx.assign_word_speakers(
        diarize_segments, alignment_result, fill_nearest=settings.fill_nearest
    )
    logger.info("Diarization done.")
    return result
