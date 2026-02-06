"""Core data models for Parlona Core."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    """Status of a processing job."""
    queued = "queued"
    stt_in_progress = "stt_in_progress"
    stt_done = "stt_done"
    stt_failed = "stt_failed"
    summary_in_progress = "summary_in_progress"
    summary_done = "summary_done"
    postprocess_in_progress = "postprocess_in_progress"
    done = "done"
    failed = "failed"


@dataclass
class Word:
    """Word-level timestamp information."""
    start: float
    end: float
    text: str
    probability: Optional[float] = None
    channel: Optional[int] = None


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with speaker and timing information."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    channel: Optional[int] = None
    confidence: Optional[float] = None
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: list[TranscriptionSegment]
    language: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    label: str  # "positive", "negative", or "neutral"
    score: float  # 0.0 to 1.0


@dataclass
class ProcessResult:
    """Complete processing result from the pipeline."""
    transcript: str
    segments: list[TranscriptionSegment]
    summary: str
    headline: str
    language: str
    sentiment: SentimentResult
    entities: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


# Pydantic models for server mode (optional)
class JobMetadata(BaseModel):
    """Job metadata for tracking processing status (server mode only)."""
    job_id: str
    audio_path: str
    status: JobStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    dummy_transcript: Optional[str] = None
    dummy_summary: Optional[str] = None
    dummy_headline: Optional[str] = None
    dummy_tags: Optional[list[str]] = None
    delivered: bool = False
    notes: Optional[str] = None
    extra_meta: Optional[dict[str, Any]] = None
    stt_text: Optional[str] = None
    stt_language: Optional[str] = None
    stt_segments: Optional[list[dict[str, Any]]] = None
    stt_diarization_mode: Optional[str] = None
    stt_engine: Optional[str] = None
    stt_metadata: Optional[dict[str, Any]] = None
    stt_error: Optional[str] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    entities: Optional[dict[str, Any]] = None


class QueueMessage(BaseModel):
    """Message for Redis queue (server mode only)."""
    job_id: str
    audio_path: Optional[str] = None
