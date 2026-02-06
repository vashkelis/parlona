"""Parlona - Call analytics pipeline.

A Python package for processing call audio through speech-to-text,
summarization, and insights extraction.
"""

__version__ = "1.0.0"

import logging
import os
from typing import Optional

from parlona.models import (
    ProcessResult,
    SentimentResult,
    TranscriptionSegment,
    TranscriptionResult,
    Word,
)
from parlona.pipeline import CallProcessor
from parlona.stt import STTEngine, STTConfig
from parlona.llm import LLMClient, LLMConfig

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    "process",
    "CallProcessor",
    "ProcessResult",
    "SentimentResult",
    "TranscriptionSegment",
    "TranscriptionResult",
    "Word",
    "STTEngine",
    "STTConfig",
    "LLMClient",
    "LLMConfig",
]


def process(
    audio_path: str,
    *,
    stt_model: str = "Systran/faster-whisper-small",
    stt_device: str = "auto",
    diarization_mode: str = "none",
    speaker_mapping: Optional[dict[int, str]] = None,
    llm_backend: str = "openai",
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> ProcessResult:
    """Process an audio file through the complete pipeline.
    
    This is the simplest way to use Parlona. It transcribes audio,
    generates a summary, analyzes sentiment, and extracts entities.
    
    Args:
        audio_path: Path to the audio file to process
        stt_model: Whisper model to use (default: "Systran/faster-whisper-small")
        stt_device: Device for STT ("auto", "cpu", or "cuda")
        diarization_mode: "none" or "stereo_channels" for speaker separation
        speaker_mapping: Dict mapping channel numbers to speaker labels
        llm_backend: LLM backend ("openai", "groq", "vllm", "ollama")
        llm_api_key: API key for LLM (if not set in environment)
        llm_model: Override default model for the LLM backend
        
    Returns:
        ProcessResult containing transcript, summary, sentiment, entities, etc.
        
    Example:
        >>> import parlona
        >>> result = parlona.process("call.wav")
        >>> print(result.transcript)
        >>> print(result.summary)
        >>> print(result.headline)
        >>> print(result.sentiment.label, result.sentiment.score)
        >>> print(result.entities)
    """
    # Create configurations
    stt_config = STTConfig(
        model_name=stt_model,
        device=stt_device,
        diarization_mode=diarization_mode,
        speaker_mapping=speaker_mapping,
    )
    
    llm_config = LLMConfig(
        backend=llm_backend,
        api_key=llm_api_key,
        model=llm_model,
    )
    
    # Create processor and run
    processor = CallProcessor(stt_config=stt_config, llm_config=llm_config)
    return processor.process(audio_path)
