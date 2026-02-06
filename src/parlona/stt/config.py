"""Configuration for STT engine."""

from __future__ import annotations

import json
import os
from typing import Optional


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and visible_devices != "-1":
        return True
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class STTConfig:
    """Configuration for STT engine."""
    
    def __init__(
        self,
        model_name: str = "Systran/faster-whisper-small",
        device: str = "auto",
        compute_type: str = "float16",
        diarization_mode: str = "none",
        speaker_mapping: Optional[dict[int, str]] = None,
        language: Optional[str] = None,
        beam_size: int = 5,
        temperature: float = 0.0,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 500,
        initial_prompt: Optional[str] = None,
        model_dir: str = "/tmp/whisper_models",
        local_files_only: bool = False,
        # Alignment parameters
        alignment_overlap_eps: float = 0.2,
        alignment_pad_left: float = 0.2,
        alignment_pad_right: float = 0.2,
        alignment_min_word_duration: float = 0.2,
        alignment_min_segment_duration: float = 0.1,
        alignment_gap_threshold: float = 1.0,
        alignment_merge_threshold: float = 2.5,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.diarization_mode = diarization_mode
        self.speaker_mapping = speaker_mapping or {0: "agent", 1: "customer"}
        self.language = language
        self.beam_size = beam_size
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.vad_min_silence_ms = vad_min_silence_ms
        self.initial_prompt = initial_prompt
        self.model_dir = model_dir
        self.local_files_only = local_files_only
        
        # Alignment settings
        self.alignment_overlap_eps = alignment_overlap_eps
        self.alignment_pad_left = alignment_pad_left
        self.alignment_pad_right = alignment_pad_right
        self.alignment_min_word_duration = alignment_min_word_duration
        self.alignment_min_segment_duration = alignment_min_segment_duration
        self.alignment_gap_threshold = alignment_gap_threshold
        self.alignment_merge_threshold = alignment_merge_threshold
    
    @property
    def resolved_device(self) -> str:
        """Resolve device to actual value (auto -> cuda/cpu)."""
        if self.device.lower() in {"auto", ""}:
            return "cuda" if _cuda_available() else "cpu"
        return self.device
    
    @property
    def resolved_compute_type(self) -> str:
        """Resolve compute type based on device."""
        compute = self.compute_type.lower()
        device = self.resolved_device
        if device == "cpu":
            if compute in {"int8", "int8_float32"}:
                return compute
            return "int8"
        return compute
