"""STT Engine implementation using faster-whisper."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from parlona.models import Word, TranscriptionSegment, TranscriptionResult
from parlona.stt.config import STTConfig
from parlona.stt.diarization import (
    analyze_audio,
    cleanup_temp_files,
    resolve_speaker_labels,
    split_stereo_to_mono,
)
from parlona.stt.alignment import (
    AlignmentConfig,
    DiarizationSegment,
    align_diarization_with_words,
    Word as AlignmentWord,
)

logger = logging.getLogger("parlona.stt")


class STTEngine:
    """Speech-to-text engine using faster-whisper."""
    
    def __init__(self, config: Optional[STTConfig] = None):
        """Initialize the STT engine.
        
        Args:
            config: STT configuration. If None, uses defaults.
        """
        self.config = config or STTConfig()
        
        logger.info(
            "Initializing STT engine: model=%s device=%s compute_type=%s",
            self.config.model_name,
            self.config.resolved_device,
            self.config.resolved_compute_type,
        )
        
        self.model = WhisperModel(
            self.config.model_name,
            device=self.config.resolved_device,
            compute_type=self.config.resolved_compute_type,
            download_root=self.config.model_dir,
            local_files_only=self.config.local_files_only,
        )
    
    def transcribe(
        self,
        audio_path: str,
        diarization_mode: Optional[str] = None,
        speaker_mapping: Optional[dict[int, str]] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            diarization_mode: Override default diarization mode
            speaker_mapping: Override default speaker mapping
            
        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        mode = diarization_mode or self.config.diarization_mode
        
        if mode == "stereo_channels":
            return self._transcribe_stereo(audio_path, speaker_mapping)
        return self._transcribe_standard(audio_path)
    
    def _transcribe_standard(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio without diarization."""
        segments, info = self._run_transcription(audio_path)
        
        # Apply word-level alignment
        if segments and any(seg.words for seg in segments):
            segments = self._apply_word_alignment(segments)
        
        segments.sort(key=lambda seg: seg.start)
        
        # Build full text
        interleaved_texts = []
        for segment in segments:
            segment_text = f"[{segment.speaker or 'unknown'}] {segment.text}"
            interleaved_texts.append(segment_text)
        
        text = "\n".join(interleaved_texts)
        metadata = self._build_metadata(info)
        metadata["diarization_mode"] = "none"
        
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=getattr(info, 'language', None),
            metadata=metadata,
        )
    
    def _transcribe_stereo(
        self,
        audio_path: str,
        speaker_mapping: Optional[dict[int, str]] = None,
    ) -> TranscriptionResult:
        """Transcribe stereo audio with channel-based diarization."""
        audio_info = analyze_audio(audio_path)
        if audio_info.channels < 2:
            logger.warning(
                "Audio has %s channel(s); falling back to non-diarized mode",
                audio_info.channels,
            )
            return self._transcribe_standard(audio_path)
        
        # Split stereo audio
        channel_files, temp_files = split_stereo_to_mono(audio_path)
        
        all_segments: List[TranscriptionSegment] = []
        metadata = {
            "diarization_mode": "stereo_channels",
            "channels": {},
            "audio": {
                "channels": audio_info.channels,
                "sample_rate": audio_info.sample_rate,
                "duration": audio_info.duration,
            },
        }
        
        detected_languages = []
        mapping = speaker_mapping or self.config.speaker_mapping
        mapping = resolve_speaker_labels(mapping, channel_files.keys())
        
        try:
            for channel, mono_path in channel_files.items():
                speaker_label = mapping.get(channel, f"speaker_{channel}")
                logger.info(
                    "Transcribing channel %s (%s) from %s",
                    channel,
                    speaker_label,
                    mono_path,
                )
                
                channel_segments, info = self._run_transcription(
                    mono_path, speaker=speaker_label, channel=channel
                )
                
                channel_language = getattr(info, 'language', None)
                if channel_language:
                    detected_languages.append(channel_language)
                
                channel_metadata = self._build_metadata(info)
                if channel_language and "language" not in channel_metadata:
                    channel_metadata["language"] = channel_language
                
                channel_metadata["speaker"] = speaker_label
                metadata["channels"][channel] = channel_metadata
                
                all_segments.extend(channel_segments)
            
            # Apply alignment
            all_segments = self._apply_word_alignment(all_segments)
            all_segments.sort(key=lambda seg: seg.start)
            
            # Build full text
            interleaved_texts = []
            for segment in all_segments:
                segment_text = f"[{segment.speaker}] {segment.text}"
                interleaved_texts.append(segment_text)
            
            full_text = "\n".join(interleaved_texts)
            
            # Get primary language
            primary_language = None
            for channel, channel_meta in metadata["channels"].items():
                if "language" in channel_meta and channel_meta["language"]:
                    primary_language = channel_meta["language"]
                    break
            
            if primary_language is None and detected_languages:
                primary_language = detected_languages[0]
            
            return TranscriptionResult(
                text=full_text,
                segments=all_segments,
                language=primary_language,
                metadata=metadata,
            )
        
        finally:
            cleanup_temp_files(temp_files)
    
    def _run_transcription(
        self,
        audio_path: str,
        speaker: Optional[str] = None,
        channel: Optional[int] = None,
    ) -> Tuple[List[TranscriptionSegment], Any]:
        """Run transcription on audio file."""
        generator, info = self.model.transcribe(
            audio_path,
            beam_size=self.config.beam_size,
            temperature=self.config.temperature,
            vad_filter=self.config.vad_filter,
            vad_parameters={"min_silence_duration_ms": self.config.vad_min_silence_ms},
            language=self.config.language,
            initial_prompt=self.config.initial_prompt,
            word_timestamps=True,
        )
        
        segments: List[TranscriptionSegment] = []
        for seg in generator:
            text = seg.text.strip()
            if not text:
                continue
            
            # Extract word-level timestamps
            words = []
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    words.append(Word(
                        start=float(w.start),
                        end=float(w.end),
                        text=w.word.strip(),
                        probability=getattr(w, 'probability', None),
                        channel=channel,
                    ))
            
            segment = TranscriptionSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                speaker=speaker,
                channel=channel,
                confidence=self._confidence(seg),
                words=words,
            )
            segments.append(segment)
        
        return segments, info
    
    def _apply_word_alignment(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """Apply word-level alignment to clean up timestamps."""
        # Extract all words
        all_words: List[AlignmentWord] = []
        for seg in segments:
            for word in seg.words:
                all_words.append(AlignmentWord(
                    start=word.start,
                    end=word.end,
                    text=word.text,
                    probability=word.probability,
                    channel=word.channel,
                ))
        
        # Create diarization segments
        diar_segments: List[DiarizationSegment] = []
        for seg in segments:
            diar_segments.append(DiarizationSegment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker or "speaker_unknown",
                channel=seg.channel,
            ))
        
        # Get alignment configuration
        config = AlignmentConfig(
            overlap_eps=self.config.alignment_overlap_eps,
            pad_left=self.config.alignment_pad_left,
            pad_right=self.config.alignment_pad_right,
            min_word_duration=self.config.alignment_min_word_duration,
            min_segment_duration=self.config.alignment_min_segment_duration,
            gap_threshold=self.config.alignment_gap_threshold,
            merge_threshold=self.config.alignment_merge_threshold,
        )
        
        # Apply alignment
        cleaned_segments = align_diarization_with_words(
            words=all_words,
            diar_segments=diar_segments,
            config=config,
        )
        
        logger.info(
            "Alignment cleaned %d segments → %d segments",
            len(segments),
            len(cleaned_segments),
        )
        
        # Convert back to TranscriptionSegment
        result_segments: List[TranscriptionSegment] = []
        for cleaned in cleaned_segments:
            words = [Word(
                start=w.start,
                end=w.end,
                text=w.text,
                probability=w.probability,
                channel=w.channel,
            ) for w in cleaned.words]
            
            result_segments.append(TranscriptionSegment(
                start=cleaned.start,
                end=cleaned.end,
                text=cleaned.text,
                speaker=cleaned.speaker,
                channel=cleaned.channel,
                confidence=cleaned.confidence,
                words=words,
            ))
        
        return result_segments
    
    @staticmethod
    def _confidence(segment: Any) -> Optional[float]:
        """Extract confidence from segment."""
        no_speech = getattr(segment, "no_speech_prob", None)
        if no_speech is None:
            return None
        return max(0.0, min(1.0, 1.0 - no_speech))
    
    @staticmethod
    def _build_metadata(info: Any) -> Dict[str, Any]:
        """Build metadata from transcription info."""
        if not info:
            return {}
        
        metadata: Dict[str, Any] = {}
        for attr in ("language", "language_probability", "duration", "duration_after_vad"):
            value = getattr(info, attr, None)
            if value is not None:
                metadata[attr] = value
        
        if "language" not in metadata and hasattr(info, 'language'):
            metadata["language"] = info.language
        
        return metadata
