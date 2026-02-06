"""Call processing pipeline."""

import logging
from typing import Optional

from parlona.models import ProcessResult, SentimentResult
from parlona.stt import STTEngine, STTConfig
from parlona.llm import LLMClient, LLMConfig

logger = logging.getLogger("parlona.pipeline")


class CallProcessor:
    """Orchestrates STT and LLM processing."""
    
    def __init__(
        self,
        stt_config: Optional[STTConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        """Initialize the call processor.
        
        Args:
            stt_config: STT configuration. If None, uses defaults.
            llm_config: LLM configuration. If None, uses defaults.
        """
        self.stt_engine = STTEngine(stt_config)
        self.llm_client = LLMClient(llm_config)
        logger.info("CallProcessor initialized")
    
    def process(self, audio_path: str) -> ProcessResult:
        """Process audio file through full pipeline.
        
        Args:
            audio_path: Path to audio file to process
            
        Returns:
            ProcessResult with transcript, summary, sentiment, entities, etc.
        """
        logger.info("Processing audio file: %s", audio_path)
        
        # Step 1: Transcribe audio
        logger.info("Step 1: Transcribing audio...")
        stt_result = self.stt_engine.transcribe(audio_path)
        logger.info("Transcription complete: %d segments, %s", 
                   len(stt_result.segments), stt_result.language)
        
        # Step 2: Generate insights
        logger.info("Step 2: Generating summary and insights...")
        summary, headline, language, sentiment_label, entities, sentiment_score = \
            self.llm_client.summarize_with_headline(stt_result.text)
        logger.info("Insights complete: sentiment=%s (%.2f)", sentiment_label, sentiment_score)
        
        # Step 3: Combine results
        result = ProcessResult(
            transcript=stt_result.text,
            segments=stt_result.segments,
            summary=summary,
            headline=headline,
            language=language or stt_result.language or "unknown",
            sentiment=SentimentResult(
                label=sentiment_label,
                score=sentiment_score,
            ),
            entities=entities,
            metadata={
                "stt_metadata": stt_result.metadata,
                "num_segments": len(stt_result.segments),
            },
        )
        
        logger.info("Processing complete for: %s", audio_path)
        return result
