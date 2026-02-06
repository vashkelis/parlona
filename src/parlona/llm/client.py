"""LLM client for summarization and entity extraction."""

import json
import logging
from typing import Optional, Tuple, Any

from openai import OpenAI

try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

logger = logging.getLogger("parlona_core.llm")


class LLMConfig:
    """Configuration for LLM client."""
    
    def __init__(
        self,
        backend: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # Set defaults based on backend
        if self.backend == "openai":
            self.base_url = self.base_url or "https://api.openai.com/v1"
            self.model = self.model or "gpt-4o-mini"
        elif self.backend == "groq":
            self.base_url = self.base_url or "https://api.groq.com/openai/v1"
            self.model = self.model or "llama3-8b-8192"
        elif self.backend == "vllm":
            self.base_url = self.base_url or "http://localhost:8000/v1"
            self.model = self.model or "meta-llama/Meta-Llama-3-8B-Instruct"
            self.api_key = self.api_key or "EMPTY"
        elif self.backend == "ollama":
            self.base_url = self.base_url or "http://localhost:11434/v1"
            self.model = self.model or "llama3"
            self.api_key = "ollama"


class LLMClient:
    """Unified client for LLM backends."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client.
        
        Args:
            config: LLM configuration. If None, uses OpenAI with env API key.
        """
        self.config = config or LLMConfig()
        
        if not self.config.api_key and self.config.backend == "openai":
            import os
            self.config.api_key = os.environ.get("OPENAI_API_KEY")
        
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        
        logger.info(
            "Initialized LLM client: backend=%s model=%s",
            self.config.backend,
            self.config.model,
        )
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not HAS_LANGDETECT:
            return "en"
        
        try:
            return detect(text)
        except Exception as e:
            logger.warning("Failed to detect language: %s", e)
            return "en"
    
    def summarize_with_headline(
        self,
        transcript: str,
        max_sentences: int = 4,
    ) -> Tuple[str, str, str, str, dict[str, Any], float]:
        """Generate summary, headline, sentiment, and entities.
        
        Args:
            transcript: The transcript to analyze
            max_sentences: Maximum sentences in summary
            
        Returns:
            Tuple of (summary, headline, language, sentiment_label, entities, sentiment_score)
        """
        if not transcript:
            return "", "No conversation", "en", "neutral", {}, 0.5
        
        language = self.detect_language(transcript)
        logger.info("Detected language: %s", language)
        
        prompt = f"""
Please analyze the following conversation and provide a summary, headline, sentiment analysis, and named entities.
Respond in JSON format with the following fields: "summary", "headline", "sentiment_label", "sentiment_score", and "entities".

Requirements:
1. The summary should be in {max_sentences} sentences or fewer
2. All fields should be in the same language as the conversation
3. The headline should be a single sentence describing the main topic of the call
4. For sentiment_label, choose one of: "positive", "negative", or "neutral"
5. For sentiment_score, provide a value between 0.0 and 1.0 (0.0 = very negative, 1.0 = very positive)
6. For entities, extract all important named entities such as person names, organizations, locations, dates, times, monetary values, etc.
   IMPORTANT: Separate entities by speaker role ("agent" and "customer"). This is critical for attribution.
   Format entities as a dictionary with two keys: "agent" and "customer". 
   Each should contain a dictionary of entity types and lists of values.
   Example: {{
     "agent": {{"PRODUCT": ["Internet 500"], "OFFER": ["10% discount"]}}, 
     "customer": {{"PERSON": ["John Smith"], "LOCATION": ["New York"], "ORGANIZATION": ["Acme Corp"]}}
   }}
7. Focus on the key points, main topics, emotional tone, and important entities discussed

Conversation:
{transcript}

Response format example:
{{
    "summary": "Summary text here...",
    "headline": "Headline text here...",
    "sentiment_label": "positive|negative|neutral",
    "sentiment_score": 0.8,
    "entities": {{
        "agent": {{
            "PRODUCT": ["Fiber Optic"],
            "PERSON": ["Agent Sarah"]
        }},
        "customer": {{
            "PERSON": ["John Smith"],
            "ORGANIZATION": ["ABC Company"],
            "LOCATION": ["New York"]
        }}
    }}
}}
        """.strip()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes conversations and responds in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug("LLM response: %s", response_text)
            
            try:
                result = json.loads(response_text)
                summary = result.get("summary", "").strip()
                headline = result.get("headline", "").strip()
                sentiment_label = result.get("sentiment_label", "neutral").strip().lower()
                sentiment_score = float(result.get("sentiment_score", 0.5))
                entities = result.get("entities", {})
                
                # Validate sentiment
                if sentiment_label not in ["positive", "negative", "neutral"]:
                    sentiment_label = "neutral"
                
                if not (0.0 <= sentiment_score <= 1.0):
                    sentiment_score = 0.5
                
                if not summary or not headline:
                    logger.warning("LLM response missing summary or headline")
                    raise ValueError("Missing summary or headline")
                
                logger.info("Generated summary: %s", summary[:100])
                logger.info("Generated headline: %s", headline)
                logger.info("Sentiment: %s (%.2f)", sentiment_label, sentiment_score)
                
                return summary, headline, language, sentiment_label, entities, sentiment_score
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Failed to parse LLM response: %s", e)
                raise
        
        except Exception as e:
            logger.error("Failed to generate summary: %s", e)
            # Return fallback
            fallback_summary = f"Summary of conversation in {language} (error: {str(e)})"
            fallback_headline = f"Conversation in {language}"
            return fallback_summary, fallback_headline, language, "neutral", {}, 0.5
    
    def summarize(self, transcript: str, max_sentences: int = 4) -> Tuple[str, str]:
        """Summarize transcript (backward compatibility).
        
        Args:
            transcript: Text to summarize
            max_sentences: Max sentences in summary
            
        Returns:
            Tuple of (summary, language)
        """
        summary, _, language, _, _, _ = self.summarize_with_headline(transcript, max_sentences)
        return summary, language
