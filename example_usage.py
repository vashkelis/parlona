"""Example usage of parlona package.

This demonstrates how users would interact with the package.
"""

import os

# Set environment variable to work around OpenMP issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def example_basic_usage():
    """Example: Basic usage with defaults."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    import parlona
    
    print(f"Parlona version: {parlona.__version__}")
    print("\nUsage:")
    print("  result = parlona.process('audio.wav')")
    print("  print(result.transcript)")
    print("  print(result.summary)")
    print("  print(result.headline)")
    print("  print(f'Sentiment: {result.sentiment.label} ({result.sentiment.score})')")
    print("\nNote: Requires audio file and OPENAI_API_KEY to run")
    print()

def example_advanced_usage():
    """Example: Advanced usage with custom configuration."""
    print("=" * 60)
    print("Example 2: Advanced Usage")
    print("=" * 60)
    
    from parlona import CallProcessor, STTConfig, LLMConfig
    
    print("Custom STT Configuration:")
    stt_config = STTConfig(
        model_name="Systran/faster-whisper-medium",
        device="cuda",  # Use GPU if available
        diarization_mode="stereo_channels",  # Separate speakers by channel
        speaker_mapping={0: "agent", 1: "customer"}
    )
    print(f"  Model: {stt_config.model_name}")
    print(f"  Device: {stt_config.resolved_device}")
    print(f"  Diarization: {stt_config.diarization_mode}")
    
    print("\nCustom LLM Configuration:")
    llm_config = LLMConfig(
        backend="openai",
        model="gpt-4o-mini"
    )
    print(f"  Backend: {llm_config.backend}")
    print(f"  Model: {llm_config.model}")
    
    print("\nUsage:")
    print("  processor = CallProcessor(stt_config=stt_config, llm_config=llm_config)")
    print("  result = processor.process('audio.wav')")
    print()

def example_modular_usage():
    """Example: Using STT and LLM separately."""
    print("=" * 60)
    print("Example 3: Modular Usage")
    print("=" * 60)
    
    from parlona.stt import STTEngine, STTConfig
    from parlona.llm import LLMClient, LLMConfig
    
    print("Using STT Engine separately:")
    print("  stt_engine = STTEngine(STTConfig(model_name='Systran/faster-whisper-tiny'))")
    print("  transcription = stt_engine.transcribe('audio.wav')")
    print("  print(transcription.text)")
    print("  print(transcription.language)")
    print("  for segment in transcription.segments:")
    print("      print(f'[{segment.speaker}] {segment.text}')")
    
    print("\nUsing LLM Client separately:")
    print("  llm_client = LLMClient(LLMConfig(backend='openai'))")
    print("  summary, headline, lang, sentiment, entities, score = \\")
    print("      llm_client.summarize_with_headline(transcript)")
    print()

def example_data_models():
    """Example: Working with data models."""
    print("=" * 60)
    print("Example 4: Data Models")
    print("=" * 60)
    
    from parlona.models import (
        ProcessResult,
        SentimentResult,
        TranscriptionSegment,
        Word,
    )
    
    print("Available Data Models:")
    print("  - ProcessResult: Complete pipeline result")
    print("  - TranscriptionResult: STT output")
    print("  - TranscriptionSegment: Individual speech segment")
    print("  - Word: Word-level timestamp")
    print("  - SentimentResult: Sentiment analysis")
    
    print("\nExample ProcessResult structure:")
    print("  result.transcript (str)")
    print("  result.segments (List[TranscriptionSegment])")
    print("  result.summary (str)")
    print("  result.headline (str)")
    print("  result.language (str)")
    print("  result.sentiment (SentimentResult)")
    print("    - result.sentiment.label (str: 'positive'/'negative'/'neutral')")
    print("    - result.sentiment.score (float: 0.0-1.0)")
    print("  result.entities (dict)")
    print("  result.metadata (dict)")
    print()

def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 14 + "PARLONA - USAGE EXAMPLES" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    example_basic_usage()
    example_advanced_usage()
    example_modular_usage()
    example_data_models()
    
    print("=" * 60)
    print("Installation Instructions")
    print("=" * 60)
    print("\n1. Install the package:")
    print("   pip install parlona")
    print("\n2. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("\n3. Run your script:")
    print("   python your_script.py")
    print("\n4. Optional: Use other LLM backends (Groq, vLLM, Ollama)")
    print("   result = parlona.process('audio.wav', llm_backend='groq')")
    print()
    print("=" * 60)
    print("For more information, see README.md")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
