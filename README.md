# Parlona

Call analytics pipeline for speech-to-text, summarization, and insights extraction.

## Installation

```bash
pip install parlona
```

## Quick Start

```python
import parlona

# Process an audio file
result = parlona.process("call.wav")

# Access results
print(result.transcript)
print(result.summary)
print(result.headline)
print(f"Sentiment: {result.sentiment.label} ({result.sentiment.score})")
print(f"Entities: {result.entities}")
```

## Features

- **Speech-to-Text**: Powered by faster-whisper with stereo channel diarization
- **Call Summarization**: LLM-powered summaries via OpenAI, Groq, vLLM, or Ollama
- **Sentiment Analysis**: Automatic sentiment detection and scoring
- **Entity Extraction**: Named entity recognition with speaker attribution
- **Multi-language Support**: Automatic language detection

## Advanced Usage

```python
from parlona import CallProcessor, STTConfig, LLMConfig

# Custom configuration
stt_config = STTConfig(
    model_name="Systran/faster-whisper-medium",
    device="cuda",
    diarization_mode="stereo_channels",
    speaker_mapping={0: "agent", 1: "customer"}
)

llm_config = LLMConfig(
    backend="openai",
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Create processor and process
processor = CallProcessor(stt_config=stt_config, llm_config=llm_config)
result = processor.process("call.wav")
```

## Modular Usage

Use STT and LLM components separately:

```python
from parlona.stt import STTEngine, STTConfig
from parlona.llm import LLMClient, LLMConfig

# STT only
stt_engine = STTEngine(STTConfig())
transcription = stt_engine.transcribe("audio.wav")

# LLM only
llm_client = LLMClient(LLMConfig(backend="openai"))
summary, headline, lang, sentiment, entities, score = \
    llm_client.summarize_with_headline(transcript)
```

## Requirements

- Python 3.9+
- OpenAI API key (or other LLM backend)

## License

MIT
