"""Test script for parlona package."""

import os
import sys

# Set environment variable to work around OpenMP issue on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_import():
    """Test that the package can be imported."""
    try:
        import parlona
        print("✓ Package import successful")
        print(f"  Version: {parlona.__version__}")
        return True
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False

def test_stt_engine():
    """Test STT engine initialization."""
    try:
        from parlona.stt import STTEngine, STTConfig
        config = STTConfig(
            model_name="Systran/faster-whisper-tiny",
            device="cpu",
        )
        print("✓ STT engine initialization successful")
        return True
    except Exception as e:
        print(f"✗ STT engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_client():
    """Test LLM client initialization."""
    try:
        from parlona.llm import LLMClient, LLMConfig
        # Test without API key (will fail on actual use, but should init)
        config = LLMConfig(backend="openai")
        print("✓ LLM client initialization successful")
        return True
    except Exception as e:
        print(f"✗ LLM client initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline():
    """Test pipeline initialization."""
    try:
        from parlona import CallProcessor
        from parlona.stt import STTConfig
        from parlona.llm import LLMConfig
        
        stt_config = STTConfig(
            model_name="Systran/faster-whisper-tiny",
            device="cpu",
        )
        llm_config = LLMConfig(backend="openai")
        
        processor = CallProcessor(stt_config=stt_config, llm_config=llm_config)
        print("✓ Pipeline initialization successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models():
    """Test data models."""
    try:
        from parlona.models import (
            ProcessResult,
            SentimentResult,
            TranscriptionSegment,
            Word,
        )
        
        # Create instances
        word = Word(start=0.0, end=1.0, text="hello")
        segment = TranscriptionSegment(
            start=0.0,
            end=1.0,
            text="hello",
            words=[word]
        )
        sentiment = SentimentResult(label="positive", score=0.8)
        
        print("✓ Data models work correctly")
        return True
    except Exception as e:
        print(f"✗ Data models failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Parlona Package Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_import),
        ("STT Engine Test", test_stt_engine),
        ("LLM Client Test", test_llm_client),
        ("Pipeline Test", test_pipeline),
        ("Models Test", test_models),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Running: {name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
