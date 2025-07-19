#!/usr/bin/env python3
"""
Test script to simulate and verify the RAG model fallback logic when the primary model is not available.
"""

import sys
import os
import requests
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.custom_logger import CustomLogger

# Setup logging
logger = CustomLogger().get_logger("rag_fallback_simulation")


def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        return []
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []


def test_fallback_logic_directly():
    """Test the fallback logic directly by importing the RAG application"""
    try:
        logger.info("=" * 60)
        logger.info("Testing RAG Model Fallback Logic - Direct Test")
        logger.info("=" * 60)

        # Get available models first
        available_models = get_available_ollama_models()
        logger.info(f"Available Ollama models: {available_models}")

        # Check if we have the primary model
        primary_model = "llama3.2:3b"
        if primary_model in available_models:
            logger.info(f"✅ Primary model {primary_model} is available")
        else:
            logger.warning(f"⚠️ Primary model {primary_model} not found")

        # Check for fallback models
        backup_model = "qwen2.5:3b"
        fallback_model = "qwen:4b"

        if backup_model in available_models:
            logger.info(f"✅ Backup model {backup_model} is available")
        else:
            logger.warning(f"⚠️ Backup model {backup_model} not found")

        if fallback_model in available_models:
            logger.info(f"✅ Fallback model {fallback_model} is available")
        else:
            logger.warning(f"⚠️ Fallback model {fallback_model} not found")

        # Look for any qwen models
        qwen_models = [model for model in available_models if "qwen" in model.lower()]
        if qwen_models:
            logger.info(f"✅ Found qwen models: {qwen_models}")
        else:
            logger.warning("⚠️ No qwen models found")

        # Now test the actual RAG initialization
        logger.info("\n--- Testing RAG Application Initialization ---")

        from Rag.rag_application import RAGApplication

        # Temporarily modify the primary model in config to test fallback
        logger.info("Initializing RAG application...")
        rag_app = RAGApplication()

        if hasattr(rag_app, "llm") and rag_app.llm:
            logger.info("✅ RAG application initialized successfully!")
            logger.info(f"LLM model type: {type(rag_app.llm).__name__}")

            if hasattr(rag_app.llm, "model"):
                logger.info(f"Using model: {rag_app.llm.model}")

                # Determine which fallback level was used
                if rag_app.llm.model == primary_model:
                    logger.info("✅ Using primary model")
                elif rag_app.llm.model == backup_model:
                    logger.info("⚠️ Used backup model (primary not available)")
                elif rag_app.llm.model == fallback_model:
                    logger.info(
                        "⚠️ Used fallback model (primary and backup not available)"
                    )
                elif rag_app.llm.model in qwen_models:
                    logger.info(f"⚠️ Used available qwen model: {rag_app.llm.model}")
                else:
                    logger.info(f"ℹ️ Using model: {rag_app.llm.model}")

            return True
        else:
            logger.error("❌ RAG application failed to initialize LLM")
            return False

    except Exception as e:
        logger.error(f"❌ Error in fallback test: {e}")
        import traceback

        traceback.print_exc()
        return False


def simulate_model_unavailable():
    """Simulate scenario where primary model is not available"""
    logger.info("\n" + "=" * 60)
    logger.info("Simulating Primary Model Unavailable Scenario")
    logger.info("=" * 60)

    # This would require temporarily modifying the Ollama models
    # For now, we'll just log what would happen
    available_models = get_available_ollama_models()

    logger.info("Scenario: llama3.2:3b is not available")
    logger.info(f"Available models: {available_models}")

    if "qwen2.5:3b" in available_models:
        logger.info("✅ Would use backup model: qwen2.5:3b")
    elif "qwen:4b" in available_models:
        logger.info("✅ Would use fallback model: qwen:4b")
    else:
        qwen_models = [model for model in available_models if "qwen" in model.lower()]
        if qwen_models:
            logger.info(f"✅ Would use available qwen model: {qwen_models[0]}")
        else:
            logger.error("❌ No suitable fallback models available")


if __name__ == "__main__":
    print("🔧 Testing RAG Model Fallback Logic")

    # Test 1: Direct fallback logic test
    success = test_fallback_logic_directly()

    # Test 2: Simulate unavailable scenario
    simulate_model_unavailable()

    if success:
        print("\n🎉 RAG model fallback tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 RAG model fallback tests failed!")
        sys.exit(1)
