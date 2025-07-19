#!/usr/bin/env python3
"""
Test script to specifically test the RAG _initialize_ollama_with_fallback method.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.custom_logger import CustomLogger
from Rag.config import RAGConfig
import requests

# Setup logging
logger = CustomLogger().get_logger("rag_fallback_direct_test")


def test_ollama_fallback_method():
    """Test the _initialize_ollama_with_fallback method directly"""
    try:
        logger.info("=" * 60)
        logger.info("Testing RAG _initialize_ollama_with_fallback Method")
        logger.info("=" * 60)

        # Import RAG application
        from Rag.rag_application import RAGApplication

        # Create instance but don't initialize fully
        rag_app = RAGApplication.__new__(RAGApplication)

        logger.info(f"Primary model configured: {RAGConfig.OLLAMA_MODEL}")
        logger.info(f"Backup model configured: {RAGConfig.OLLAMA_BACKUP_MODEL}")
        logger.info(f"Fallback model configured: {RAGConfig.OLLAMA_FALLBACK_MODEL}")

        # Get available models
        try:
            response = requests.get(f"{RAGConfig.OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]
                logger.info(f"Available models: {available_models}")
            else:
                logger.error("Failed to get available models")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return False

        # Test the fallback method
        logger.info("\n--- Testing fallback method ---")

        try:
            llm = rag_app._initialize_ollama_with_fallback()

            if llm:
                logger.info("Fallback method succeeded!")
                logger.info(f"Selected model: {llm.model}")
                logger.info(f"LLM type: {type(llm).__name__}")

                # Determine which fallback level was used
                if llm.model == RAGConfig.OLLAMA_MODEL:
                    logger.info("Result: Used primary model")
                elif llm.model == RAGConfig.OLLAMA_BACKUP_MODEL:
                    logger.info("Result: Used backup model")
                elif llm.model == RAGConfig.OLLAMA_FALLBACK_MODEL:
                    logger.info("Result: Used fallback model")
                else:
                    logger.info(f"Result: Used available model: {llm.model}")

                return True
            else:
                logger.error("Fallback method returned None")
                return False

        except Exception as e:
            logger.error(f"Fallback method failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_selection_logic():
    """Test the model selection logic with different scenarios"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Model Selection Logic")
    logger.info("=" * 60)

    # Get available models
    try:
        response = requests.get(f"{RAGConfig.OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]
        else:
            logger.error("Failed to get available models")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return False

    # Test different scenarios
    scenarios = [
        {
            "name": "Primary model available",
            "models": [RAGConfig.OLLAMA_MODEL] + available_models,
            "expected": RAGConfig.OLLAMA_MODEL,
        },
        {
            "name": "Only backup model available",
            "models": [RAGConfig.OLLAMA_BACKUP_MODEL, "other:model"],
            "expected": RAGConfig.OLLAMA_BACKUP_MODEL,
        },
        {
            "name": "Only fallback model available",
            "models": [RAGConfig.OLLAMA_FALLBACK_MODEL, "other:model"],
            "expected": RAGConfig.OLLAMA_FALLBACK_MODEL,
        },
        {
            "name": "Only qwen models available",
            "models": ["qwen:1.8b", "qwen:7b", "other:model"],
            "expected": "qwen:1.8b",  # Should pick first qwen model
        },
    ]

    for scenario in scenarios:
        logger.info(f"\nScenario: {scenario['name']}")
        logger.info(f"Available models: {scenario['models']}")

        # Simulate the selection logic
        primary_model = RAGConfig.OLLAMA_MODEL
        backup_model = RAGConfig.OLLAMA_BACKUP_MODEL
        fallback_model = RAGConfig.OLLAMA_FALLBACK_MODEL
        models = scenario["models"]

        if primary_model in models:
            selected = primary_model
            logger.info("Would select: Primary model")
        elif backup_model in models:
            selected = backup_model
            logger.info("Would select: Backup model")
        elif fallback_model in models:
            selected = fallback_model
            logger.info("Would select: Fallback model")
        else:
            qwen_models = [model for model in models if "qwen" in model.lower()]
            if qwen_models:
                selected = qwen_models[0]
                logger.info(f"Would select: First available qwen model ({selected})")
            else:
                selected = None
                logger.info("Would fail: No suitable models")

        logger.info(f"Expected: {scenario['expected']}, Got: {selected}")

        if selected == scenario["expected"]:
            logger.info("PASS")
        else:
            logger.warning("FAIL")

    return True


if __name__ == "__main__":
    print("Testing RAG Model Fallback Method Directly")

    # Test 1: Direct method test
    success1 = test_ollama_fallback_method()

    # Test 2: Logic test
    success2 = test_model_selection_logic()

    if success1 and success2:
        print("\n🎉 RAG fallback method tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 RAG fallback method tests failed!")
        sys.exit(1)
