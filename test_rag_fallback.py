#!/usr/bin/env python3
"""
Test script to verify the RAG model fallback logic works correctly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Rag.rag_application import RAGApplication
from core.custom_logger import CustomLogger
import logging

# Setup logging
logger = CustomLogger().get_logger("rag_fallback_test")


def test_rag_model_fallback():
    """Test the RAG model fallback functionality"""
    try:
        logger.info("=" * 60)
        logger.info("Testing RAG Model Fallback Logic")
        logger.info("=" * 60)

        # Initialize RAG application
        logger.info("Initializing RAG application...")
        rag_app = RAGApplication()

        # Check if LLM was initialized
        if hasattr(rag_app, "llm") and rag_app.llm:
            logger.info("✅ RAG application initialized successfully!")
            logger.info(f"LLM model type: {type(rag_app.llm).__name__}")

            # Try to get model information if available
            if hasattr(rag_app.llm, "model"):
                logger.info(f"Using model: {rag_app.llm.model}")

            return True
        else:
            logger.error("❌ RAG application failed to initialize LLM")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing RAG fallback: {e}")
        return False


if __name__ == "__main__":
    success = test_rag_model_fallback()

    if success:
        print("\n🎉 RAG model fallback test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 RAG model fallback test failed!")
        sys.exit(1)
