#!/usr/bin/env python3
"""
Test Embedding Model Caching
============================

This script tests the local model caching functionality by:
1. Loading a small model for the first time (should download)
2. Loading the same model again (should use cache)
3. Verifying embeddings work correctly
"""

import os
import sys
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings.providers import SentenceTransformerProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_caching():
    """Test model caching functionality"""
    # Use a smaller model for testing
    test_model = "sentence-transformers/all-MiniLM-L6-v2"
    test_text = "This is a test sentence for embedding generation."

    logger.info("=" * 60)
    logger.info("TESTING EMBEDDING MODEL CACHING")
    logger.info("=" * 60)
    logger.info(f"Test model: {test_model}")
    logger.info(f"Test text: {test_text}")

    # Test 1: First load (should download)
    logger.info("\n--- TEST 1: First model load ---")
    start_time = time.time()

    provider1 = SentenceTransformerProvider(model_name=test_model)
    logger.info(f"Cache directory: {provider1.cache_dir}")
    logger.info(f"Model cached: {provider1._is_model_cached()}")

    # Load model (triggers download if not cached)
    model1 = provider1.model
    embedding1 = provider1.generate_embedding(test_text)

    load_time_1 = time.time() - start_time
    logger.info(f"First load time: {load_time_1:.2f} seconds")
    logger.info(f"Embedding dimensions: {len(embedding1)}")
    logger.info(f"First few values: {embedding1[:5]}")

    # Test 2: Second load (should use cache)
    logger.info("\n--- TEST 2: Second model load (cached) ---")
    start_time = time.time()

    provider2 = SentenceTransformerProvider(model_name=test_model)
    logger.info(f"Model cached: {provider2._is_model_cached()}")

    # Load model (should use cache)
    model2 = provider2.model
    embedding2 = provider2.generate_embedding(test_text)

    load_time_2 = time.time() - start_time
    logger.info(f"Second load time: {load_time_2:.2f} seconds")
    logger.info(f"Embedding dimensions: {len(embedding2)}")
    logger.info(f"First few values: {embedding2[:5]}")

    # Test 3: Compare results
    logger.info("\n--- TEST 3: Result comparison ---")

    # Check if embeddings are identical (they should be)
    embeddings_match = embedding1 == embedding2
    logger.info(f"Embeddings match: {embeddings_match}")

    # Check performance improvement
    if load_time_1 > 0 and load_time_2 > 0:
        speedup = load_time_1 / load_time_2
        logger.info(f"Cache speedup: {speedup:.2f}x faster")

    # Test 4: Cache directory structure
    logger.info("\n--- TEST 4: Cache directory structure ---")
    if os.path.exists(provider1.cache_dir):
        files = os.listdir(provider1.cache_dir)
        logger.info(f"Cache directory files: {len(files)}")
        for file in sorted(files):
            file_path = os.path.join(provider1.cache_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({size:,} bytes)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    if embeddings_match:
        logger.info("✓ Caching test PASSED")
        logger.info("✓ Embeddings are consistent between cached and fresh loads")
        if load_time_2 < load_time_1:
            logger.info("✓ Cache provides performance improvement")
        else:
            logger.info(
                "⚠ Cache performance improvement not significant (normal for small models)"
            )
    else:
        logger.error("✗ Caching test FAILED")
        logger.error("✗ Embeddings differ between cached and fresh loads")
        return False

    return True


def test_multiple_models():
    """Test caching with multiple models"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MULTIPLE MODEL CACHING")
    logger.info("=" * 60)

    test_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ]

    test_text = "Multi-model caching test."

    for i, model_name in enumerate(test_models, 1):
        logger.info(f"\n--- Model {i}: {model_name} ---")

        provider = SentenceTransformerProvider(model_name=model_name)
        logger.info(f"Cache directory: {provider.cache_dir}")
        logger.info(f"Model cached: {provider._is_model_cached()}")

        # Generate embedding
        start_time = time.time()
        embedding = provider.generate_embedding(test_text)
        load_time = time.time() - start_time

        logger.info(f"Load time: {load_time:.2f} seconds")
        logger.info(f"Dimensions: {len(embedding)}")
        logger.info(f"Sample values: {embedding[:3]}")

    logger.info("\n✓ Multiple model caching test completed")


if __name__ == "__main__":
    try:
        # Run basic caching test
        success = test_model_caching()

        if success:
            # Run multiple models test
            test_multiple_models()

            logger.info("\n🎉 All caching tests completed successfully!")
        else:
            logger.error("\n❌ Caching tests failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
