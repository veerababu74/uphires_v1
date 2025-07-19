#!/usr/bin/env python3
"""
Test Auto Model Download System
===============================

This script tests the automatic model downloading functionality
that will be used during FastAPI application startup.
"""

import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.auto_model_downloader import (
    AutoModelDownloader,
    ensure_embedding_models_on_startup,
)
from core.production_models import get_deployment_models, print_deployment_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_auto_download():
    """Test the automatic model download functionality"""
    logger.info("🧪 TESTING AUTO MODEL DOWNLOAD SYSTEM")
    logger.info("=" * 60)

    # Test 1: Show deployment configurations
    logger.info("\n📋 Available deployment configurations:")
    for config_name in ["minimal", "balanced", "full"]:
        models = get_deployment_models(config_name)
        logger.info(f"   {config_name}: {len(models)} models - {models}")

    # Test 2: Test with minimal configuration (small download)
    logger.info("\n🧪 Testing with 'minimal' configuration...")
    test_models = get_deployment_models("minimal")
    logger.info(f"Test models: {test_models}")

    # Check current model availability
    downloader = AutoModelDownloader()

    logger.info("\n📊 Current model status:")
    for model in test_models:
        is_available = downloader.is_model_available(model)
        status = "✅ Cached" if is_available else "⬇️ Needs download"
        logger.info(f"   {model}: {status}")

    # Test the auto-download function
    logger.info("\n🚀 Testing auto-download function...")

    try:
        results = await ensure_embedding_models_on_startup(
            required_models=test_models, timeout_seconds=300  # 5 minutes for testing
        )

        logger.info("\n📊 Final Results:")
        for model, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            logger.info(f"   {model}: {status}")

        successful = sum(1 for success in results.values() if success)
        total = len(results)

        logger.info(f"\n🎯 Summary: {successful}/{total} models ready")

        if successful == total:
            logger.info("🎉 All models are ready for production!")
            return True
        else:
            logger.warning("⚠️ Some models failed - check logs above")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False


async def test_model_performance():
    """Test embedding generation performance with cached models"""
    logger.info("\n🏃 TESTING MODEL PERFORMANCE")
    logger.info("=" * 60)

    from embeddings.providers import SentenceTransformerProvider
    import time

    # Test text
    test_text = "This is a test sentence for embedding generation performance."

    # Test models
    test_models = ["BAAI/bge-large-en-v1.5", "BAAI/bge-large-zh-v1.5"]

    for model_name in test_models:
        try:
            logger.info(f"\n📊 Testing model: {model_name}")

            # Create provider
            provider = SentenceTransformerProvider(model_name)

            # Check if cached
            is_cached = provider._is_model_cached()
            logger.info(f"   Cached: {'✅ Yes' if is_cached else '❌ No'}")

            if not is_cached:
                logger.warning(f"   Model not cached - skipping performance test")
                continue

            # Test loading time
            start_time = time.time()
            model = provider.model
            load_time = time.time() - start_time

            # Test embedding generation
            start_time = time.time()
            embedding = provider.generate_embedding(test_text)
            embed_time = time.time() - start_time

            # Results
            logger.info(f"   Load time: {load_time:.3f}s")
            logger.info(f"   Embedding time: {embed_time:.3f}s")
            logger.info(f"   Dimensions: {len(embedding)}")
            logger.info(f"   First 3 values: {embedding[:3]}")

        except Exception as e:
            logger.error(f"   ❌ Error testing {model_name}: {e}")


async def main():
    """Main test function"""
    logger.info("🧪 STARTING AUTO MODEL DOWNLOAD TESTS")
    logger.info("=" * 60)

    # Show deployment options
    print_deployment_summary("balanced")

    # Test auto-download
    download_success = await test_auto_download()

    if download_success:
        # Test performance with cached models
        await test_model_performance()

    logger.info("\n🏁 Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
