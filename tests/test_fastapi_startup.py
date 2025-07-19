#!/usr/bin/env python3
"""
Test FastAPI Startup with Auto Model Download
============================================

This script simulates the FastAPI startup process to test
the automatic model downloading functionality.
"""

import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging like FastAPI
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("startup_test")


async def simulate_fastapi_startup():
    """Simulate the FastAPI lifespan startup process"""

    # Import the same modules as main.py
    from core.auto_model_downloader import ensure_embedding_models_on_startup
    from core.production_models import get_deployment_models

    logger.info("🚀 Simulating FastAPI application startup...")

    try:
        # Step 1: Ensure embedding models are available (same as main.py)
        logger.info("🚀 Starting application initialization...")

        # Get production models based on deployment configuration
        deployment_type = os.getenv("EMBEDDING_DEPLOYMENT", "balanced")
        production_models = get_deployment_models(deployment_type)

        logger.info(f"📦 Using '{deployment_type}' deployment configuration")
        logger.info(f"📥 Models to ensure: {production_models}")

        model_results = await ensure_embedding_models_on_startup(
            required_models=production_models,
            timeout_seconds=600,  # 10 minutes max for downloads
        )

        # Check if critical models are available
        critical_models = ["BAAI/bge-large-en-v1.5", "BAAI/bge-large-zh-v1.5"]
        missing_critical = [
            model for model in critical_models if not model_results.get(model, False)
        ]

        if missing_critical:
            logger.warning(
                f"⚠️ Some critical models failed to download: {missing_critical}"
            )
            logger.warning("Application will continue but some features may be limited")
        else:
            logger.info("✅ All critical embedding models are ready!")

        # Step 2: Simulate other startup tasks
        logger.info("📊 Simulating other application startup tasks...")
        await asyncio.sleep(1)  # Simulate some work

        logger.info("🎉 Application startup completed successfully!")

        # Test embedding generation
        logger.info("🧪 Testing embedding generation...")
        from embeddings.providers import SentenceTransformerProvider

        provider = SentenceTransformerProvider("BAAI/bge-large-en-v1.5")
        embedding = provider.generate_embedding("FastAPI startup test")

        logger.info(f"✅ Embedding test successful - {len(embedding)} dimensions")

        return True

    except Exception as e:
        logger.error(f"❌ Application startup failed: {str(e)}")
        return False


async def main():
    """Main test function"""
    logger.info("🧪 TESTING FASTAPI STARTUP SIMULATION")
    logger.info("=" * 60)

    # Show current environment
    deployment = os.getenv("EMBEDDING_DEPLOYMENT", "balanced")
    logger.info(f"Environment: EMBEDDING_DEPLOYMENT={deployment}")

    # Run startup simulation
    success = await simulate_fastapi_startup()

    if success:
        logger.info("\n🎉 Startup simulation successful!")
        logger.info(
            "✅ Your FastAPI app is ready for deployment with auto model downloading!"
        )
    else:
        logger.error("\n❌ Startup simulation failed!")
        logger.error("❌ Check the errors above before deploying")

    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
