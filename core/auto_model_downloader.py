"""
Automatic Model Downloader for Production Deployment
===================================================

This module handles automatic downloading of required embedding models
during application startup. Models are only downloaded if not already cached.
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional
from embeddings.config import EMBEDDING_CONFIGS
from embeddings.providers import SentenceTransformerProvider

logger = logging.getLogger(__name__)


class AutoModelDownloader:
    """Handles automatic model downloading during app startup"""

    def __init__(self):
        self.required_models = [
            "BAAI/bge-large-en-v1.5",  # Best overall performance
            "thenlper/gte-large",  # Excellent speed
            "BAAI/bge-large-zh-v1.5",  # Chinese support
            "intfloat/e5-large-v2",  # Fast & reliable
        ]
        self.download_results = {}

    def set_required_models(self, models: List[str]):
        """Set which models should be auto-downloaded"""
        self.required_models = models
        logger.info(f"Set required models for auto-download: {models}")

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is already cached locally"""
        try:
            # Handle both old and new config formats
            if model_name not in EMBEDDING_CONFIGS:
                logger.warning(f"Model {model_name} not found in config")
                return False

            config = EMBEDDING_CONFIGS[model_name]

            # Get actual model name from config
            if "model_name" in config:
                actual_model_name = config["model_name"]
                trust_remote_code = config.get("trust_remote_code", False)
            else:
                actual_model_name = model_name
                trust_remote_code = config.get("trust_remote_code", False)

            # Create provider to check cache
            provider = SentenceTransformerProvider(
                model_name=actual_model_name, trust_remote_code=trust_remote_code
            )

            return provider._is_model_cached()

        except Exception as e:
            logger.error(f"Error checking model availability for {model_name}: {e}")
            return False

    async def download_model_async(self, model_name: str) -> bool:
        """Download a single model asynchronously"""
        try:
            logger.info(f"Starting async download for model: {model_name}")

            # Run the download in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._download_model_sync, model_name
            )

            return result

        except Exception as e:
            logger.error(f"Async download failed for {model_name}: {e}")
            return False

    def _download_model_sync(self, model_name: str) -> bool:
        """Synchronous model download (called in thread pool)"""
        try:
            if model_name not in EMBEDDING_CONFIGS:
                logger.error(f"Model {model_name} not found in config")
                return False

            config = EMBEDDING_CONFIGS[model_name]

            # Handle both old and new config formats
            if "model_name" in config:
                actual_model_name = config["model_name"]
                trust_remote_code = config.get("trust_remote_code", False)
            else:
                actual_model_name = model_name
                trust_remote_code = config.get("trust_remote_code", False)

            # Create provider (this will trigger download if not cached)
            provider = SentenceTransformerProvider(
                model_name=actual_model_name, trust_remote_code=trust_remote_code
            )

            # Access the model property to trigger loading/downloading
            model = provider.model

            logger.info(f"Model {model_name} downloaded successfully")
            logger.info(f"   - Actual model: {actual_model_name}")
            logger.info(f"   - Dimensions: {provider.embedding_dim}")
            logger.info(f"   - Cache location: {provider.cache_dir}")

            return True

        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False

    async def ensure_models_available(
        self, required_models: Optional[List[str]] = None, timeout_seconds: int = 300
    ) -> Dict[str, bool]:
        """
        Ensure all required models are available (download if needed)

        Args:
            required_models: List of models to check/download (uses default if None)
            timeout_seconds: Maximum time to wait for downloads

        Returns:
            Dict mapping model names to success status
        """
        models_to_check = required_models or self.required_models

        logger.info("=" * 60)
        logger.info("CHECKING EMBEDDING MODEL AVAILABILITY")
        logger.info("=" * 60)

        results = {}
        models_to_download = []

        # Check which models need downloading
        for model_name in models_to_check:
            if self.is_model_available(model_name):
                logger.info(f"{model_name} - Already cached")
                results[model_name] = True
            else:
                logger.info(f"{model_name} - Needs download")
                models_to_download.append(model_name)
                results[model_name] = False

        if not models_to_download:
            logger.info("All required models are already cached!")
            return results

        logger.info(f"\nDownloading {len(models_to_download)} models...")
        logger.info("This may take a few minutes on first startup...")

        # Download models with timeout
        try:
            download_tasks = [
                self.download_model_async(model_name)
                for model_name in models_to_download
            ]

            download_results = await asyncio.wait_for(
                asyncio.gather(*download_tasks, return_exceptions=True),
                timeout=timeout_seconds,
            )

            # Update results
            for i, model_name in enumerate(models_to_download):
                if isinstance(download_results[i], bool):
                    results[model_name] = download_results[i]
                else:
                    logger.error(
                        f"Download exception for {model_name}: {download_results[i]}"
                    )
                    results[model_name] = False

        except asyncio.TimeoutError:
            logger.error(f"Model downloads timed out after {timeout_seconds} seconds")
            for model_name in models_to_download:
                if model_name not in results or not results[model_name]:
                    results[model_name] = False

        # Summary
        successful = [model for model, success in results.items() if success]
        failed = [model for model, success in results.items() if not success]

        logger.info("\n" + "=" * 60)
        logger.info("MODEL DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")

        if successful:
            logger.info("\nAvailable models:")
            for model in successful:
                logger.info(f"   - {model}")

        if failed:
            logger.warning("\nFailed models:")
            for model in failed:
                logger.warning(f"   - {model}")

        self.download_results = results
        return results

    def get_download_summary(self) -> str:
        """Get a summary of the last download operation"""
        if not self.download_results:
            return "No download operations performed yet"

        successful = sum(1 for success in self.download_results.values() if success)
        total = len(self.download_results)

        return f"Models: {successful}/{total} successful"


# Global instance
auto_downloader = AutoModelDownloader()


async def ensure_embedding_models_on_startup(
    required_models: Optional[List[str]] = None, timeout_seconds: int = 300
) -> Dict[str, bool]:
    """
    Convenience function for FastAPI startup to ensure models are available

    Usage in main.py lifespan:
        from core.auto_model_downloader import ensure_embedding_models_on_startup

        results = await ensure_embedding_models_on_startup([
            "BAAI/bge-large-en-v1.5",
            "BAAI/bge-large-zh-v1.5"
        ])
    """
    return await auto_downloader.ensure_models_available(
        required_models, timeout_seconds
    )


def set_production_models(models: List[str]):
    """Set which models should be auto-downloaded in production"""
    auto_downloader.set_required_models(models)
