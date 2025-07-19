#!/usr/bin/env python3
"""
Embedding Models Download Script
===============================

This script downloads and caches all the best-performing embedding models
to the local 'emmodels' directory for deployment use.

Usage:
    python download_embedding_models.py [--models MODEL1,MODEL2] [--all]

Examples:
    python download_embedding_models.py --all
    python download_embedding_models.py --models "BAAI/bge-large-en-v1.5,thenlper/gte-large"
"""

import os
import sys
import argparse
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings.config import EMBEDDING_CONFIGS
from embeddings.providers import SentenceTransformerProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_recommended_models() -> List[str]:
    """Get list of recommended high-performance models"""
    recommended = [
        "BAAI/bge-large-en-v1.5",  # Best English general purpose
        "thenlper/gte-large",  # Excellent performance
        "BAAI/bge-m3",  # Multilingual
        "intfloat/e5-large-v2",  # Strong performance
        "sentence-transformers/all-mpnet-base-v2",  # Reliable baseline
        "BAAI/bge-large-zh-v1.5",  # Original target model
        "mixedbread-ai/mxbai-embed-large-v1",  # High performance
    ]

    # Filter to only include models that exist in config
    available_models = set(EMBEDDING_CONFIGS.keys())
    return [model for model in recommended if model in available_models]


def get_all_models() -> List[str]:
    """Get all available models from config"""
    return list(EMBEDDING_CONFIGS.keys())


def download_model(model_name: str) -> bool:
    """Download a single model and return success status"""
    try:
        logger.info(f"Starting download for model: {model_name}")

        # Get model config
        if model_name not in EMBEDDING_CONFIGS:
            logger.error(f"Model {model_name} not found in config")
            return False

        config = EMBEDDING_CONFIGS[model_name]

        # Handle both old and new config formats
        if "model_name" in config:
            # Old format: use model_name field
            actual_model_name = config["model_name"]
            trust_remote_code = config.get("trust_remote_code", False)
        else:
            # New format: key is the model name
            actual_model_name = model_name
            trust_remote_code = config.get("trust_remote_code", False)

        # Create provider (this will trigger download if not cached)
        provider = SentenceTransformerProvider(
            model_name=actual_model_name, trust_remote_code=trust_remote_code
        )

        # Access the model property to trigger loading/downloading
        model = provider.model

        logger.info(f"✓ Model {model_name} downloaded successfully")
        logger.info(f"  - Actual model: {actual_model_name}")
        logger.info(f"  - Dimensions: {provider.embedding_dim}")
        logger.info(f"  - Cache location: {provider.cache_dir}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to download model {model_name}: {e}")
        return False


def download_models(model_names: List[str]) -> Dict[str, bool]:
    """Download multiple models and return results"""
    results = {}
    total_models = len(model_names)

    logger.info(f"Starting download of {total_models} models...")
    logger.info("=" * 60)

    for i, model_name in enumerate(model_names, 1):
        logger.info(f"[{i}/{total_models}] Processing: {model_name}")

        # Check if already cached
        try:
            provider = SentenceTransformerProvider(model_name=model_name)
            if provider._is_model_cached():
                logger.info(f"✓ Model {model_name} already cached - skipping")
                results[model_name] = True
                continue
        except Exception:
            pass  # Continue with download attempt

        results[model_name] = download_model(model_name)
        logger.info("-" * 40)

    return results


def print_summary(results: Dict[str, bool]):
    """Print download summary"""
    successful = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]

    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models processed: {len(results)}")
    logger.info(f"Successful downloads: {len(successful)}")
    logger.info(f"Failed downloads: {len(failed)}")

    if successful:
        logger.info("\n✓ SUCCESSFUL DOWNLOADS:")
        for model in successful:
            logger.info(f"  - {model}")

    if failed:
        logger.info("\n✗ FAILED DOWNLOADS:")
        for model in failed:
            logger.info(f"  - {model}")

    # Show disk usage info
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        emmodels_dir = os.path.join(project_root, "emmodels")
        if os.path.exists(emmodels_dir):
            total_size = 0
            for root, dirs, files in os.walk(emmodels_dir):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))

            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024

            if size_gb > 1:
                logger.info(f"\n📁 Total cache size: {size_gb:.2f} GB")
            else:
                logger.info(f"\n📁 Total cache size: {size_mb:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not calculate cache size: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download embedding models for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    Download all available models
  %(prog)s --recommended            Download recommended models only (default)
  %(prog)s --models "model1,model2" Download specific models
  %(prog)s --list                   List available models
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Download all available models"
    )

    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Download recommended models only (default)",
    )

    parser.add_argument(
        "--models", type=str, help="Comma-separated list of specific models to download"
    )

    parser.add_argument(
        "--list", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list:
        print("\nAVAILABLE MODELS:")
        print("=" * 50)
        for model_name, config in EMBEDDING_CONFIGS.items():
            # Handle both old and new config formats
            if "model_name" in config:
                # Old format
                actual_model = config["model_name"]
                dims = config.get("embedding_dimension", "auto")
                desc = config.get("description", "No description")
            else:
                # New format
                actual_model = model_name
                dims = config.get("dimensions", "auto")
                desc = config.get("description", "No description")

            print(f"{model_name}")
            print(f"  Model: {actual_model}")
            print(f"  Dimensions: {dims}")
            print(f"  Description: {desc}")
            print()
        return

    # Determine which models to download
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
        logger.info(f"Selected specific models: {model_names}")
    elif args.all:
        model_names = get_all_models()
        logger.info("Selected all available models")
    else:
        # Default to recommended
        model_names = get_recommended_models()
        logger.info("Selected recommended models (use --all for all models)")

    if not model_names:
        logger.error("No models selected for download")
        return

    logger.info(f"Models to download: {len(model_names)}")
    for model in model_names:
        logger.info(f"  - {model}")

    print("\nStarting downloads...")
    results = download_models(model_names)
    print_summary(results)

    # Exit with error code if any downloads failed
    failed_count = sum(1 for success in results.values() if not success)
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
