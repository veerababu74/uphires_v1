"""
Production Model Configuration
=============================

Configure which embedding models should be automatically downloaded
during application startup based on your deployment needs.
"""

# RECOMMENDED MODELS FOR PRODUCTION
# Fast + High Accuracy + 1024 Dimensions

PRODUCTION_MODELS = {
    # TIER 1: Critical Models (Always Download)
    "critical": [
        "BAAI/bge-large-en-v1.5",  # 🥇 Best overall - Top MTEB performance
        "BAAI/bge-large-zh-v1.5",  # 🇨🇳 Chinese optimized - Your original target
    ],
    # TIER 2: High Performance Models (Recommended)
    "recommended": [
        "thenlper/gte-large",  # 🥈 Excellent speed - Very fast inference
        "intfloat/e5-large-v2",  # ⚡ Fast & reliable - Stable performance
    ],
    # TIER 3: Specialized Models (Optional)
    "optional": [
        "BAAI/bge-m3",  # 🌍 Multilingual - Multi-functionality
        "sentence-transformers/all-mpnet-base-v2",  # 🔒 Reliable fallback
    ],
}

# MODEL PERFORMANCE CHARACTERISTICS
MODEL_SPECS = {
    "BAAI/bge-large-en-v1.5": {
        "speed": "fast",
        "accuracy": "highest",
        "size": "1.34GB",
        "dimensions": 1024,
        "best_for": "English text, general purpose",
        "mteb_score": 63.98,  # Top performer
    },
    "thenlper/gte-large": {
        "speed": "very_fast",
        "accuracy": "high",
        "size": "670MB",
        "dimensions": 1024,
        "best_for": "Fast inference, multilingual",
        "mteb_score": 63.13,
    },
    "BAAI/bge-large-zh-v1.5": {
        "speed": "fast",
        "accuracy": "highest",
        "size": "1.34GB",
        "dimensions": 1024,
        "best_for": "Chinese text, your target model",
        "mteb_score": "62.96 (Chinese optimized)",
    },
    "intfloat/e5-large-v2": {
        "speed": "fast",
        "accuracy": "high",
        "size": "1.34GB",
        "dimensions": 1024,
        "best_for": "Stable performance, reliable",
        "mteb_score": 62.25,
    },
}

# DEPLOYMENT CONFIGURATIONS
DEPLOYMENT_CONFIGS = {
    # Minimal setup - just the essentials
    "minimal": PRODUCTION_MODELS["critical"],
    # Balanced setup - good performance with reasonable download size
    "balanced": PRODUCTION_MODELS["critical"] + PRODUCTION_MODELS["recommended"][:1],
    # Full setup - all recommended models
    "full": PRODUCTION_MODELS["critical"] + PRODUCTION_MODELS["recommended"],
    # Complete setup - everything (large download)
    "complete": (
        PRODUCTION_MODELS["critical"]
        + PRODUCTION_MODELS["recommended"]
        + PRODUCTION_MODELS["optional"]
    ),
}

# DEFAULT CONFIGURATION FOR AUTO-DOWNLOAD
DEFAULT_DEPLOYMENT = "balanced"  # Change this based on your needs


def get_deployment_models(deployment_type: str = None) -> list:
    """Get the list of models for a specific deployment configuration"""
    config_type = deployment_type or DEFAULT_DEPLOYMENT
    return DEPLOYMENT_CONFIGS.get(config_type, DEPLOYMENT_CONFIGS["balanced"])


def get_model_info(model_name: str) -> dict:
    """Get performance information for a specific model"""
    return MODEL_SPECS.get(model_name, {})


def print_deployment_summary(deployment_type: str = None):
    """Print a summary of models in a deployment configuration"""
    config_type = deployment_type or DEFAULT_DEPLOYMENT
    models = get_deployment_models(config_type)

    print(f"\n🚀 DEPLOYMENT CONFIG: {config_type.upper()}")
    print("=" * 50)
    print(f"Models to download: {len(models)}")

    total_size = 0
    for model in models:
        info = get_model_info(model)
        size_str = info.get("size", "Unknown")
        speed = info.get("speed", "Unknown")
        accuracy = info.get("accuracy", "Unknown")
        best_for = info.get("best_for", "General purpose")

        print(f"\n📦 {model}")
        print(f"   Size: {size_str} | Speed: {speed} | Accuracy: {accuracy}")
        print(f"   Best for: {best_for}")

        # Calculate total size (rough estimate)
        if "GB" in size_str:
            total_size += float(size_str.replace("GB", ""))
        elif "MB" in size_str:
            total_size += float(size_str.replace("MB", "")) / 1024

    print(f"\n📊 Estimated total download: ~{total_size:.2f} GB")
    print("=" * 50)


if __name__ == "__main__":
    # Show all deployment configurations
    for config_name in DEPLOYMENT_CONFIGS.keys():
        print_deployment_summary(config_name)
