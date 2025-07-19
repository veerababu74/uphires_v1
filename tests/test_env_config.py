#!/usr/bin/env python3
"""
Test .env Configuration (UPDATED)
=================================

This script tests the updated .env configuration to ensure
all settings are correctly loaded and work with the new
auto-download system and best models.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_updated_env_configuration():
    """Test the updated .env configuration"""
    print("🧪 TESTING UPDATED .ENV CONFIGURATION")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Test critical configurations
    configs_to_test = {
        "Vector Search (Updated)": {
            "MODEL_NAME": os.getenv("MODEL_NAME"),
            "DIMENSIONS": os.getenv("DIMENSIONS"),
            "VECTOR_FIELD": os.getenv("VECTOR_FIELD"),
            "INDEX_NAME": os.getenv("INDEX_NAME"),
        },
        "Embedding Configuration (Best Models)": {
            "SENTENCE_TRANSFORMER_MODEL": os.getenv("SENTENCE_TRANSFORMER_MODEL"),
            "EMBEDDING_DIMENSIONS": os.getenv("EMBEDDING_DIMENSIONS"),
            "EMBEDDING_DEPLOYMENT": os.getenv("EMBEDDING_DEPLOYMENT"),
            "MODEL_CACHE_DIR": os.getenv("MODEL_CACHE_DIR"),
        },
        "Auto-Download Settings (New)": {
            "AUTO_DOWNLOAD_MODELS": os.getenv("AUTO_DOWNLOAD_MODELS"),
            "MODEL_DOWNLOAD_TIMEOUT": os.getenv("MODEL_DOWNLOAD_TIMEOUT"),
            "PRODUCTION_MODELS": os.getenv("PRODUCTION_MODELS"),
            "VERIFY_MODELS_ON_STARTUP": os.getenv("VERIFY_MODELS_ON_STARTUP"),
        },
        "Performance Settings (Optimized)": {
            "VECTOR_SEARCH_LIMIT": os.getenv("VECTOR_SEARCH_LIMIT"),
            "EMBEDDING_BATCH_SIZE": os.getenv("EMBEDDING_BATCH_SIZE"),
            "ENABLE_MODEL_CACHING": os.getenv("ENABLE_MODEL_CACHING"),
        },
        "Database Configuration": {
            "ATLAS_SEARCH_INDEX": os.getenv("ATLAS_SEARCH_INDEX"),
            "DB_NAME": os.getenv("DB_NAME"),
            "COLLECTION_NAME": os.getenv("COLLECTION_NAME"),
        },
    }

    all_good = True

    for section, configs in configs_to_test.items():
        print(f"\n📋 {section}:")
        for key, value in configs.items():
            if value:
                print(f"   ✅ {key}: {value}")
            else:
                print(f"   ❌ {key}: NOT SET")
                if key in ["MODEL_NAME", "DIMENSIONS", "SENTENCE_TRANSFORMER_MODEL"]:
                    all_good = False

    return all_good


def validate_configuration():
    """Validate specific requirements"""
    print(f"\n🔍 VALIDATION CHECKS:")

    all_valid = True

    # Check dimensions consistency
    dimensions = os.getenv("DIMENSIONS")
    embedding_dims = os.getenv("EMBEDDING_DIMENSIONS")
    if dimensions == embedding_dims == "1024":
        print("   ✅ Dimensions consistent (1024)")
    else:
        print(
            f"   ❌ Dimension mismatch: DIMENSIONS={dimensions}, EMBEDDING_DIMENSIONS={embedding_dims}"
        )
        all_valid = False

    # Check model names consistency
    vector_model = os.getenv("MODEL_NAME")
    embedding_model = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    if vector_model == embedding_model:
        print(f"   ✅ Model names consistent: {vector_model}")
    else:
        print(
            f"   ❌ Model mismatch: MODEL_NAME={vector_model}, SENTENCE_TRANSFORMER_MODEL={embedding_model}"
        )
        all_valid = False

    # Check index names updated for 1024
    index_name = os.getenv("INDEX_NAME")
    atlas_index = os.getenv("ATLAS_SEARCH_INDEX")
    if "1024" in (index_name or "") and "1024" in (atlas_index or ""):
        print("   ✅ Index names updated for 1024 dimensions")
    else:
        print(
            f"   ⚠️ Consider updating index names: INDEX_NAME={index_name}, ATLAS_SEARCH_INDEX={atlas_index}"
        )

    # Check production models
    prod_models = os.getenv("PRODUCTION_MODELS", "")
    expected_models = [
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-large-zh-v1.5",
        "thenlper/gte-large",
    ]
    if all(model in prod_models for model in expected_models):
        print("   ✅ Production models configured correctly")
    else:
        print(f"   ⚠️ Production models: {prod_models}")

    return all_valid


def test_model_loading():
    """Test model loading with new configuration"""
    print(f"\n🧪 TESTING MODEL LOADING:")

    try:
        from embeddings.providers import SentenceTransformerProvider

        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL")

        if not model_name:
            print("   ❌ No model name configured")
            return False

        provider = SentenceTransformerProvider(model_name)
        print(f"   ✅ Model provider created: {model_name}")
        print(f"   ✅ Expected dimensions: {provider.embedding_dim}")

        # Test if model is cached
        is_cached = provider._is_model_cached()
        cache_status = "✅ Cached" if is_cached else "⬇️ Will download on first use"
        print(f"   {cache_status}")

        return True

    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False


def show_deployment_recommendations():
    """Show deployment recommendations"""
    print("\n" + "=" * 60)
    print("🚀 DEPLOYMENT RECOMMENDATIONS")
    print("=" * 60)

    current_model = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    current_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

    print(f"📋 Current Configuration:")
    print(f"   Model: {current_model}")
    print(f"   Deployment: {current_deployment}")
    print(f"   Dimensions: {os.getenv('EMBEDDING_DIMENSIONS')}")

    print(f"\n🎯 Alternative Configurations:")

    recommendations = {
        "🥇 Best Overall (Current)": {
            "model": "BAAI/bge-large-en-v1.5",
            "deployment": "balanced",
            "description": "Top accuracy + fast inference",
            "download_size": "~3.3GB",
        },
        "⚡ Fastest Inference": {
            "model": "thenlper/gte-large",
            "deployment": "minimal",
            "description": "Very fast + smaller download",
            "download_size": "~2.0GB",
        },
        "🇨🇳 Chinese Optimized": {
            "model": "BAAI/bge-large-zh-v1.5",
            "deployment": "balanced",
            "description": "Best for Chinese text",
            "download_size": "~3.3GB",
        },
    }

    for name, config in recommendations.items():
        is_current = config["model"] == current_model
        marker = "👈 CURRENT" if is_current else ""
        print(f"\n{name} {marker}")
        print(f"   Model: {config['model']}")
        print(f"   Deployment: {config['deployment']}")
        print(f"   Use case: {config['description']}")
        print(f"   Download: {config['download_size']}")


def main():
    """Main test function"""
    print("🧪 TESTING UPDATED .ENV CONFIGURATION")
    print("=" * 60)

    # Run tests
    config_ok = test_updated_env_configuration()
    validation_ok = validate_configuration()
    model_ok = test_model_loading()

    # Show recommendations
    show_deployment_recommendations()

    # Final result
    all_ok = config_ok and validation_ok and model_ok

    print(
        f"\n{'🎉' if all_ok else '⚠️'} OVERALL RESULT: {'PASSED' if all_ok else 'NEEDS ATTENTION'}"
    )

    if all_ok:
        print("✅ Your .env configuration is ready for production deployment!")
        print("✅ Auto-download system will handle model management")
        print("✅ Using best performing models with 1024 dimensions")
    else:
        print("❌ Please fix the issues above before deployment")

    return all_ok


load_dotenv()


def test_configuration():
    """Test the multi-LLM configuration"""
    print("🔧 Testing Multi-LLM Configuration")
    print("=" * 50)

    # Test 1: Basic configuration loading
    print("1. Testing configuration loading...")
    try:
        from core.llm_config import LLMConfigManager, LLMProvider

        config = LLMConfigManager()
        print(f"   ✅ Current provider: {config.provider.value}")
        print(f"   ✅ Hugging Face model: {os.getenv('HUGGINGFACE_MODEL_ID')}")
        print(f"   ✅ Fallback providers: {os.getenv('LLM_FALLBACK_PROVIDERS')}")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

    # Test 2: LLM Factory
    print("\n2. Testing LLM Factory...")
    try:
        from core.llm_factory import LLMFactory

        # Note: We don't actually create the LLM here to avoid downloading models
        print("   ✅ LLM Factory imported successfully")
    except Exception as e:
        print(f"   ❌ LLM Factory error: {e}")
        return False

    # Test 3: Main resume parser
    print("\n3. Testing main resume parser...")
    try:
        from GroqcloudLLM.main import ResumeParser

        print("   ✅ Main resume parser imported successfully")
    except Exception as e:
        print(f"   ❌ Main resume parser error: {e}")
        return False

    # Test 4: Multiple resume parser
    print("\n4. Testing multiple resume parser...")
    try:
        from multipleresumepraser.main import ResumeParser as MultiResumeParser

        print("   ✅ Multiple resume parser imported successfully")
    except Exception as e:
        print(f"   ❌ Multiple resume parser error: {e}")
        return False

    # Test 5: RAG Application
    print("\n5. Testing RAG application...")
    try:
        from Rag.rag_application import RAGApplication

        print("   ✅ RAG application imported successfully")
    except Exception as e:
        print(f"   ❌ RAG application error: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 All configuration tests passed!")
    print("\n📋 Configuration Summary:")
    print(f"   • Primary Provider: {config.provider.value}")
    print(f"   • Hugging Face Model: {os.getenv('HUGGINGFACE_MODEL_ID')}")
    print(f"   • Fallback Providers: {os.getenv('LLM_FALLBACK_PROVIDERS')}")
    print(f"   • Device: {os.getenv('HUGGINGFACE_DEVICE')}")

    print("\n🚀 Your system is ready to use!")
    print("\n💡 Next steps:")
    print("   1. Install Hugging Face dependencies: pip install transformers torch")
    print("   2. Test with a resume: python GroqcloudLLM/main.py")
    print("   3. Try different providers by changing LLM_PROVIDER in .env")

    return True


if __name__ == "__main__":
    # Run the updated configuration tests first
    print("🔧 RUNNING UPDATED CONFIGURATION TESTS")
    print("=" * 60)
    main()  # This runs the new comprehensive tests

    print("\n\n🔧 RUNNING LEGACY CONFIGURATION TESTS")
    print("=" * 60)
    test_configuration()  # This runs the existing tests
