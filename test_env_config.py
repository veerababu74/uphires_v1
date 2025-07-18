#!/usr/bin/env python3
"""
Test script to verify the updated .env configuration works with all modules
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
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
    test_configuration()
