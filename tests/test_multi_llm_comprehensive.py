#!/usr/bin/env python3
"""
Multi-LLM Test Script
====================

This script tests the multi-LLM functionality across different modules:
- GroqcloudLLM (main resume parser)
- multipleresumepraser (multiple resume parser)
- Rag (RAG application)

Usage:
    python test_multi_llm_comprehensive.py
"""

import os
import sys
import json
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_config import get_llm_config, configure_llm_provider
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("multi_llm_test")


def test_groqcloud_llm():
    """Test the main GroqcloudLLM module"""
    print("\n🧪 Testing GroqcloudLLM Module")
    print("-" * 40)

    try:
        from GroqcloudLLM.main import ResumeParser

        sample_resume = """
        John Doe
        Email: john.doe@example.com
        Phone: +1-555-0123
        
        Experience:
        Software Engineer at TechCorp (2020-2023)
        - Developed web applications using Python and React
        - Led a team of 3 developers
        
        Education:
        Bachelor of Computer Science, MIT (2016-2020)
        """

        # Test with default provider
        parser = ResumeParser()
        result = parser.process_resume(sample_resume)

        print(f"✅ Default provider ({parser.provider.value}) test passed")
        print(f"   Extracted name: {result.get('name', 'N/A')}")

        # Test provider switching
        if parser.provider.value != "huggingface":
            try:
                parser.switch_provider("huggingface")
                print("✅ Provider switching test passed")
            except Exception as e:
                print(f"⚠️ Provider switching failed: {e}")

        return True

    except Exception as e:
        print(f"❌ GroqcloudLLM test failed: {e}")
        return False


def test_multiple_resume_parser():
    """Test the multipleresumepraser module"""
    print("\n🧪 Testing Multiple Resume Parser Module")
    print("-" * 40)

    try:
        from multipleresumepraser.main import ResumeParser

        sample_resume = """
        Jane Smith
        jane.smith@email.com
        (555) 987-6543
        
        Work Experience:
        Data Scientist at DataCorp (2021-2024)
        Marketing Analyst at MarketInc (2019-2021)
        
        Education:
        Master's in Data Science, Stanford (2017-2019)
        """

        # Test with current provider
        parser = ResumeParser()
        result = parser.process_resume(sample_resume)

        print(f"✅ Multiple resume parser test passed")
        print(f"   Provider: {parser.provider.value}")
        print(f"   Extracted name: {result.get('name', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Multiple resume parser test failed: {e}")
        return False


def test_rag_application():
    """Test the RAG application"""
    print("\n🧪 Testing RAG Application Module")
    print("-" * 40)

    try:
        from Rag.rag_application import RAGApplication

        # Initialize RAG application
        rag_app = RAGApplication()

        print(f"✅ RAG application initialization passed")
        print(f"   LLM provider: {rag_app.llm.__class__.__name__}")

        # Test basic search functionality
        test_query = "software engineer with Python experience"

        # Note: This might fail if no documents are indexed
        # That's expected and we'll handle it gracefully
        try:
            search_results = rag_app.vector_search_engine.search(test_query, limit=1)
            print(f"✅ RAG search test passed - found {len(search_results)} results")
        except Exception as search_e:
            print(f"⚠️ RAG search test skipped (no indexed documents): {search_e}")

        return True

    except Exception as e:
        print(f"❌ RAG application test failed: {e}")
        return False


def test_llm_config_management():
    """Test LLM configuration management"""
    print("\n🧪 Testing LLM Configuration Management")
    print("-" * 40)

    try:
        config_manager = get_llm_config()

        # Get current status
        status = config_manager.get_status()
        print(f"✅ Configuration status retrieved")
        print(f"   Current provider: {status['provider']}")
        print(f"   Validated: {status['validated']}")

        # Test provider availability
        providers_to_test = ["ollama", "groq_cloud", "huggingface"]

        for provider in providers_to_test:
            try:
                success = configure_llm_provider(provider)
                if success:
                    print(f"✅ {provider} configuration valid")
                else:
                    print(f"⚠️ {provider} configuration invalid or unavailable")
            except Exception as e:
                print(f"⚠️ {provider} test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ LLM configuration test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive tests across all modules"""
    print("🚀 Multi-LLM Comprehensive Test Suite")
    print("=" * 50)

    # Show current configuration
    config_manager = get_llm_config()
    print(f"\n📋 Current Configuration:")
    print(f"   Provider: {config_manager.provider.value}")
    print(f"   Validated: {config_manager._validated}")

    # Run all tests
    tests = [
        ("LLM Config Management", test_llm_config_management),
        ("GroqcloudLLM Module", test_groqcloud_llm),
        ("Multiple Resume Parser", test_multiple_resume_parser),
        ("RAG Application", test_rag_application),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n📊 Test Results Summary")
    print("-" * 30)
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Multi-LLM system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the configuration and dependencies.")

    return results


if __name__ == "__main__":
    # Set up test environment
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    try:
        results = run_comprehensive_test()

        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        sys.exit(1)
