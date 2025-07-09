#!/usr/bin/env python3
"""
Ollama Connection and RAG Diagnostic Script

This script helps diagnose issues with Ollama connectivity and RAG functionality.
Run this to identify and fix common problems.
"""

import sys
import os
import requests
import time
import traceback
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))


def test_ollama_connection():
    """Test basic Ollama connection"""
    print("=" * 50)
    print("TESTING OLLAMA CONNECTION")
    print("=" * 50)

    try:
        # Test Ollama API endpoint
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running and accessible")
            print(
                f"Available models: {[model['name'] for model in models.get('models', [])]}"
            )

            # Check if qwen:4b is available
            model_names = [model["name"] for model in models.get("models", [])]
            if "qwen:4b" in model_names:
                print("✅ qwen:4b model is available")
                return True
            else:
                print("❌ qwen:4b model is NOT available")
                print("Available models:", model_names)
                print("\nTo install qwen:4b, run: ollama pull qwen:4b")
                return False
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("Make sure Ollama is running. Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False


def test_ollama_llm():
    """Test Ollama LLM directly"""
    print("\n" + "=" * 50)
    print("TESTING OLLAMA LLM")
    print("=" * 50)

    try:
        from langchain_ollama import OllamaLLM

        # Initialize Ollama LLM
        llm = OllamaLLM(
            model="qwen:4b",
            temperature=0.7,
            base_url="http://localhost:11434",
            timeout=60,
            request_timeout=60,
        )

        print("✅ Ollama LLM initialized successfully")

        # Test simple query
        print("Testing simple query...")
        start_time = time.time()
        response = llm.invoke("Hello! Please respond with just 'Hi there!'")
        end_time = time.time()

        print(f"✅ LLM Response: {response}")
        print(f"✅ Response time: {end_time - start_time:.2f} seconds")
        return True

    except Exception as e:
        print(f"❌ Error testing Ollama LLM: {e}")
        traceback.print_exc()
        return False


def test_rag_initialization():
    """Test RAG application initialization"""
    print("\n" + "=" * 50)
    print("TESTING RAG INITIALIZATION")
    print("=" * 50)

    try:
        from Rag.runner import initialize_rag_app

        print("Initializing RAG application...")
        rag_app = initialize_rag_app()
        print("✅ RAG application initialized successfully")

        # Test simple vector search
        print("Testing vector similarity search...")
        result = rag_app.vector_similarity_search("python developer", limit=5)
        print(
            f"✅ Vector search completed. Found {result.get('total_found', 0)} results"
        )

        return True

    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        traceback.print_exc()
        return False


def test_llm_context_search():
    """Test LLM context search"""
    print("\n" + "=" * 50)
    print("TESTING LLM CONTEXT SEARCH")
    print("=" * 50)

    try:
        from Rag.runner import initialize_rag_app

        print("Initializing RAG application for LLM search...")
        rag_app = initialize_rag_app()

        print("Testing LLM context search...")
        result = rag_app.llm_context_search(
            "python developer with machine learning", context_size=3
        )

        if "error" in result:
            print(f"❌ LLM context search failed: {result['error']}")
            return False
        else:
            print(
                f"✅ LLM context search completed. Found {result.get('total_analyzed', 0)} analyzed results"
            )
            return True

    except Exception as e:
        print(f"❌ Error in LLM context search: {e}")
        traceback.print_exc()
        return False


def test_resume_parser():
    """Test resume parser"""
    print("\n" + "=" * 50)
    print("TESTING RESUME PARSER")
    print("=" * 50)

    try:
        from GroqcloudLLM.main import ResumeParser

        print("Initializing Resume Parser...")
        parser = ResumeParser()
        print("✅ Resume Parser initialized successfully")

        # Test with sample resume text
        sample_resume = """
        John Doe
        Email: john.doe@example.com
        Phone: +1234567890
        
        Experience:
        Software Developer at ABC Company (2020-2023)
        - Developed Python applications
        - Worked with machine learning models
        
        Skills: Python, Machine Learning, FastAPI
        Education: BS Computer Science
        """

        print("Testing resume parsing...")
        result = parser.process_resume(sample_resume)

        if "error" in result:
            print(f"❌ Resume parsing failed: {result['error']}")
            return False
        else:
            print("✅ Resume parsing successful")
            print(
                f"Extracted name: {result.get('contact_details', {}).get('name', 'N/A')}"
            )
            return True

    except Exception as e:
        print(f"❌ Error testing resume parser: {e}")
        traceback.print_exc()
        return False


def provide_recommendations(test_results):
    """Provide recommendations based on test results"""
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)

    if not test_results["ollama_connection"]:
        print("🔧 PRIORITY 1: Fix Ollama Connection")
        print("   1. Start Ollama: ollama serve")
        print("   2. Install qwen:4b model: ollama pull qwen:4b")
        print("   3. Verify with: curl http://localhost:11434/api/tags")

    elif not test_results["ollama_llm"]:
        print("🔧 PRIORITY 1: Fix Ollama LLM Issues")
        print("   1. Check if qwen:4b model is fully downloaded")
        print("   2. Restart Ollama service")
        print("   3. Check system resources (RAM/CPU)")

    elif not test_results["rag_init"]:
        print("🔧 PRIORITY 1: Fix RAG Initialization")
        print("   1. Check MongoDB connection")
        print("   2. Verify embeddings model")
        print("   3. Check database permissions")

    elif not test_results["llm_search"]:
        print("🔧 PRIORITY 1: Fix LLM Context Search")
        print("   1. Check LLM response parsing")
        print("   2. Verify prompt templates")
        print("   3. Check JSON output parsing")

    elif not test_results["resume_parser"]:
        print("🔧 PRIORITY 1: Fix Resume Parser")
        print("   1. Check resume parser initialization")
        print("   2. Verify LLM model access")
        print("   3. Check output parsing")
    else:
        print("✅ All tests passed! Your system should be working correctly.")
        print("If you're still experiencing issues, check the application logs.")


def main():
    """Run all diagnostic tests"""
    print("🔍 OLLAMA & RAG DIAGNOSTIC TOOL")
    print(
        "This tool will help identify and fix issues with Ollama and RAG functionality.\n"
    )

    test_results = {
        "ollama_connection": test_ollama_connection(),
        "ollama_llm": False,
        "rag_init": False,
        "llm_search": False,
        "resume_parser": False,
    }

    # Only run subsequent tests if Ollama connection works
    if test_results["ollama_connection"]:
        test_results["ollama_llm"] = test_ollama_llm()

        if test_results["ollama_llm"]:
            test_results["rag_init"] = test_rag_initialization()

            if test_results["rag_init"]:
                test_results["llm_search"] = test_llm_context_search()
                test_results["resume_parser"] = test_resume_parser()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    provide_recommendations(test_results)


if __name__ == "__main__":
    main()
