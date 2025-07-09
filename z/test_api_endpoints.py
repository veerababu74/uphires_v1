#!/usr/bin/env python3
"""
Test script to verify Ollama endpoints are working correctly
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"


def test_ollama_health():
    """Test Ollama health endpoint"""
    print("=" * 50)
    print("TESTING OLLAMA HEALTH")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/api/ollama/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_ollama_models():
    """Test Ollama models endpoint"""
    print("\n" + "=" * 50)
    print("TESTING OLLAMA MODELS")
    print("=" * 50)

    try:
        response = requests.get(
            f"{BASE_URL}/api/ollama/models", timeout=30
        )  # Increased timeout
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Success: {result.get('success', False)}")
        print(f"Models: {result.get('models', [])}")
        return result.get("success", False)
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_ollama_ask():
    """Test Ollama ask endpoint"""
    print("\n" + "=" * 50)
    print("TESTING OLLAMA ASK")
    print("=" * 50)

    try:
        payload = {
            "question": "What is Python? Please respond in one sentence.",
            "model": "qwen:4b",
            "temperature": 0.7,
        }

        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/ollama/ask", json=payload, timeout=60)
        end_time = time.time()

        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Success: {result.get('success', False)}")
        print(f"Response Time: {result.get('response_time', 0):.2f}s")
        print(f"Answer: {result.get('answer', 'No answer')[:200]}...")

        return result.get("success", False)
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_rag_vector_search():
    """Test RAG vector search endpoint"""
    print("\n" + "=" * 50)
    print("TESTING RAG VECTOR SEARCH")
    print("=" * 50)

    try:
        payload = {
            "query": "python developer with machine learning experience",
            "limit": 5,
        }

        response = requests.post(
            f"{BASE_URL}/vector-similarity-search", json=payload, timeout=30
        )
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Total Found: {result.get('total_found', 0)}")
            print(f"Results Count: {len(result.get('results', []))}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_rag_llm_search():
    """Test RAG LLM context search endpoint"""
    print("\n" + "=" * 50)
    print("TESTING RAG LLM CONTEXT SEARCH")
    print("=" * 50)

    try:
        payload = {
            "query": "python developer with machine learning experience",
            "context_size": 3,
        }

        response = requests.post(
            f"{BASE_URL}/llm-context-search", json=payload, timeout=60
        )
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Total Analyzed: {result.get('total_analyzed', 0)}")
            print(f"Results Count: {len(result.get('results', []))}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_resume_parser():
    """Test resume parser endpoint"""
    print("\n" + "=" * 50)
    print("TESTING RESUME PARSER")
    print("=" * 50)

    try:
        # Create a test text file
        test_resume_content = """
John Doe
Software Developer
Email: john.doe@example.com
Phone: +1234567890

Experience:
- Software Developer at ABC Company (2020-2023)
- Developed Python applications using Django and FastAPI
- Worked with machine learning models using scikit-learn

Skills: Python, Django, FastAPI, Machine Learning, PostgreSQL
Education: BS Computer Science from XYZ University
        """.strip()

        # Save to a temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(test_resume_content)
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as f:
                files = {"file": ("test_resume.txt", f, "text/plain")}
                response = requests.post(
                    f"{BASE_URL}/resume_parser/resume-parser", files=files, timeout=120
                )  # Increased timeout

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Filename: {result.get('filename', 'N/A')}")
                resume_data = result.get("resume_parser", {})
                contact_details = resume_data.get("contact_details", {})
                print(f"Extracted Name: {contact_details.get('name', 'N/A')}")
                print(f"Extracted Email: {contact_details.get('email', 'N/A')}")
                return True
            else:
                print(f"Error: {response.text}")
                return False

        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 TESTING OLLAMA & RAG API ENDPOINTS")
    print("Testing endpoints after fixes...\n")

    test_results = {
        "Ollama Health": test_ollama_health(),
        "Ollama Models": test_ollama_models(),
        "Ollama Ask": test_ollama_ask(),
        "RAG Vector Search": test_rag_vector_search(),
        "RAG LLM Search": test_rag_llm_search(),
        "Resume Parser": test_resume_parser(),
    }

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ollama and RAG are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for more details.")

        # Provide specific recommendations
        if not test_results["Ollama Health"]:
            print("- Ollama service might not be running. Start with: ollama serve")
        if not test_results["Ollama Models"]:
            print("- qwen:4b model might not be installed. Run: ollama pull qwen:4b")
        if not test_results["RAG LLM Search"]:
            print("- LLM responses might need better prompt tuning")


if __name__ == "__main__":
    main()
