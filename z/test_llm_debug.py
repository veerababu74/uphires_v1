#!/usr/bin/env python3
"""
Debug LLM responses for RAG system
"""

import requests
import json
from Rag.runner import initialize_rag_app


def test_direct_ollama():
    """Test Ollama directly"""
    print("=== TESTING DIRECT OLLAMA ===")

    simple_prompt = """You are an AI assistant. Return ONLY a valid JSON response.
    
    Task: Return a simple JSON with matches array.
    
    Return this exact format:
    {
      "total_candidates": 1,
      "matches": [
        {
          "_id": "test123",
          "relevance_score": 0.8,
          "match_reason": "test match"
        }
      ]
    }
    
    JSON RESPONSE:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen:4b",
                "prompt": simple_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 500},
            },
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Raw response: {result.get('response', 'No response')}")
            return result.get("response", "")
        else:
            print(f"Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Direct Ollama test failed: {e}")
        return None


def test_rag_llm():
    """Test RAG LLM chain"""
    print("\n=== TESTING RAG LLM CHAIN ===")

    try:
        rag_app = initialize_rag_app()

        # Simple test with minimal context
        test_context = """
        {
          "_id": "683b0a9f1aa118b254288b80",
          "contact_details": {
            "name": "John Doe",
            "email": "john@example.com"
          },
          "skills": ["Python", "Machine Learning", "Django"],
          "total_experience": "5 years"
        }
        """

        test_query = "python developer"

        print(f"Testing with query: {test_query}")
        print(f"Context: {test_context[:200]}...")

        # Test the chain directly
        result = rag_app.chain_manager.ranking_chain.invoke(
            {"context": test_context, "question": test_query}
        )

        print(f"Chain result type: {type(result)}")
        print(f"Chain result: {result}")

        return result

    except Exception as e:
        print(f"RAG LLM test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_llm_search_engine():
    """Test LLM search engine"""
    print("\n=== TESTING LLM SEARCH ENGINE ===")

    try:
        rag_app = initialize_rag_app()
        result = rag_app.llm_context_search("python", context_size=2)

        print(f"Search result: {json.dumps(result, indent=2)}")
        return result

    except Exception as e:
        print(f"LLM search engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🔍 LLM DEBUG TOOL")
    print("=" * 50)

    # Test 1: Direct Ollama
    direct_result = test_direct_ollama()

    # Test 2: RAG LLM Chain
    chain_result = test_rag_llm()

    # Test 3: Full search engine
    search_result = test_llm_search_engine()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Direct Ollama: {'✅ OK' if direct_result else '❌ FAIL'}")
    print(f"RAG Chain: {'✅ OK' if chain_result else '❌ FAIL'}")
    print(f"Search Engine: {'✅ OK' if search_result else '❌ FAIL'}")
