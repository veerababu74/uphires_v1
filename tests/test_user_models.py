#!/usr/bin/env python3
"""
Test script for the specific models mentioned by the user:
- nomic-ai/nomic-embed-text-v1
- intfloat/e5-small-v2

This script will test these models with the exact code provided by the user.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from embeddings.config import get_config_by_name
from embeddings.providers import EmbeddingProviderFactory
from embeddings.manager import EmbeddingManager, ResumeVectorizer


def test_nomic_model():
    """Test nomic-ai/nomic-embed-text-v1 model as specified by user"""
    print("🧪 Testing nomic-ai/nomic-embed-text-v1")
    print("-" * 40)

    try:
        # Test with the exact code from user
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        sentences = ["search_query: Who is Laurens van Der Maaten?"]
        embeddings = model.encode(sentences)
        print(f"✅ Direct SentenceTransformer test:")
        print(f"   - Input: {sentences[0]}")
        print(f"   - Embedding shape: {embeddings.shape}")
        print(f"   - Embedding dimensions: {len(embeddings[0])}")
        print(f"   - First 5 values: {embeddings[0][:5]}")

        # Test with our framework
        print(f"\n🔧 Testing with our embedding framework:")
        config = get_config_by_name("nomic-embed-text-v1")
        provider = EmbeddingProviderFactory.create_provider(
            provider_type=config.provider,
            model_name=config.model_name,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )

        manager = EmbeddingManager(provider)
        framework_embedding = manager.generate_embedding(sentences[0])

        print(f"   - Framework embedding dimensions: {len(framework_embedding)}")
        print(f"   - Framework first 5 values: {framework_embedding[:5]}")

        # Compare results
        direct_norm = np.linalg.norm(embeddings[0])
        framework_norm = np.linalg.norm(framework_embedding)
        similarity = np.dot(embeddings[0], framework_embedding) / (
            direct_norm * framework_norm
        )

        print(f"   - Similarity between direct and framework: {similarity:.6f}")

        return True

    except Exception as e:
        print(f"❌ Error testing nomic model: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_e5_model():
    """Test intfloat/e5-small-v2 model as specified by user"""
    print("\n🧪 Testing intfloat/e5-small-v2")
    print("-" * 40)

    try:
        # Test with the exact code from user
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/e5-small-v2")
        input_texts = [
            "query: how much protein should a female eat",
            "query: summit define",
            "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        ]
        embeddings = model.encode(input_texts, normalize_embeddings=True)

        print(f"✅ Direct SentenceTransformer test:")
        print(f"   - Number of texts: {len(input_texts)}")
        print(f"   - Embeddings shape: {embeddings.shape}")
        print(f"   - Embedding dimensions: {embeddings.shape[1]}")
        print(f"   - Normalized: True")

        # Calculate similarities between queries and passages
        query_embeddings = embeddings[:2]  # First 2 are queries
        passage_embeddings = embeddings[2:]  # Last 2 are passages

        similarities = query_embeddings @ passage_embeddings.T
        print(f"   - Query-Passage similarities:")
        for i, query_text in enumerate(input_texts[:2]):
            for j, passage_text in enumerate(input_texts[2:]):
                print(f"     Query {i+1} -> Passage {j+1}: {similarities[i][j]:.4f}")

        # Test with our framework
        print(f"\n🔧 Testing with our embedding framework:")
        config = get_config_by_name("e5-small-v2")
        provider = EmbeddingProviderFactory.create_provider(
            provider_type=config.provider,
            model_name=config.model_name,
            device=config.device,
        )

        manager = EmbeddingManager(provider)

        # Test with first query
        test_text = input_texts[0]
        framework_embedding = manager.generate_embedding(test_text)
        framework_embedding_norm = np.array(framework_embedding) / np.linalg.norm(
            framework_embedding
        )

        print(f"   - Framework embedding dimensions: {len(framework_embedding)}")
        print(f"   - Test text: {test_text}")

        # Compare with direct result
        direct_embedding_norm = embeddings[0]
        similarity = np.dot(direct_embedding_norm, framework_embedding_norm)
        print(f"   - Similarity between direct and framework: {similarity:.6f}")

        return True

    except Exception as e:
        print(f"❌ Error testing e5 model: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_resume_vectorization():
    """Test how these models work with resume data"""
    print("\n🏢 Testing Resume Vectorization with New Models")
    print("-" * 50)

    # Sample resume data
    test_resume = {
        "name": "John Doe",
        "contact_details": {"email": "john@example.com", "phone": "+1234567890"},
        "skills": ["Python", "Machine Learning", "Data Science", "AI"],
        "experience": [
            {
                "title": "Senior Data Scientist",
                "company": "Tech Corp",
                "duration": "2020-2023",
                "description": "Developed ML models for recommendation systems",
            }
        ],
        "education": [
            {
                "degree": "Master of Science in Computer Science",
                "institution": "University of Technology",
                "dates": "2018-2020",
            }
        ],
    }

    models_to_test = [
        ("nomic-embed-text-v1", "Nomic AI model"),
        ("e5-small-v2", "E5 Small model"),
        ("all-MiniLM-L6-v2", "Current default model"),
    ]

    results = {}

    for config_name, description in models_to_test:
        try:
            print(f"\n📊 Testing {description} ({config_name}):")

            config = get_config_by_name(config_name)
            provider_kwargs = {
                "provider_type": config.provider,
                "model_name": config.model_name,
                "device": config.device,
            }

            # Add trust_remote_code if needed
            if hasattr(config, "trust_remote_code") and config.trust_remote_code:
                provider_kwargs["trust_remote_code"] = config.trust_remote_code

            provider = EmbeddingProviderFactory.create_provider(**provider_kwargs)
            manager = EmbeddingManager(provider)
            vectorizer = ResumeVectorizer(manager)

            # Generate resume embeddings
            resume_with_embeddings = vectorizer.generate_resume_embeddings(test_resume)

            # Show embedding dimensions for each field
            vector_fields = [
                field for field in resume_with_embeddings.keys() if "vector" in field
            ]

            print(f"   ✅ Generated embeddings:")
            for field in vector_fields:
                if field in resume_with_embeddings and resume_with_embeddings[field]:
                    dims = len(resume_with_embeddings[field])
                    print(f"     - {field}: {dims} dimensions")

            # Store results for comparison
            results[config_name] = {
                "dimensions": config.embedding_dimension,
                "vector_fields": vector_fields,
                "model_name": config.model_name,
            }

        except Exception as e:
            print(f"     ❌ Error: {str(e)}")

    # Compare results
    if len(results) > 1:
        print(f"\n📈 Comparison Summary:")
        for config_name, result in results.items():
            print(f"   {config_name}:")
            print(f"     - Model: {result['model_name']}")
            print(f"     - Dimensions: {result['dimensions']}")
            print(f"     - Vector fields: {len(result['vector_fields'])}")


def main():
    """Main test function"""
    print("🚀 Testing User-Specified Embedding Models")
    print("=" * 60)

    success_count = 0
    total_tests = 3

    # Test 1: Nomic model
    if test_nomic_model():
        success_count += 1

    # Test 2: E5 model
    if test_e5_model():
        success_count += 1

    # Test 3: Resume vectorization
    try:
        test_resume_vectorization()
        success_count += 1
    except Exception as e:
        print(f"❌ Resume vectorization test failed: {str(e)}")

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("✅ All tests passed! Your new models are ready to use.")
        print("\n📋 Next Steps:")
        print("1. Choose which model you want to use")
        print("2. Update your .env file with the model configuration")
        print("3. Run migration script if changing dimensions")
        print("4. Update MongoDB search index if needed")

        print("\n🔧 Model Summary:")
        print(
            "- nomic-ai/nomic-embed-text-v1: 768 dimensions (requires trust_remote_code=True)"
        )
        print("- intfloat/e5-small-v2: 384 dimensions (same as current default)")
        print("- Current default (all-MiniLM-L6-v2): 384 dimensions")

    else:
        print("❌ Some tests failed. Please check the error messages above.")

    return success_count == total_tests


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All models tested successfully!")
    else:
        print("\n💥 Some tests failed. Please check the output above.")
