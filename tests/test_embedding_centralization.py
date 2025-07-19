#!/usr/bin/env python3
"""
Test script for centralized embedding system
"""

import sys
import os
import warnings

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_new_centralized_system():
    """Test the new centralized embedding system"""
    print("Testing new centralized embedding system...")

    try:
        from embeddings import (
            EmbeddingManager,
            ResumeVectorizer,
            get_default_resume_vectorizer,
        )

        # Test 1: Direct usage
        print("1. Testing direct manager creation...")
        manager = EmbeddingManager()
        vectorizer = ResumeVectorizer(manager)

        test_text = "Python developer with machine learning experience"
        embedding = vectorizer.generate_embedding(test_text)
        print(f"   Generated embedding length: {len(embedding)}")

        # Test 2: Singleton usage
        print("2. Testing singleton pattern...")
        singleton_vectorizer = get_default_resume_vectorizer()
        embedding2 = singleton_vectorizer.generate_embedding(test_text)
        print(f"   Generated embedding length: {len(embedding2)}")

        # Test 3: Resume data processing
        print("3. Testing resume data processing...")
        sample_resume = {
            "name": "John Doe",
            "skills": ["Python", "Machine Learning", "FastAPI"],
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "duration": "2 years",
                }
            ],
            "education": [
                {
                    "degree": "Computer Science",
                    "institution": "University XYZ",
                    "dates": "2020-2024",
                }
            ],
        }

        resume_with_vectors = vectorizer.generate_resume_embeddings(sample_resume)
        print(f"   Resume processed with {len(resume_with_vectors)} fields")
        print(
            f"   Vector fields: {[k for k in resume_with_vectors.keys() if k.endswith('_vector')]}"
        )

        print("✅ New centralized system works correctly!")
        return True

    except Exception as e:
        print(f"❌ Error in new system: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with old imports"""
    print("\nTesting backward compatibility...")

    try:
        # Suppress deprecation warnings for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            from embeddings.vectorizer import Vectorizer, AddUserDataVectorizer

            # Test old Vectorizer class
            print("1. Testing old Vectorizer class...")
            vectorizer = Vectorizer()

            test_text = "Python developer with machine learning experience"
            embedding = vectorizer.generate_embedding(test_text)
            print(f"   Generated embedding length: {len(embedding)}")

            # Test old AddUserDataVectorizer class
            print("2. Testing old AddUserDataVectorizer class...")
            user_vectorizer = AddUserDataVectorizer()
            embedding2 = user_vectorizer.generate_embedding(test_text)
            print(f"   Generated embedding length: {len(embedding2)}")

            print("✅ Backward compatibility works correctly!")
            return True

    except Exception as e:
        print(f"❌ Error in backward compatibility: {e}")
        return False


def test_provider_flexibility():
    """Test different embedding providers"""
    print("\nTesting provider flexibility...")

    try:
        from embeddings import EmbeddingManager
        from embeddings.providers import EmbeddingProviderFactory

        # Test SentenceTransformer provider
        print("1. Testing SentenceTransformer provider...")
        st_provider = EmbeddingProviderFactory.create_provider(
            "sentence_transformer", model_name="all-MiniLM-L6-v2"
        )

        manager = EmbeddingManager(st_provider)
        print(f"   Provider: {manager.get_provider_info()}")
        print(f"   Embedding dimension: {manager.get_embedding_dimension()}")

        embedding = manager.generate_embedding("Test text")
        print(f"   Generated embedding length: {len(embedding)}")

        print("✅ Provider flexibility works correctly!")
        return True

    except Exception as e:
        print(f"❌ Error in provider testing: {e}")
        return False


def test_singleton_efficiency():
    """Test that singleton instances are reused"""
    print("\nTesting singleton efficiency...")

    try:
        from embeddings import (
            get_default_embedding_manager,
            get_default_resume_vectorizer,
        )

        # Get multiple instances
        manager1 = get_default_embedding_manager()
        manager2 = get_default_embedding_manager()

        vectorizer1 = get_default_resume_vectorizer()
        vectorizer2 = get_default_resume_vectorizer()

        # Check if they're the same instance
        print(f"1. Manager instances are same: {manager1 is manager2}")
        print(f"2. Vectorizer instances are same: {vectorizer1 is vectorizer2}")

        if manager1 is manager2 and vectorizer1 is vectorizer2:
            print("✅ Singleton pattern works correctly!")
            return True
        else:
            print("❌ Singleton pattern not working properly")
            return False

    except Exception as e:
        print(f"❌ Error in singleton testing: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING CENTRALIZED EMBEDDING SYSTEM")
    print("=" * 60)

    tests = [
        test_new_centralized_system,
        test_backward_compatibility,
        test_provider_flexibility,
        test_singleton_efficiency,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("🎉 All tests passed! Centralized embedding system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
