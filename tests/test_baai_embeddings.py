#!/usr/bin/env python3
"""
Test script to verify BAAI/bge-large-zh-v1.5 embedding generation

This script tests the BAAI model functionality before running the full migration.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings.config import EmbeddingConfig
from embeddings.providers import EmbeddingProviderFactory
from embeddings.manager import EmbeddingManager, ResumeVectorizer
import numpy as np


def test_baai_embeddings():
    """Test BAAI/bge-large-zh-v1.5 model"""
    print("🧪 Testing BAAI/bge-large-zh-v1.5 embedding generation...")

    try:
        # Create configuration for BAAI model
        config = EmbeddingConfig(
            provider="sentence_transformer",
            model_name="BAAI/bge-large-zh-v1.5",
            embedding_dimension=1024,
            device="cpu",  # Change to "cuda" if you have GPU
        )

        print(f"📋 Configuration: {config.to_dict()}")

        # Create provider
        provider = EmbeddingProviderFactory.create_provider(
            provider_type=config.provider,
            model_name=config.model_name,
            device=config.device,
        )

        print(f"🔧 Provider created: {provider.get_provider_name()}")
        print(f"📏 Embedding dimension: {provider.get_embedding_dimension()}")

        # Test basic embedding generation
        test_texts = [
            "样例数据-1",
            "样例数据-2",
            "Python developer with 5 years experience",
            "Machine learning engineer",
            "Data scientist with expertise in AI",
        ]

        print("\n🔍 Testing embedding generation...")
        embeddings = []

        for i, text in enumerate(test_texts):
            embedding = provider.generate_embedding(text)
            embeddings.append(embedding)
            print(f"Text {i+1}: '{text}' -> Embedding shape: {len(embedding)}")

            # Verify dimension
            if len(embedding) != 1024:
                print(f"❌ ERROR: Expected 1024 dimensions, got {len(embedding)}")
                return False

        # Test similarity calculation (like in your original code)
        print("\n🔢 Testing similarity calculation...")
        embeddings_1 = np.array(embeddings[:2])  # First 2 embeddings
        embeddings_2 = np.array(embeddings[2:4])  # Next 2 embeddings

        # Normalize embeddings
        embeddings_1_norm = embeddings_1 / np.linalg.norm(
            embeddings_1, axis=1, keepdims=True
        )
        embeddings_2_norm = embeddings_2 / np.linalg.norm(
            embeddings_2, axis=1, keepdims=True
        )

        # Calculate similarity
        similarity = embeddings_1_norm @ embeddings_2_norm.T
        print(f"Similarity matrix shape: {similarity.shape}")
        print(f"Similarity matrix:\n{similarity}")

        # Test with ResumeVectorizer
        print("\n🏢 Testing with ResumeVectorizer...")
        embedding_manager = EmbeddingManager(provider)
        vectorizer = ResumeVectorizer(embedding_manager)

        # Test resume data
        test_resume = {
            "name": "John Doe",
            "contact_details": {"email": "john@example.com"},
            "skills": ["Python", "Machine Learning", "Data Science"],
            "experience": [
                {
                    "title": "Data Scientist",
                    "company": "Tech Corp",
                    "duration": "2020-2023",
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

        resume_embeddings = vectorizer.generate_resume_embeddings(test_resume)

        print("✅ Resume embedding fields generated:")
        vector_fields = [
            field for field in resume_embeddings.keys() if "vector" in field
        ]
        for field in vector_fields:
            if field in resume_embeddings:
                print(f"  - {field}: {len(resume_embeddings[field])} dimensions")

        print("\n✅ All tests passed! BAAI/bge-large-zh-v1.5 is working correctly.")
        print("📝 Summary:")
        print(f"  - Model: BAAI/bge-large-zh-v1.5")
        print(f"  - Embedding dimension: 1024")
        print(f"  - Provider: {provider.get_provider_name()}")
        print(f"  - Device: {config.device}")

        return True

    except Exception as e:
        print(f"❌ Error testing BAAI embeddings: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def compare_with_current_model():
    """Compare BAAI model with current model"""
    print("\n🔍 Comparing BAAI model with current model...")

    try:
        # Current model (384 dimensions)
        current_provider = EmbeddingProviderFactory.create_provider(
            provider_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        # New model (1024 dimensions)
        new_provider = EmbeddingProviderFactory.create_provider(
            provider_type="sentence_transformer",
            model_name="BAAI/bge-large-zh-v1.5",
            device="cpu",
        )

        test_text = "Python developer with machine learning experience"

        # Generate embeddings
        current_embedding = current_provider.generate_embedding(test_text)
        new_embedding = new_provider.generate_embedding(test_text)

        print("📊 Comparison:")
        print(
            f"  Current model (all-MiniLM-L6-v2): {len(current_embedding)} dimensions"
        )
        print(f"  New model (BAAI/bge-large-zh-v1.5): {len(new_embedding)} dimensions")
        print(f"  Dimension increase: {len(new_embedding) - len(current_embedding)}")
        print(f"  Size ratio: {len(new_embedding) / len(current_embedding):.2f}x")

        return True

    except Exception as e:
        print(f"❌ Error in comparison: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 BAAI/bge-large-zh-v1.5 Embedding Test")
    print("=" * 50)

    # Test BAAI embeddings
    if test_baai_embeddings():
        # Compare models
        compare_with_current_model()

        print("\n" + "=" * 50)
        print("✅ Testing completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the migration script: python update_embeddings_to_1024.py")
        print("2. The script will backup your data and update embeddings")
        print("3. Update your .env file if needed")
        print("4. Restart your application")
    else:
        print("\n❌ Testing failed. Please check the error messages above.")
