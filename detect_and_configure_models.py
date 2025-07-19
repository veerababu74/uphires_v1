#!/usr/bin/env python3
"""
Script to detect embedding dimensions of various models and update configuration

This script will:
1. Test multiple embedding models to detect their dimensions
2. Update the embedding configuration to support them
3. Provide options for future model additions
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
import json


def detect_model_dimensions(
    model_name: str, trust_remote_code: bool = False
) -> Tuple[int, bool]:
    """
    Detect the embedding dimensions of a given model

    Returns:
        Tuple of (dimensions, success)
    """
    try:
        print(f"🔍 Testing model: {model_name}")

        # Load model
        model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

        # Test with sample text
        test_text = "This is a test sentence to detect embedding dimensions."
        embedding = model.encode(test_text)

        dimensions = (
            len(embedding) if hasattr(embedding, "__len__") else embedding.shape[0]
        )

        print(f"  ✅ Dimensions: {dimensions}")
        return dimensions, True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return 0, False


def test_multiple_models() -> Dict[str, Dict]:
    """Test multiple embedding models and return their configurations"""

    models_to_test = [
        # Current models
        ("all-MiniLM-L6-v2", False),
        ("BAAI/bge-large-zh-v1.5", False),
        # New models from user request
        ("nomic-ai/nomic-embed-text-v1", True),  # Requires trust_remote_code=True
        ("intfloat/e5-small-v2", False),
        # Additional common models
        ("intfloat/e5-large-v2", False),
        ("intfloat/e5-base-v2", False),
        ("sentence-transformers/all-mpnet-base-v2", False),
        ("sentence-transformers/all-roberta-large-v1", False),
    ]

    configurations = {}

    print("🧪 Testing embedding models for dimension detection...")
    print("=" * 60)

    for model_name, trust_remote_code in models_to_test:
        dimensions, success = detect_model_dimensions(model_name, trust_remote_code)

        if success:
            # Create configuration name
            config_name = model_name.replace("/", "-").replace("_", "-").lower()

            configurations[config_name] = {
                "provider": "sentence_transformer",
                "model_name": model_name,
                "embedding_dimension": dimensions,
                "device": "cpu",
                "trust_remote_code": trust_remote_code,
            }

    return configurations


def update_embedding_config_file(new_configs: Dict[str, Dict]) -> bool:
    """Update the embedding config file with new model configurations"""
    try:
        config_file_path = os.path.join(
            os.path.dirname(__file__), "embeddings", "config.py"
        )

        # Read current config file
        with open(config_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the EMBEDDING_CONFIGS dictionary
        start_marker = "EMBEDDING_CONFIGS = {"
        end_marker = "}"

        start_idx = content.find(start_marker)
        if start_idx == -1:
            print("❌ Could not find EMBEDDING_CONFIGS in config file")
            return False

        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx + len(start_marker)

        for i, char in enumerate(content[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        # Extract existing configs
        existing_config_text = content[start_idx:end_idx]

        # Build new config dictionary text
        new_config_lines = ["EMBEDDING_CONFIGS = {"]

        # Add existing configs (parse them manually for safety)
        existing_configs = {
            "all-MiniLM-L6-v2": {
                "provider": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "device": "cpu",
            },
            "qwen-embedding-0.6b": {
                "provider": "sentence_transformer",
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_dimension": 1024,
                "device": "cpu",
            },
            "baai-bge-large-zh": {
                "provider": "sentence_transformer",
                "model_name": "BAAI/bge-large-zh-v1.5",
                "embedding_dimension": 1024,
                "device": "cpu",
            },
            "openai-small": {
                "provider": "openai",
                "model_name": "text-embedding-3-small",
                "embedding_dimension": 1536,
            },
        }

        # Merge existing and new configs
        all_configs = {**existing_configs, **new_configs}

        # Generate config text
        for config_name, config in all_configs.items():
            new_config_lines.append(f'    "{config_name}": {{')
            new_config_lines.append(f'        "provider": "{config["provider"]}",')
            new_config_lines.append(f'        "model_name": "{config["model_name"]}",')
            new_config_lines.append(
                f'        "embedding_dimension": {config["embedding_dimension"]},'
            )
            new_config_lines.append(f'        "device": "{config["device"]}",')

            # Add trust_remote_code if needed
            if config.get("trust_remote_code", False):
                new_config_lines.append(
                    f'        "trust_remote_code": {config["trust_remote_code"]},'
                )

            new_config_lines.append("    },")

        new_config_lines.append("}")

        # Replace in content
        new_content = (
            content[:start_idx] + "\n".join(new_config_lines) + content[end_idx:]
        )

        # Write updated content
        with open(config_file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"✅ Updated embedding config file: {config_file_path}")
        return True

    except Exception as e:
        print(f"❌ Error updating config file: {str(e)}")
        return False


def update_provider_dimensions(new_configs: Dict[str, Dict]) -> bool:
    """Update the SentenceTransformerProvider to recognize new model dimensions"""
    try:
        provider_file_path = os.path.join(
            os.path.dirname(__file__), "embeddings", "providers.py"
        )

        # Read current provider file
        with open(provider_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the __init__ method where dimensions are set
        init_method_start = content.find(
            'def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):'
        )
        if init_method_start == -1:
            print("❌ Could not find __init__ method in provider file")
            return False

        # Find the dimension setting section
        dimension_section_start = content.find(
            "# Update embedding dimension based on model", init_method_start
        )
        if dimension_section_start == -1:
            print("❌ Could not find dimension setting section")
            return False

        # Find the end of the dimension setting section
        dimension_section_end = content.find(
            'elif "text-embedding-004" in model_name:', dimension_section_start
        )
        dimension_section_end = content.find(
            "self.embedding_dim = 768", dimension_section_end
        ) + len("self.embedding_dim = 768")

        # Build new dimension setting code
        new_dimension_lines = ["        # Update embedding dimension based on model"]

        # Add existing conditions
        existing_conditions = [
            ('if "Qwen3-Embedding-0.6B" in model_name:', 1024),
            ('elif "bge-large-zh-v1.5" in model_name:', 1024),
            ('elif "text-embedding-3-small" in model_name:', 1536),
            ('elif "jina-embeddings-v3" in model_name:', 1024),
            ('elif "text-embedding-004" in model_name:', 768),
        ]

        # Add new conditions based on detected models
        for config_name, config in new_configs.items():
            model_name = config["model_name"]
            dimensions = config["embedding_dimension"]

            # Create a unique identifier for the model
            if "nomic-embed-text-v1" in model_name:
                existing_conditions.append(
                    (f'elif "nomic-embed-text-v1" in model_name:', dimensions)
                )
            elif "e5-small-v2" in model_name:
                existing_conditions.append(
                    (f'elif "e5-small-v2" in model_name:', dimensions)
                )
            elif "e5-large-v2" in model_name:
                existing_conditions.append(
                    (f'elif "e5-large-v2" in model_name:', dimensions)
                )
            elif "e5-base-v2" in model_name:
                existing_conditions.append(
                    (f'elif "e5-base-v2" in model_name:', dimensions)
                )
            elif "all-mpnet-base-v2" in model_name:
                existing_conditions.append(
                    (f'elif "all-mpnet-base-v2" in model_name:', dimensions)
                )
            elif "all-roberta-large-v1" in model_name:
                existing_conditions.append(
                    (f'elif "all-roberta-large-v1" in model_name:', dimensions)
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_conditions = []
        for condition, dim in existing_conditions:
            if condition not in seen:
                seen.add(condition)
                unique_conditions.append((condition, dim))

        # Generate the new dimension setting code
        for i, (condition, dimensions) in enumerate(unique_conditions):
            new_dimension_lines.append(f"        {condition}")
            new_dimension_lines.append(f"            self.embedding_dim = {dimensions}")

        # Replace in content
        new_content = (
            content[:dimension_section_start]
            + "\n".join(new_dimension_lines)
            + content[dimension_section_end:]
        )

        # Write updated content
        with open(provider_file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"✅ Updated provider file: {provider_file_path}")
        return True

    except Exception as e:
        print(f"❌ Error updating provider file: {str(e)}")
        return False


def generate_usage_examples(configs: Dict[str, Dict]) -> None:
    """Generate usage examples for the new models"""
    print("\n" + "=" * 60)
    print("📋 USAGE EXAMPLES")
    print("=" * 60)

    print("\n🔧 Environment Variable Examples:")
    print("Add these to your .env file to use different models:\n")

    for config_name, config in configs.items():
        print(f"# {config['model_name']} ({config['embedding_dimension']} dimensions)")
        print(f"EMBEDDING_PROVIDER=sentence_transformer")
        print(f"SENTENCE_TRANSFORMER_MODEL={config['model_name']}")
        print(f"EMBEDDING_DIMENSIONS={config['embedding_dimension']}")
        print(f"EMBEDDING_DEVICE=cpu")
        if config.get("trust_remote_code", False):
            print(f"TRUST_REMOTE_CODE=true")
        print()

    print("\n🐍 Python Code Examples:")
    print("Use these code snippets to test the models:\n")

    for config_name, config in configs.items():
        model_name = config["model_name"]
        dimensions = config["embedding_dimension"]
        trust_remote_code = config.get("trust_remote_code", False)

        print(f"# {model_name}")
        print("from embeddings.config import get_config_by_name")
        print("from embeddings.providers import EmbeddingProviderFactory")
        print("from embeddings import EmbeddingManager, ResumeVectorizer")
        print()
        print(f"config = get_config_by_name('{config_name}')")
        print("provider = EmbeddingProviderFactory.create_provider(")
        print("    provider_type=config.provider,")
        print("    model_name=config.model_name,")
        print("    device=config.device")
        if trust_remote_code:
            print("    # Note: This model requires trust_remote_code=True")
        print(")")
        print("manager = EmbeddingManager(provider)")
        print("vectorizer = ResumeVectorizer(manager)")
        print(f"# This will generate {dimensions}-dimensional embeddings")
        print()


def main():
    """Main function to detect and configure new embedding models"""
    print("🚀 Multi-Model Embedding Configuration Tool")
    print("=" * 60)

    # Test models and detect dimensions
    new_configs = test_multiple_models()

    if not new_configs:
        print("❌ No new models were successfully tested")
        return False

    print(f"\n✅ Successfully detected {len(new_configs)} model configurations:")
    for config_name, config in new_configs.items():
        trust_info = (
            " (trust_remote_code=True)" if config.get("trust_remote_code") else ""
        )
        print(
            f"  - {config['model_name']}: {config['embedding_dimension']} dimensions{trust_info}"
        )

    # Update configuration files
    print("\n🔧 Updating configuration files...")

    config_updated = update_embedding_config_file(new_configs)
    provider_updated = update_provider_dimensions(new_configs)

    if config_updated and provider_updated:
        print("\n✅ All configuration files updated successfully!")

        # Generate usage examples
        generate_usage_examples(new_configs)

        print("\n📝 Next Steps:")
        print("1. Choose which model you want to use by updating your .env file")
        print(
            "2. Run the migration script if switching from 384 to different dimensions"
        )
        print("3. Test the new model configuration")
        print("4. Update your MongoDB search index if dimensions changed")

        return True
    else:
        print("\n❌ Some configuration updates failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Multi-model configuration completed successfully!")
    else:
        print("\n💥 Configuration failed. Please check the errors above.")
