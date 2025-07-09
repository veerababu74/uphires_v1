#!/usr/bin/env python3
"""
Ollama Setup Helper Script
Helps diagnose and setup Ollama for resume parsing
"""

import requests
import subprocess
import time
import sys


def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_available_models():
    """List all available models in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception:
        return []


def pull_model(model_name):
    """Pull a model using Ollama CLI"""
    try:
        print(f"Pulling model {model_name}...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False


def main():
    print("🔍 Ollama Setup Helper")
    print("=" * 40)

    # Check if Ollama is running
    print("1. Checking if Ollama is running...")
    if check_ollama_running():
        print("✅ Ollama is running!")
    else:
        print("❌ Ollama is not running!")
        print("\n🔧 To start Ollama:")
        print("   Windows: Start 'Ollama' from Start Menu or run 'ollama serve'")
        print("   Mac/Linux: Run 'ollama serve' in terminal")
        return

    # List available models
    print("\n2. Checking available models...")
    models = list_available_models()
    if models:
        print("✅ Available models:")
        for model in models:
            print(f"   - {model}")
    else:
        print("⚠️ No models found!")

    # Check for recommended models
    print("\n3. Checking for recommended models...")
    recommended_models = ["llama3.2:3b", "qwen2.5:3b", "llama3.2:1b"]
    available_recommended = [
        model for model in recommended_models if any(rec in model for rec in models)
    ]

    if available_recommended:
        print("✅ Found recommended models:")
        for model in available_recommended:
            print(f"   - {model}")
    else:
        print("⚠️ No recommended models found!")
        print("\n🔧 Recommended models for speed and accuracy:")
        for model in recommended_models:
            print(f"   - {model}")

        # Offer to install a model
        if len(sys.argv) > 1 and sys.argv[1] == "--install":
            model_to_install = "llama3.2:3b"
            print(f"\n🔽 Installing {model_to_install}...")
            if pull_model(model_to_install):
                print(f"✅ Successfully installed {model_to_install}")
            else:
                print(f"❌ Failed to install {model_to_install}")
        else:
            print(f"\n💡 To install a recommended model, run:")
            print(f"   python {sys.argv[0]} --install")
            print(f"   or manually: ollama pull llama3.2:3b")

    # Test model performance
    print("\n4. Testing model performance...")
    if available_recommended:
        test_model = available_recommended[0]
        print(f"Testing with {test_model}...")

        try:
            # Simple test query
            test_data = {
                "model": test_model.split(":")[0],  # Remove tag for API call
                "prompt": "Extract name from: John Smith, Software Engineer",
                "stream": False,
            }

            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate", json=test_data, timeout=30
            )
            end_time = time.time()

            if response.status_code == 200:
                response_time = end_time - start_time
                print(f"✅ Model responded in {response_time:.2f} seconds")
                if response_time < 10:
                    print("✅ Good response time!")
                elif response_time < 20:
                    print("⚠️ Acceptable response time")
                else:
                    print("❌ Slow response time - consider using a smaller model")
            else:
                print(f"❌ Model test failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Model test failed: {e}")

    print("\n" + "=" * 40)
    print("✅ Setup check complete!")
    print("\n🚀 To test the resume parser:")
    print("   python main.py test")


if __name__ == "__main__":
    main()
