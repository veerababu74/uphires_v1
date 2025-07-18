#!/usr/bin/env python3
"""
Test script to verify the Hugging Face authentication fix
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_huggingface_auth():
    """Test if Hugging Face authentication issue is resolved"""
    print("Testing Hugging Face authentication fix...")

    try:
        # Import the ResumeParser class
        from GroqcloudLLM.main import ResumeParser

        # Check current LLM provider setting
        llm_provider = os.getenv("LLM_PROVIDER", "not_set")
        print(f"Current LLM_PROVIDER: {llm_provider}")

        if llm_provider.lower() == "huggingface":
            # Check if HF token is set
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if (
                hf_token
                and hf_token != "your_hf_token"
                and hf_token != "your_actual_hf_token_here"
            ):
                print("✅ Hugging Face token is set")
                print("Testing ResumeParser initialization with Hugging Face...")
                parser = ResumeParser()
                print("✅ ResumeParser initialized successfully with Hugging Face!")
            else:
                print("❌ Hugging Face token not properly set")
                print("Please update HUGGINGFACE_TOKEN in your .env file")
                return False

        elif llm_provider.lower() == "ollama":
            print("Testing ResumeParser initialization with Ollama...")
            parser = ResumeParser()
            print("✅ ResumeParser initialized successfully with Ollama!")

        else:
            print(f"Testing ResumeParser initialization with {llm_provider}...")
            parser = ResumeParser()
            print(f"✅ ResumeParser initialized successfully with {llm_provider}!")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("HUGGING FACE AUTHENTICATION FIX TEST")
    print("=" * 60)

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    success = test_huggingface_auth()

    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Authentication issue resolved!")
        print("Your application should now work correctly.")
    else:
        print("❌ TEST FAILED: Please check the error above.")
        print("\nRecommended fixes:")
        print("1. Get HuggingFace token from: https://huggingface.co/settings/tokens")
        print("2. Update HUGGINGFACE_TOKEN in your .env file")
        print("3. Or switch to Ollama by setting LLM_PROVIDER=ollama")
    print("=" * 60)


if __name__ == "__main__":
    main()
