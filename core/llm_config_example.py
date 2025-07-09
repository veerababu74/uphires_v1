"""
LLM Configuration Example
========================

This script demonstrates how to use the new centralized LLM configuration system.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_config import get_llm_config, LLMProvider, configure_llm_provider
from core.llm_factory import create_llm, test_llm_connection, get_provider_status


def main():
    """Demonstrate LLM configuration usage"""

    print("=== LLM Configuration Example ===\n")

    # Get the configuration manager
    config_manager = get_llm_config()

    # Show current status
    print("1. Current Configuration Status:")
    status = config_manager.get_status()
    print(f"   Provider: {status['provider']}")
    print(f"   Validated: {status['validated']}")
    print(f"   Ollama API URL: {status['ollama_config']['api_url']}")
    print(f"   Groq Keys Count: {status['groq_keys_count']}")
    print()

    # Validate current configuration
    print("2. Validating Configuration:")
    is_valid = config_manager.validate_configuration()
    print(f"   Configuration Valid: {is_valid}")

    if is_valid:
        print("   ✓ Configuration is valid and ready to use")
    else:
        print("   ✗ Configuration has issues")
    print()

    # Show provider status
    print("3. Provider Status:")
    provider_status = get_provider_status()
    print(f"   Current Provider: {provider_status['current_provider']}")
    print(f"   Ollama Available: {provider_status['ollama']['available']}")
    print(f"   Groq Available: {provider_status['groq']['available']}")

    if provider_status["ollama"]["available"]:
        print(
            f"   Ollama Models: {provider_status['ollama']['models'][:3]}..."
        )  # Show first 3

    print()

    # Test LLM connection
    print("4. Testing LLM Connection:")
    try:
        test_success = test_llm_connection()
        if test_success:
            print("   ✓ LLM connection test successful")
        else:
            print("   ✗ LLM connection test failed")
    except Exception as e:
        print(f"   ✗ LLM test error: {e}")
    print()

    # Create and use LLM
    print("5. Creating LLM Instance:")
    try:
        llm = create_llm()
        print(f"   ✓ LLM created successfully: {type(llm).__name__}")

        # Test a simple prompt
        print("   Testing simple prompt...")
        try:
            if hasattr(llm, "invoke"):
                response = llm.invoke("Say hello in one word")
            else:
                response = llm("Say hello in one word")
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   ✗ Prompt test failed: {e}")

    except Exception as e:
        print(f"   ✗ Failed to create LLM: {e}")
    print()

    # Demonstrate provider switching (if both are available)
    print("6. Provider Switching Demo:")
    current_provider = config_manager.provider.value

    if provider_status["ollama"]["available"] and provider_status["groq"]["available"]:
        print("   Both providers available, demonstrating switch...")

        # Switch to the other provider
        other_provider = "groq_cloud" if current_provider == "ollama" else "ollama"

        print(f"   Switching from {current_provider} to {other_provider}...")
        success = configure_llm_provider(other_provider)

        if success:
            print(f"   ✓ Successfully switched to {other_provider}")

            # Test the new provider
            test_success = test_llm_connection()
            if test_success:
                print("   ✓ New provider working correctly")
            else:
                print("   ✗ New provider test failed")

            # Switch back
            print(f"   Switching back to {current_provider}...")
            configure_llm_provider(current_provider)

        else:
            print(f"   ✗ Failed to switch to {other_provider}")

    else:
        available_providers = []
        if provider_status["ollama"]["available"]:
            available_providers.append("ollama")
        if provider_status["groq"]["available"]:
            available_providers.append("groq_cloud")

        print(f"   Available providers: {available_providers}")
        print("   Need both Ollama and Groq configured to demo switching")

    print()

    # Show configuration recommendations
    print("7. Configuration Recommendations:")

    if not provider_status["ollama"]["available"]:
        print("   • Install and start Ollama for local LLM support")
        print("     - Download from: https://ollama.ai/")
        print("     - Run: ollama pull llama3.2:3b")

    if not provider_status["groq"]["available"]:
        print("   • Add Groq API keys for cloud LLM support")
        print("     - Get API key from: https://console.groq.com/")
        print("     - Set GROQ_API_KEYS in .env file")

    if provider_status["ollama"]["available"] and provider_status["groq"]["available"]:
        print("   ✓ Both providers configured - you can switch between them anytime!")

    print()
    print("=== Example Complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()
