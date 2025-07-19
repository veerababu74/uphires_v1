#!/usr/bin/env python3
"""
Demonstration script that shows the RAG model fallback behavior similar to what's shown in the logs.
This script will temporarily simulate the scenario where llama3.2:3b is not available.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.custom_logger import CustomLogger
from Rag.config import RAGConfig
import requests

# Setup logging
logger = CustomLogger().get_logger("rag_application")


class MockedRAGApp:
    """Mocked RAG application to demonstrate fallback behavior"""

    def simulate_fallback_scenario(self):
        """Simulate the exact scenario from the logs"""
        logger.info("LangChain vector store initialized successfully")
        logger.info("Initializing ResumeParser with Ollama")

        # Simulate the model checking and fallback logic
        self._initialize_ollama_with_fallback_demo()

    def _initialize_ollama_with_fallback_demo(self):
        """Demonstrate the fallback logic with logging similar to the original logs"""
        import requests
        import time

        # Define model fallback order
        primary_model = "llama3.2:3b"  # The model that's not available in your logs
        backup_model = RAGConfig.OLLAMA_BACKUP_MODEL
        fallback_model = "qwen:4b"  # The model that was used as fallback

        # Add some delay to simulate checking
        time.sleep(4)

        # Check Ollama connection and get available models
        try:
            response = requests.get(f"{RAGConfig.OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]

                # Remove llama3.2:3b from available models to simulate the scenario
                demo_available_models = [
                    model for model in available_models if model != "llama3.2:3b"
                ]

                logger.warning(f"{primary_model} not found.")

                time.sleep(2)

                logger.info(f"Available models: {demo_available_models}")
                logger.info(f"Using fallback model: {fallback_model}")

                time.sleep(1)

                logger.info(
                    f"Ollama LLM initialized successfully with model: {fallback_model}"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to connect to Ollama service: {e}")
            return False


def demonstrate_actual_behavior():
    """Show the actual current behavior"""
    print("\n" + "=" * 80)
    print("DEMONSTRATING RAG MODEL FALLBACK BEHAVIOR")
    print("=" * 80)

    logger.info("Starting RAG model fallback demonstration...")

    # Get current available models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]

            print(f"\nCurrent available models: {available_models}")

            if "llama3.2:3b" in available_models:
                print("✅ llama3.2:3b is currently available (no fallback needed)")

                # Simulate what would happen if it wasn't available
                print("\n🎭 SIMULATING SCENARIO WHERE llama3.2:3b IS NOT AVAILABLE:")
                print("-" * 60)

                # Create mocked RAG app
                mock_app = MockedRAGApp()
                mock_app.simulate_fallback_scenario()

            else:
                print("⚠️ llama3.2:3b is not available (fallback will be used)")

                # Test actual fallback
                print("\n🔧 TESTING ACTUAL FALLBACK:")
                print("-" * 60)

                from Rag.rag_application import RAGApplication

                rag_app = RAGApplication()

                if hasattr(rag_app, "llm") and rag_app.llm:
                    print(f"✅ RAG initialized with model: {rag_app.llm.model}")

        else:
            print("❌ Could not connect to Ollama service")

    except Exception as e:
        print(f"❌ Error: {e}")


def show_configuration():
    """Show current RAG configuration"""
    print("\n" + "=" * 80)
    print("CURRENT RAG CONFIGURATION")
    print("=" * 80)

    print(f"Primary model:  {RAGConfig.OLLAMA_MODEL}")
    print(f"Backup model:   {RAGConfig.OLLAMA_BACKUP_MODEL}")
    print(f"Fallback model: {RAGConfig.OLLAMA_FALLBACK_MODEL}")
    print(f"Ollama API URL: {RAGConfig.OLLAMA_API_URL}")


if __name__ == "__main__":
    print("🚀 RAG Model Fallback Demonstration")

    # Show configuration
    show_configuration()

    # Demonstrate behavior
    demonstrate_actual_behavior()

    print("\n✨ Demonstration completed!")
    print(
        "\nNow your RAG system includes the same model fallback logic as other components:"
    )
    print("1. Checks if llama3.2:3b is available")
    print("2. If not found, tries backup model (qwen2.5:3b)")
    print("3. If backup not found, tries fallback model (qwen:4b)")
    print("4. As last resort, uses any available qwen model")
    print("5. Logs the selection process similar to other components")
