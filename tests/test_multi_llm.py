#!/usr/bin/env python3
"""
Multi-LLM Resume Parser Test Script
==================================

This script demonstrates how to use the updated resume parser with different LLM providers.
"""

import os
import sys
import json
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GroqcloudLLM.main import ResumeParser
from core.llm_config import get_llm_config, configure_llm_provider


def test_sample_resume():
    """Sample resume text for testing"""
    return """
    RESUME YADAV PANAKJ INDRESHKUMAR 
    Email: yadavanush1234@gmail.com 
    Phone: 9023891599 
    C -499, umiyanagar behind taxshila school Vastral road – ahmedabad -382418 
    
    Career Objective 
    To develop career with an organization which provides me excellent opportunity and enable me to learn skill to achieve organization's goal 
    
    Personal Details  
    Full Name : YADAV PANKAJ INDRESHKUMAR  
    Date of Birth : 14/05/1993  
    Gender : male  
    Marital Status : Married  
    Nationality : Indian  
    Languages Known : Hindi, English, Gujarati  
    Hobbies : Reading 
    
    Work Experience  
    I Have Two Years Experience (BHARAT PETROLEUM ) As Oil Department Supervisor  
    I Have ONE Years Experience ( H D B FINACE SERVICES ) As Sales Executive  
    I Have One Years Experience (MAY GATE SOFTWARE ) As Sales Executive  
    I Have One Years Experience ( BY U Me – SHOREA SOFECH PRIVATE LTD ) As Sales Executive  
    
    Education Details 
    Pass Out 2008 - CGPA/Percentage : 51.00% G.S.E.B 
    Pass Out 2010 - CGPA/Percentage : 55.00% G.H.S.E.B 
    Pass Out 2022 – Running Gujarat.uni 
    
    Interests/Hobbies 
    Listening, music, traveling 
    
    Declaration 
    I hereby declare that all the details furnished above are true to the best of my knowledge and belief. 
    Date://2019 Place: odhav
    """


def print_result(provider: str, result: Dict[Any, Any]):
    """Print formatted result"""
    print(f"\n{'='*60}")
    print(f"PROVIDER: {provider.upper()}")
    print(f"{'='*60}")

    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        if "details" in result:
            print(f"Details: {result['details']}")
    else:
        print("✅ SUCCESS!")
        print(f"Name: {result.get('name', 'N/A')}")
        print(f"Email: {result.get('contact_details', {}).get('email', 'N/A')}")
        print(f"Total Experience: {result.get('total_experience', 'N/A')}")
        print(f"Skills Count: {len(result.get('skills', [{}])[0].get('skills', []))}")
        print(f"Experience Entries: {len(result.get('experience', []))}")

    print(f"{'='*60}")


def test_provider(provider_name: str, sample_resume: str):
    """Test a specific provider"""
    print(f"\n🔄 Testing {provider_name}...")

    try:
        # Configure provider
        if not configure_llm_provider(provider_name):
            print(f"❌ Failed to configure {provider_name}")
            return

        # Initialize parser
        parser = ResumeParser(llm_provider=provider_name)

        # Process resume
        result = parser.process_resume(sample_resume)

        # Print result
        print_result(provider_name, result)

    except Exception as e:
        print(f"❌ Exception with {provider_name}: {str(e)}")


def test_dynamic_switching():
    """Test dynamic provider switching"""
    print("\n🔄 Testing dynamic provider switching...")

    sample_resume = test_sample_resume()

    try:
        # Start with one provider
        parser = ResumeParser(llm_provider="ollama")
        print("✅ Initialized with Ollama")

        # Switch to different providers
        providers = ["groq_cloud", "huggingface"]

        for provider in providers:
            try:
                parser.switch_provider(provider)
                print(f"✅ Switched to {provider}")

                # Test processing
                result = parser.process_resume(sample_resume[:500])  # Short sample
                if "error" not in result:
                    print(f"✅ {provider} processing successful")
                else:
                    print(f"⚠️ {provider} processing failed: {result['error']}")

            except Exception as e:
                print(f"❌ Failed to switch to {provider}: {str(e)}")

    except Exception as e:
        print(f"❌ Dynamic switching test failed: {str(e)}")


def show_configuration_status():
    """Show current configuration status"""
    print("\n📊 CONFIGURATION STATUS")
    print("=" * 60)

    config = get_llm_config()
    status = config.get_status()

    print(f"Current Provider: {status['provider']}")
    print(f"Validated: {status['validated']}")

    print(f"\nOllama Config:")
    print(f"  API URL: {status['ollama_config']['api_url']}")
    print(f"  Primary Model: {status['ollama_config']['primary_model']}")

    print(f"\nGroq Config:")
    print(f"  Primary Model: {status['groq_config']['primary_model']}")
    print(f"  API Keys: {status['groq_config']['keys_count']}")

    if "openai_config" in status:
        print(f"\nOpenAI Config:")
        print(f"  Primary Model: {status['openai_config']['primary_model']}")
        print(f"  API Keys: {status['openai_config']['keys_count']}")

    if "google_config" in status:
        print(f"\nGoogle Config:")
        print(f"  Primary Model: {status['google_config']['primary_model']}")
        print(f"  API Keys: {status['google_config']['keys_count']}")

    if "huggingface_config" in status:
        print(f"\nHugging Face Config:")
        print(f"  Model ID: {status['huggingface_config']['model_id']}")
        print(f"  Device: {status['huggingface_config']['device']}")


def main():
    """Main test function"""
    print("🚀 Multi-LLM Resume Parser Test")
    print("=" * 60)

    # Show configuration
    show_configuration_status()

    # Sample resume
    sample_resume = test_sample_resume()

    # Test available providers
    providers = ["ollama", "huggingface"]  # Start with local providers

    # Try API providers if configured
    config = get_llm_config()
    if config.groq_config.api_keys:
        providers.append("groq_cloud")
    if config.openai_config.api_keys:
        providers.append("openai")
    if config.google_config.api_keys:
        providers.append("google_gemini")

    # Test each provider
    for provider in providers:
        test_provider(provider, sample_resume)

    # Test dynamic switching
    test_dynamic_switching()

    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()
