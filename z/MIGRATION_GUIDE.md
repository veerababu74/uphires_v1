# Multi-LLM Migration Guide

This guide helps you migrate from the old dual-provider system (Ollama + Groq) to the new multi-LLM system supporting 5 providers.

## 🔄 Migration Overview

### Before (Old System)
```python
# Old way - limited to Ollama or Groq
parser = ResumeParser(use_ollama=True)  # or False for Groq
```

### After (New System)
```python
# New way - supports all providers
parser = ResumeParser(llm_provider="huggingface")  # or any provider
```

## 📋 Environment Variable Changes

### Old Variables (Still Supported)
```bash
LLM_PROVIDER=ollama  # or groq_cloud
GROQ_API_KEY=your_key
```

### New Variables (Recommended)
```bash
# Provider Selection
LLM_PROVIDER=huggingface  # or ollama, groq_cloud, openai, google_gemini

# Hugging Face Configuration
HUGGINGFACE_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
HUGGINGFACE_DEVICE=auto

# OpenAI Configuration (if using)
OPENAI_API_KEYS=your_key_1,your_key_2
OPENAI_PRIMARY_MODEL=gpt-3.5-turbo

# Google Gemini Configuration (if using)
GOOGLE_API_KEYS=your_key_1,your_key_2
GOOGLE_PRIMARY_MODEL=gemini-1.5-flash

# Enhanced Groq Configuration
GROQ_API_KEYS=your_key_1,your_key_2,your_key_3  # Multiple keys supported
GROQ_PRIMARY_MODEL=gemma2-9b-it
```

## 🔧 Code Migration Examples

### 1. GroqcloudLLM Module

#### Before:
```python
from GroqcloudLLM.main import ResumeParser

# Old constructor
parser = ResumeParser(
    use_ollama=False,  # Boolean choice
    api_keys=["your_groq_key"]
)
```

#### After:
```python
from GroqcloudLLM.main import ResumeParser

# New constructor - supports all providers
parser = ResumeParser(
    llm_provider="huggingface",  # String choice
    api_keys=["your_key"]  # Only needed for API-based providers
)

# Or use environment configuration
parser = ResumeParser()  # Uses LLM_PROVIDER from .env
```

### 2. Multiple Resume Parser

#### Before:
```python
from multipleresumepraser.main import ResumeParser

parser = ResumeParser(use_ollama=True)
```

#### After:
```python
from multipleresumepraser.main import ResumeParser

parser = ResumeParser(llm_provider="ollama")
# or
parser = ResumeParser()  # Uses environment config
```

### 3. RAG Application

#### Before:
```python
from Rag.rag_application import RAGApplication

# RAG used hardcoded Groq configuration
rag_app = RAGApplication()
```

#### After:
```python
from Rag.rag_application import RAGApplication

# RAG now uses centralized LLM factory
# Set LLM_PROVIDER in environment first
rag_app = RAGApplication()
```

## 🚀 Quick Migration Steps

### Step 1: Update Environment Configuration
```bash
# Copy the example configuration
cp env_config_example.env .env

# Edit .env with your preferred provider
LLM_PROVIDER=huggingface  # Choose your provider
HUGGINGFACE_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
```

### Step 2: Update Code (If Needed)
Most code will work without changes if you set the environment variables correctly. Only update code if you were passing parameters directly.

### Step 3: Install New Dependencies
```bash
# For all providers
pip install -r requirements.txt

# For Hugging Face specifically
pip install langchain-huggingface transformers torch
```

### Step 4: Test Your Migration
```bash
# Run the comprehensive test
python test_multi_llm_comprehensive.py
```

## 🔍 Troubleshooting

### Issue: ImportError for langchain_huggingface
```bash
# Solution: Install the package
pip install langchain-huggingface
```

### Issue: Model not found (Hugging Face)
```python
# Solution: Pre-download the model
from transformers import pipeline
pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")
```

### Issue: CUDA out of memory
```bash
# Solution: Use CPU
HUGGINGFACE_DEVICE=cpu
```

### Issue: API key errors
```bash
# Solution: Check your .env file
# Make sure keys are correctly formatted:
OPENAI_API_KEYS=sk-key1,sk-key2
GOOGLE_API_KEYS=AIza-key1,AIza-key2
```

## 📊 Provider Comparison

| Provider | Cost | Speed | Quality | Local | API Keys |
|----------|------|-------|---------|-------|----------|
| Ollama | Free | Fast* | Good | ✅ | ❌ |
| Hugging Face | Free | Medium* | Good | ✅ | ❌ |
| Groq Cloud | Very Low | Very Fast | Good | ❌ | ✅ |
| OpenAI | Medium | Fast | Excellent | ❌ | ✅ |
| Google Gemini | Low | Fast | Excellent | ❌ | ✅ |

*Speed depends on hardware for local models

## 🎯 Recommended Configurations

### For Development/Testing
```bash
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
HUGGINGFACE_DEVICE=cpu  # Works without GPU
```

### For Production (Cost-Effective)
```bash
LLM_PROVIDER=groq_cloud
GROQ_API_KEYS=your_key_1,your_key_2
GROQ_PRIMARY_MODEL=gemma2-9b-it
```

### For Production (High Quality)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEYS=your_key_1,your_key_2
OPENAI_PRIMARY_MODEL=gpt-3.5-turbo
```

### For Local/Private Processing
```bash
LLM_PROVIDER=ollama
OLLAMA_PRIMARY_MODEL=llama3.2:3b
```

## 🆕 New Features

1. **Automatic Fallback**: If primary provider fails, system tries fallback providers
2. **API Key Rotation**: Multiple API keys are rotated automatically for rate limiting
3. **Provider Switching**: Switch providers at runtime without restarting
4. **Centralized Configuration**: Single source of truth for all LLM settings
5. **Enhanced Logging**: Better error messages and debugging information

## 🔮 Future Additions

The new architecture makes it easy to add new providers:
- Anthropic Claude
- Cohere
- Local models via vLLM
- Custom API endpoints

To add a new provider, you only need to:
1. Add provider to `LLMProvider` enum
2. Create configuration class
3. Add factory method
4. Update validation logic

## 📞 Support

If you encounter issues during migration:

1. Check the logs in the `logs/` directory
2. Run the test suite: `python test_multi_llm_comprehensive.py`
3. Verify your .env configuration matches the examples
4. Ensure all dependencies are installed

The migration should be smooth for most users. The new system is backward compatible with the old environment variable format.
