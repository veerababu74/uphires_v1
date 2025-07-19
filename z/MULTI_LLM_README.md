# Multi-LLM Resume Parser

This updated resume parser now supports multiple LLM providers, making it easy to switch between different language models based on your needs and requirements.

## Supported LLM Providers

1. **Ollama** - Local models (recommended for privacy and cost)
2. **Groq Cloud** - Fast API-based inference
3. **OpenAI** - GPT models via API
4. **Google Gemini** - Google's latest models
5. **Hugging Face** - Local models from Hugging Face Hub

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install langchain-groq langchain-ollama langchain-openai langchain-google-genai

# For Hugging Face support (optional)
pip install langchain-huggingface transformers torch

# For GPU support with Hugging Face (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Configure Environment

Copy `env_config_example.env` to `.env` and configure your preferred provider:

```bash
# Set your preferred provider
LLM_PROVIDER=huggingface

# Configure Hugging Face
HUGGINGFACE_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
HUGGINGFACE_DEVICE=auto
```

### 3. Basic Usage

```python
from GroqcloudLLM.main import ResumeParser

# Use default provider from .env
parser = ResumeParser()

# Or specify provider explicitly
parser = ResumeParser(llm_provider="huggingface")

# Process resume
result = parser.process_resume(resume_text)
print(result)
```

## Provider-Specific Setup

### Ollama (Local)
```bash
# Install Ollama
# Download from: https://ollama.ai

# Pull required models
ollama pull llama3.2:3b
ollama pull qwen2.5:3b

# Configure in .env
LLM_PROVIDER=ollama
OLLAMA_PRIMARY_MODEL=llama3.2:3b
```

### Groq Cloud (API)
```bash
# Get API keys from: https://console.groq.com

# Configure in .env
LLM_PROVIDER=groq_cloud
GROQ_API_KEYS=your_key_1,your_key_2
GROQ_PRIMARY_MODEL=gemma2-9b-it
```

### OpenAI (API)
```bash
# Get API keys from: https://platform.openai.com

# Configure in .env
LLM_PROVIDER=openai
OPENAI_API_KEYS=your_key_1,your_key_2
OPENAI_PRIMARY_MODEL=gpt-3.5-turbo
```

### Google Gemini (API)
```bash
# Get API keys from: https://makersuite.google.com

# Configure in .env
LLM_PROVIDER=google_gemini
GOOGLE_API_KEYS=your_key_1,your_key_2
GOOGLE_PRIMARY_MODEL=gemini-1.5-flash
```

### Hugging Face (Local)
```bash
# Configure in .env
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
HUGGINGFACE_DEVICE=auto  # or 'cpu', 'cuda', etc.
```

## Advanced Usage

### Dynamic Provider Switching

```python
parser = ResumeParser(llm_provider="groq")

# Switch to different provider
parser.switch_provider("huggingface")

# Switch to OpenAI with specific API keys
parser.switch_provider("openai", api_keys=["your_key"])
```

### Provider-Specific Configuration

```python
# Initialize with specific provider
parser = ResumeParser(
    llm_provider="huggingface",
    # API keys not needed for Hugging Face
)

# For API-based providers
parser = ResumeParser(
    llm_provider="openai",
    api_keys=["sk-your-key-here"]
)
```

### Configuration Management

```python
from core.llm_config import get_llm_config, configure_llm_provider

# Get current configuration
config = get_llm_config()
print(config.get_status())

# Switch provider programmatically
configure_llm_provider("huggingface")
```

## Environment Variables Reference

### Core Settings
- `LLM_PROVIDER`: Provider to use (ollama, groq_cloud, openai, google_gemini, huggingface)

### Ollama Settings
- `OLLAMA_API_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_PRIMARY_MODEL`: Primary model name
- `OLLAMA_TEMPERATURE`: Generation temperature

### Groq Settings
- `GROQ_API_KEYS`: Comma-separated API keys
- `GROQ_PRIMARY_MODEL`: Model name
- `GROQ_TEMPERATURE`: Generation temperature

### OpenAI Settings
- `OPENAI_API_KEYS`: Comma-separated API keys
- `OPENAI_API_KEY`: Single API key (alternative)
- `OPENAI_PRIMARY_MODEL`: Model name
- `OPENAI_ORGANIZATION`: Organization ID (optional)

### Google Gemini Settings
- `GOOGLE_API_KEYS`: Comma-separated API keys
- `GOOGLE_API_KEY`: Single API key (alternative)
- `GOOGLE_PRIMARY_MODEL`: Model name

### Hugging Face Settings
- `HUGGINGFACE_MODEL_ID`: Model ID (e.g., microsoft/Phi-3-mini-4k-instruct)
- `HUGGINGFACE_DEVICE`: Device to use (auto, cpu, cuda, etc.)
- `HUGGINGFACE_TOKEN`: Hugging Face token for private models

## Recommended Models

### Local Models (Ollama/Hugging Face)
- `llama3.2:3b` - Good balance of speed and quality
- `qwen2.5:3b` - Fast and efficient
- `microsoft/Phi-3-mini-4k-instruct` - Optimized for instructions
- `microsoft/DialoGPT-medium` - Good for conversational tasks

### API Models
- **Groq**: `gemma2-9b-it`, `llama-3.1-70b-versatile`
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`
- **Google**: `gemini-1.5-flash`, `gemini-1.5-pro`

## Performance Considerations

### Speed Ranking (fastest to slowest)
1. Groq Cloud (with good internet)
2. Local Ollama (with good hardware)
3. OpenAI API
4. Google Gemini API
5. Local Hugging Face (depends on hardware)

### Cost Ranking (cheapest to most expensive)
1. Local models (Ollama/Hugging Face) - Free after setup
2. Groq Cloud - Very cheap API calls
3. Google Gemini - Moderate pricing
4. OpenAI - More expensive

### Quality Ranking
1. GPT-4 (OpenAI) - Highest quality
2. Gemini 1.5 Pro (Google)
3. Llama 3.1 70B (Groq)
4. Local large models (depends on model size)

## Troubleshooting

### Common Issues

1. **Import Error for langchain_huggingface**
   ```bash
   pip install langchain-huggingface
   ```

2. **CUDA Out of Memory (Hugging Face)**
   ```bash
   # Use CPU instead
   HUGGINGFACE_DEVICE=cpu
   ```

3. **Ollama Connection Error**
   ```bash
   # Start Ollama service
   ollama serve
   ```

4. **API Rate Limits**
   - Add multiple API keys for automatic rotation
   - Configure retry delays in environment variables

### Model Download Issues (Hugging Face)
```python
# Pre-download models
from transformers import pipeline

pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")
```

## Contributing

To add support for new providers:

1. Add new provider to `LLMProvider` enum in `core/llm_config.py`
2. Create configuration class (e.g., `NewProviderConfig`)
3. Add setup methods in `ResumeParser` class
4. Update validation and switching logic

## License

[Your License Here]
