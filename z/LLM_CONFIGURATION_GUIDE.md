# LLM Configuration System Documentation

## Overview

The new LLM Configuration System provides a centralized way to manage and switch between different LLM providers (Ollama and Groq Cloud) with comprehensive configuration options.

## Features

- **Centralized Configuration**: All LLM settings managed in one place
- **Provider Switching**: Easy switching between Ollama and Groq Cloud
- **Automatic Fallbacks**: Intelligent fallback mechanisms when providers fail
- **Environment-based Config**: Configure via environment variables
- **API Management**: RESTful API for configuration management
- **Validation**: Comprehensive configuration validation
- **Backward Compatibility**: Works with existing code

## Architecture

```
core/
├── llm_config.py       # Core configuration classes
├── llm_factory.py      # LLM instance factory
└── llm_config_example.py  # Usage examples

apis/
└── llm_config_api.py   # REST API for config management
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# ===========================================
# LLM PROVIDER CONFIGURATION
# ===========================================
# Choose your LLM provider: "ollama" or "groq_cloud"
LLM_PROVIDER=ollama

# ===========================================
# OLLAMA CONFIGURATION
# ===========================================
OLLAMA_API_URL=http://localhost:11434
OLLAMA_PRIMARY_MODEL=llama3.2:3b
OLLAMA_BACKUP_MODEL=qwen2.5:3b
OLLAMA_TEMPERATURE=0.1
OLLAMA_TIMEOUT=30

# ===========================================
# GROQ CLOUD CONFIGURATION
# ===========================================
GROQ_API_KEYS=your_api_key_1,your_api_key_2
GROQ_PRIMARY_MODEL=gemma2-9b-it
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=1024
```

## Usage Examples

### Basic Usage

```python
from core.llm_factory import create_llm

# Create LLM with current configuration
llm = create_llm()

# Use the LLM
response = llm.invoke("Hello, how are you?")
print(response)
```

### Provider Management

```python
from core.llm_config import get_llm_config, configure_llm_provider

# Get current configuration
config = get_llm_config()
print(f"Current provider: {config.provider.value}")

# Switch to Ollama
success = configure_llm_provider("ollama")
if success:
    print("Switched to Ollama")

# Switch to Groq Cloud
success = configure_llm_provider("groq_cloud")
if success:
    print("Switched to Groq Cloud")
```

### Testing and Validation

```python
from core.llm_factory import test_llm_connection, get_provider_status

# Test current LLM connection
if test_llm_connection():
    print("LLM is working correctly")

# Get status of all providers
status = get_provider_status()
print(f"Ollama available: {status['ollama']['available']}")
print(f"Groq available: {status['groq']['available']}")
```

### Advanced Configuration

```python
from core.llm_config import get_llm_config
from core.llm_factory import LLMFactory, LLMProvider

config_manager = get_llm_config()

# Get specific provider configuration
ollama_config = config_manager.ollama_config
groq_config = config_manager.groq_config

# Create LLM with specific provider
llm = LLMFactory.create_llm(force_provider=LLMProvider.OLLAMA)

# Create chat-oriented LLM
chat_llm = LLMFactory.create_chat_llm()
```

## API Endpoints

The system provides REST API endpoints for configuration management:

### Get Status
```
GET /api/llm-config/status
```
Returns current configuration status and provider availability.

### Switch Provider
```
POST /api/llm-config/switch-provider
{
    "provider": "ollama"  // or "groq_cloud"
}
```

### Test Connection
```
POST /api/llm-config/test
{
    "provider": "ollama",  // optional
    "prompt": "Hello"      // optional
}
```

### List Providers
```
GET /api/llm-config/providers
```
Returns information about all available providers.

### Health Check
```
GET /api/llm-config/health
```
Quick health check for current LLM provider.

## Configuration Classes

### OllamaConfig
Manages Ollama-specific settings:
- API URL and timeouts
- Model selection (primary/backup/fallback)
- Generation parameters (temperature, top_k, top_p, etc.)
- Advanced settings (num_predict, repeat_penalty, etc.)

### GroqConfig
Manages Groq Cloud settings:
- API key management and rotation
- Model selection and fallbacks
- Rate limiting settings
- Request timeouts and retries

### LLMConfigManager
Central configuration manager:
- Provider switching
- Configuration validation
- Status monitoring
- Environment integration

## Integration with Existing Code

### RAG Applications
```python
# Old way
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="qwen:4b", temperature=0.0)

# New way
from core.llm_factory import create_llm
llm = create_llm()  # Uses configured provider automatically
```

### Resume Parser
```python
# The ResumeParser class now automatically uses centralized config
from GroqcloudLLM.main import ResumeParser

# Uses configuration from LLM_PROVIDER environment variable
parser = ResumeParser()

# Or override manually
parser = ResumeParser(use_ollama=True)  # Force Ollama
parser = ResumeParser(use_ollama=False, api_keys=["key1", "key2"])  # Force Groq
```

## Migration Guide

### From Hardcoded Ollama
```python
# Before
llm = OllamaLLM(
    model="qwen:4b",
    temperature=0.0,
    base_url="http://localhost:11434"
)

# After
from core.llm_factory import create_llm
llm = create_llm()  # Uses environment configuration
```

### From Hardcoded Groq
```python
# Before
llm = ChatGroq(
    api_key="your_key",
    model="gemma2-9b-it",
    temperature=0.0
)

# After
# Set LLM_PROVIDER=groq_cloud in .env
from core.llm_factory import create_llm
llm = create_llm()  # Uses environment configuration
```

## Benefits

1. **Flexibility**: Easy switching between providers
2. **Scalability**: Multiple API keys and load balancing
3. **Reliability**: Automatic fallbacks and error handling
4. **Maintainability**: Centralized configuration management
5. **Developer Experience**: Simple API and comprehensive validation
6. **Production Ready**: Rate limiting, monitoring, and health checks

## Error Handling

The system provides comprehensive error handling:

- **Connection Errors**: Automatic provider fallbacks
- **API Rate Limits**: Key rotation and retry logic
- **Model Unavailability**: Fallback to backup models
- **Configuration Issues**: Clear error messages and validation

## Best Practices

1. **Environment Configuration**: Use `.env` file for all settings
2. **Provider Testing**: Test both providers in development
3. **Error Handling**: Always handle LLM creation errors
4. **Monitoring**: Use health check endpoints in production
5. **API Keys**: Rotate keys regularly and use multiple keys for load balancing

## Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

### Groq Issues
- Verify API keys are valid
- Check rate limits in Groq console
- Ensure internet connectivity

### Configuration Issues
```python
# Validate configuration
from core.llm_config import get_llm_config
config = get_llm_config()
is_valid = config.validate_configuration()
print(f"Configuration valid: {is_valid}")

# Get detailed status
status = config.get_status()
print(status)
```

## Future Enhancements

- Support for additional providers (OpenAI, Anthropic, etc.)
- Advanced load balancing strategies
- Configuration UI dashboard
- Metrics and analytics
- Cost tracking and optimization
