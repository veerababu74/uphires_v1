# LLM Provider Switching Guide

This application now supports switching between different LLM (Large Language Model) providers for resume parsing and text processing. You can dynamically switch between **Groq Cloud** and **Ollama** providers.

## Available Providers

### 1. Ollama (Local)
- **Type**: Local LLM service
- **Requirements**: Ollama must be installed and running on your machine
- **API Keys**: Not required
- **Default Models**: llama3.2:3b, qwen2.5:3b, qwen:4b
- **Advantages**: Privacy, no API costs, works offline
- **Disadvantages**: Requires local setup, may be slower

### 2. Groq Cloud (API)
- **Type**: Cloud-based LLM service
- **Requirements**: Valid Groq API keys
- **API Keys**: Required
- **Default Models**: gemma2-9b-it, llama-3.1-70b-versatile, mixtral-8x7b-32768
- **Advantages**: Fast inference, no local setup required
- **Disadvantages**: Requires internet, API costs

## Configuration

### Environment Variables

Set the default LLM provider using the `LLM_PROVIDER` environment variable:

```bash
# For Ollama (default)
LLM_PROVIDER=ollama

# For Groq Cloud
LLM_PROVIDER=groq
```

### Groq API Keys

If using Groq Cloud, set your API keys:

```bash
GROQ_API_KEYS=your-first-api-key,your-second-api-key,your-third-api-key
```

### Ollama Configuration

For Ollama, configure the service URL and models:

```bash
OLLAMA_API_URL=http://localhost:11434
OLLAMA_PRIMARY_MODEL=llama3.2:3b
OLLAMA_BACKUP_MODEL=qwen2.5:3b
OLLAMA_FALLBACK_MODEL=qwen:4b
```

## API Endpoints

### Global LLM Provider Management

#### 1. Switch LLM Provider Globally
```http
POST /api/llm-provider/switch?provider=ollama
POST /api/llm-provider/switch?provider=groq&api_keys=your-api-key1&api_keys=your-api-key2
```

**Description**: Switch the LLM provider for the entire application.

**Parameters**:
- `provider` (required): Either "groq" or "ollama"
- `api_keys` (optional): Required when switching to Groq provider

**Example Response**:
```json
{
  "status": "success",
  "previous_provider": "groq",
  "current_provider": "ollama",
  "message": "Successfully switched to ollama",
  "validation_passed": true,
  "provider_type": "local",
  "model": "llama3.2:3b",
  "api_url": "http://localhost:11434"
}
```

#### 2. Get Provider Status
```http
GET /api/llm-provider/status
```

**Description**: Get current LLM provider status and configuration.

**Example Response**:
```json
{
  "current_provider": "ollama",
  "provider_type": "local",
  "configuration_valid": true,
  "environment_variable": "ollama",
  "model": "llama3.2:3b",
  "backup_model": "qwen2.5:3b",
  "api_url": "http://localhost:11434",
  "service_status": "running",
  "available_models": ["llama3.2:3b", "qwen2.5:3b", "qwen:4b"]
}
```

#### 3. Test Provider Connection
```http
POST /api/llm-provider/test-connection
POST /api/llm-provider/test-connection?provider=ollama
```

**Description**: Test connection to current or specified LLM provider.

**Example Response**:
```json
{
  "provider": "ollama",
  "status": "success",
  "message": "Ollama connection successful",
  "available_models": ["llama3.2:3b", "qwen2.5:3b"],
  "models_count": 2,
  "primary_model_available": true
}
```

#### 4. List Available Providers
```http
GET /api/llm-provider/providers
```

**Description**: List all available LLM providers and their configurations.

### Module-Specific Provider Switching

#### GroqCloudLLM Module
```http
POST /groqcloud/switch-provider/?provider=ollama
POST /groqcloud/provider-info/
POST /groqcloud/grouqcloud/?provider=ollama
```

#### Multiple Resume Parser Module
```http
POST /resume_parser/switch-provider/?provider=groq&api_keys=your-api-key
POST /resume_parser/provider-info/
POST /resume_parser/grouqcloud/?provider=groq
```

## Usage Examples

### Example 1: Switch to Ollama
```python
import requests

# Switch to Ollama globally
response = requests.post(
    "http://localhost:8000/api/llm-provider/switch",
    params={"provider": "ollama"}
)
print(response.json())

# Test the connection
response = requests.post(
    "http://localhost:8000/api/llm-provider/test-connection"
)
print(response.json())
```

### Example 2: Switch to Groq Cloud
```python
import requests

# Switch to Groq Cloud with API keys
response = requests.post(
    "http://localhost:8000/api/llm-provider/switch",
    params={
        "provider": "groq",
        "api_keys": ["your-api-key-1", "your-api-key-2"]
    }
)
print(response.json())
```

### Example 3: Process Resume with Specific Provider
```python
import requests

# Upload and process resume with Ollama
files = {"file": open("resume.pdf", "rb")}
response = requests.post(
    "http://localhost:8000/groqcloud/grouqcloud/",
    files=files,
    params={"provider": "ollama"}
)
print(response.json())
```

## Setup Instructions

### Setting up Ollama

1. **Install Ollama**:
   ```bash
   # On macOS
   brew install ollama
   
   # On Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # On Windows
   # Download from https://ollama.ai/download
   ```

2. **Start Ollama Service**:
   ```bash
   ollama serve
   ```

3. **Pull Required Models**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ollama pull qwen:4b
   ```

### Setting up Groq Cloud

1. **Get API Keys**:
   - Visit [Groq Console](https://console.groq.com/)
   - Create an account and generate API keys

2. **Configure Environment**:
   ```bash
   export GROQ_API_KEYS="key1,key2,key3"
   ```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**:
   - Ensure Ollama service is running: `ollama serve`
   - Check if the correct port is configured (default: 11434)
   - Verify models are downloaded: `ollama list`

2. **Groq API Errors**:
   - Verify API keys are valid and active
   - Check rate limits and quotas
   - Ensure internet connectivity

3. **Model Not Available**:
   - For Ollama: Pull the required model using `ollama pull <model-name>`
   - For Groq: Check if the model is supported by your API plan

### Logs and Debugging

Check application logs for detailed error information:
- Look for logs with "groqcloud_llm" or "multiple_resume_parser" tags
- LLM provider switching logs are tagged with "llm_provider_api"

## Best Practices

1. **Provider Selection**:
   - Use Ollama for privacy-sensitive applications
   - Use Groq Cloud for high-throughput scenarios
   - Test both providers to compare quality and speed

2. **API Key Management**:
   - Rotate API keys regularly
   - Use multiple keys to avoid rate limits
   - Store keys securely (environment variables, not in code)

3. **Error Handling**:
   - Always test provider connections before processing
   - Implement fallback mechanisms
   - Monitor API usage and quotas

4. **Performance Optimization**:
   - For Ollama: Use appropriate model sizes for your hardware
   - For Groq: Optimize prompt lengths to reduce token usage
   - Monitor response times and adjust timeouts accordingly
