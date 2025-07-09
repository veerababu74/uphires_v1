# Ollama Testing API Documentation

## Overview
This API provides endpoints to interact with Ollama LLM (Qwen:4b model) for testing and development purposes. Users can ask questions and receive AI-generated responses.

## Base URL
```
http://127.0.0.1:8000/api
```

## Available Endpoints

### 1. Health Check
**GET** `/ollama/health`

Check if Ollama service is running and healthy.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "message": "Ollama service is running"
}
```

### 2. Simple Test
**GET** `/ollama/test`

Simple test endpoint to verify Ollama is working with a predefined question.

**Response:**
```json
{
  "success": true,
  "test_question": "Hello! Please introduce yourself in one sentence.",
  "test_answer": "I'm an AI language model designed to assist with tasks and answer questions.",
  "message": "Ollama is working correctly!"
}
```

### 3. Ask Question
**POST** `/ollama/ask`

Ask a question to Ollama and get a detailed response with metadata.

**Request Body:**
```json
{
  "question": "What is Python programming language?",
  "model": "qwen:4b",
  "temperature": 0.7,
  "max_tokens": null
}
```

**Response:**
```json
{
  "question": "What is Python programming language?",
  "answer": "Python is a high-level programming language...",
  "model": "qwen:4b",
  "response_time": 7.28,
  "timestamp": "2025-07-01 21:36:25",
  "success": true,
  "error_message": null
}
```

### 4. Chat Interface
**POST** `/ollama/chat`

Simple chat interface with more conversational responses.

**Request Body:**
```json
{
  "question": "Tell me a joke about programming",
  "model": "qwen:4b",
  "temperature": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "response": "Sure, here's a programming joke:\nWhy did the programmer quit his job?\nBecause he didn't get arrays!",
  "model": "qwen:4b",
  "response_time_seconds": 7.03,
  "timestamp": "2025-07-01 21:39:23"
}
```

### 5. Available Models
**GET** `/ollama/models`

Get list of available Ollama models.

**Response:**
```json
{
  "success": true,
  "models": ["qwen:4b"],
  "total_models": 1
}
```

### 6. Test UI
**GET** `/ollama/test-ui`

Redirects to the interactive testing interface.

## Interactive Web Interface

Access the interactive testing interface at:
```
http://127.0.0.1:8000/static/ollama_test.html
```

The web interface provides:
- Real-time Ollama status indicator
- Quick question buttons for common queries
- Customizable model and temperature settings
- Response time and metadata display
- User-friendly error handling

## Model Configuration

### Current Model: Qwen:4b
- **Size**: 2.3 GB
- **Type**: Chat/Instruction model
- **Temperature Range**: 0.0 (deterministic) to 1.0 (creative)
- **Recommended Temperature**: 0.7 for balanced responses

## Usage Examples

### Using cURL

1. **Health Check:**
```bash
curl http://127.0.0.1:8000/api/ollama/health
```

2. **Ask a Question:**
```bash
curl -X POST http://127.0.0.1:8000/api/ollama/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain machine learning in simple terms"}'
```

3. **Chat Interface:**
```bash
curl -X POST http://127.0.0.1:8000/api/ollama/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of life?", "temperature": 0.8}'
```

### Using Python

```python
import requests

# Ask a question
response = requests.post('http://127.0.0.1:8000/api/ollama/ask', 
                        json={'question': 'What is AI?'})
print(response.json()['answer'])

# Chat interface
response = requests.post('http://127.0.0.1:8000/api/ollama/chat', 
                        json={'question': 'Tell me about Python'})
print(response.json()['response'])
```

## Error Handling

All endpoints return structured error responses:

```json
{
  "success": false,
  "error": "Error description",
  "timestamp": "2025-07-01 21:39:23"
}
```

## Performance Notes

- Average response time: 5-10 seconds for Qwen:4b model
- Response time depends on question complexity and length
- Temperature affects creativity but not significantly response time
- Model runs locally on CPU (no GPU acceleration configured)

## API Documentation

Full interactive API documentation is available at:
```
http://127.0.0.1:8000/docs
```

Look for the "Ollama Testing" section in the Swagger UI.
