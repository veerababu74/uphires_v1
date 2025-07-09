# API Documentation - Resume Search API

## Overview

This document provides detailed information about all API endpoints available in the Resume Search API. The API is built with FastAPI and provides interactive documentation at `/docs` when running.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production use, consider implementing API key authentication or OAuth2.

## Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

For errors:
```json
{
  "success": false,
  "error": "Error description",
  "details": {},
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Endpoints

### Health Check Endpoints

#### GET /health/health
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "service": "Resume Search API",
  "version": "1.0.0"
}
```

#### GET /health/detailed
Detailed health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "service": "Resume Search API",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "message": "MongoDB connection successful"
    },
    "configuration": {
      "status": "healthy",
      "atlas_search_enabled": true,
      "llm_provider": "ollama"
    }
  },
  "response_time_ms": 45.23
}
```

#### GET /health/database
Database-specific health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "connection_time_ms": 12.34,
  "database": {
    "name": "resume_db",
    "collection": "resumes",
    "document_count": 1500,
    "storage_size_mb": 25.6
  }
}
```

### Search Endpoints

#### POST /api/vector_search
Perform vector-based similarity search on resumes.

**Request Body:**
```json
{
  "query": "experienced python developer with machine learning",
  "k": 10,
  "user_id": "user123",
  "filters": {
    "total_experience_min": 3,
    "total_experience_max": 10,
    "current_salary_min": 50000,
    "expected_salary_max": 100000,
    "skills": ["python", "machine learning"],
    "current_city": "Bangalore",
    "notice_period_max": 30
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "user_id": "user456",
        "score": 0.95,
        "resume_data": {
          "contact_details": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone_number": "+91-9876543210",
            "current_city": "Bangalore"
          },
          "skills": ["Python", "Machine Learning", "Django"],
          "total_experience": 5,
          "current_salary": 80000,
          "expected_salary": 95000,
          "notice_period": 30
        }
      }
    ],
    "total_results": 25,
    "search_metadata": {
      "query_vector_generated": true,
      "search_time_ms": 156,
      "filters_applied": 7
    }
  }
}
```

#### POST /api/manual_search
Traditional search with filters and pagination.

**Request Body:**
```json
{
  "skills": ["Python", "React"],
  "total_experience_min": 2,
  "total_experience_max": 8,
  "current_city": "Mumbai",
  "notice_period_max": 60,
  "page": 1,
  "limit": 20,
  "sort_by": "total_experience",
  "sort_order": "desc"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [...],
    "pagination": {
      "current_page": 1,
      "total_pages": 5,
      "total_results": 89,
      "limit": 20,
      "has_next": true,
      "has_previous": false
    }
  }
}
```

#### POST /api/rag_search
AI-powered contextual search using RAG (Retrieval-Augmented Generation).

**Request Body:**
```json
{
  "query": "Find me a senior full-stack developer who can lead a team and has worked with modern web technologies",
  "user_id": "user123",
  "max_results": 10,
  "include_reasoning": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "ai_response": "Based on your requirements, I found several senior full-stack developers...",
    "candidates": [...],
    "reasoning": "The search focused on leadership experience and modern web technologies...",
    "search_metadata": {
      "llm_provider": "ollama",
      "model_used": "llama3.2:3b",
      "response_time_ms": 2341
    }
  }
}
```

### Data Management Endpoints

#### POST /api/add_user_data
Add or update resume data for a user.

**Request Body:**
```json
{
  "user_id": "user789",
  "contact_details": {
    "name": "Jane Smith",
    "email": "jane@example.com",
    "phone_number": "+91-8765432109",
    "current_city": "Pune",
    "pan_card": "ABCDE1234F",
    "addhar_number": "123456789012"
  },
  "skills": ["Java", "Spring Boot", "Microservices"],
  "may_also_known_skills": ["Docker", "Kubernetes"],
  "total_experience": 6,
  "current_salary": 90000,
  "expected_salary": 110000,
  "notice_period": 45,
  "experience": [
    {
      "title": "Senior Software Engineer",
      "company": "Tech Corp",
      "duration": "3 years",
      "description": "Led development of microservices architecture"
    }
  ],
  "education": [
    {
      "degree": "B.Tech Computer Science",
      "institution": "XYZ University",
      "year": "2018"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user_id": "user789",
    "operation": "created",
    "vector_embedded": true
  },
  "message": "User data added successfully"
}
```

### Autocomplete Endpoints

#### GET /api/autocomplete_skills
Get skill suggestions for autocomplete.

**Query Parameters:**
- `q` (string): Search term
- `limit` (int): Maximum results (default: 10)

**Example:** `/api/autocomplete_skills?q=pyth&limit=5`

**Response:**
```json
{
  "success": true,
  "data": {
    "suggestions": [
      "Python",
      "Python Django",
      "Python Flask",
      "Python FastAPI",
      "Python Machine Learning"
    ]
  }
}
```

#### GET /api/autocomplete_titles
Get job title suggestions.

**Query Parameters:**
- `q` (string): Search term
- `limit` (int): Maximum results (default: 10)

**Response:**
```json
{
  "success": true,
  "data": {
    "suggestions": [
      "Software Engineer",
      "Senior Software Engineer",
      "Software Developer",
      "Full Stack Developer",
      "Backend Developer"
    ]
  }
}
```

#### GET /api/cities
Get city suggestions for location search.

**Query Parameters:**
- `q` (string): Search term
- `limit` (int): Maximum results (default: 10)

**Response:**
```json
{
  "success": true,
  "data": {
    "cities": [
      "Bangalore",
      "Mumbai",
      "Pune",
      "Delhi",
      "Hyderabad"
    ]
  }
}
```

### Search History Endpoints

#### GET /api/recent_searches
Get recent search history for a user.

**Query Parameters:**
- `user_id` (string): User identifier
- `limit` (int): Maximum results (default: 10)
- `search_type` (string): Filter by search type (manual, vector, ai)

**Response:**
```json
{
  "success": true,
  "data": {
    "searches": [
      {
        "search_id": "search123",
        "user_id": "user123",
        "search_type": "vector",
        "query": "python developer",
        "filters": {...},
        "timestamp": "2025-01-01T10:30:00Z",
        "results_count": 25
      }
    ]
  }
}
```

#### POST /api/save_search
Save a search for later reference.

**Request Body:**
```json
{
  "user_id": "user123",
  "search_query": "senior python developer",
  "search_type": "vector",
  "filters": {...},
  "search_name": "Python Seniors Q1 2025"
}
```

### Masking Endpoints

#### POST /masking/mask_resume
Apply data masking to resume information for privacy.

**Request Body:**
```json
{
  "user_id": "user123",
  "mask_fields": ["email", "phone_number", "pan_card", "addhar_number"],
  "mask_level": "partial"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "masked_resume": {
      "contact_details": {
        "name": "John Doe",
        "email": "j***@example.com",
        "phone_number": "+91-98***43210"
      }
    }
  }
}
```

### LLM Configuration Endpoints

#### GET /api/llm_config
Get current LLM configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "provider": "ollama",
    "primary_model": "llama3.2:3b",
    "backup_model": "qwen2.5:3b",
    "temperature": 0.1,
    "max_tokens": 1024,
    "api_url": "http://localhost:11434"
  }
}
```

#### POST /api/llm_config
Update LLM configuration.

**Request Body:**
```json
{
  "provider": "groq_cloud",
  "primary_model": "gemma2-9b-it",
  "temperature": 0.2,
  "max_tokens": 2048
}
```

### Search Index Management

#### GET /search_index/status
Get search index status and information.

**Response:**
```json
{
  "success": true,
  "data": {
    "indexes": [
      {
        "name": "vector_search_index",
        "type": "vectorSearch",
        "status": "READY",
        "dimensions": 384,
        "document_count": 1500
      }
    ]
  }
}
```

#### POST /search_index/rebuild
Rebuild search indexes (admin operation).

**Request Body:**
```json
{
  "index_name": "vector_search_index",
  "force": false
}
```

## Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | VALIDATION_ERROR | Invalid request data |
| 404 | NOT_FOUND | Resource not found |
| 422 | UNPROCESSABLE_ENTITY | Request data validation failed |
| 500 | INTERNAL_ERROR | Server internal error |
| 503 | SERVICE_UNAVAILABLE | External service unavailable |

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting based on:
- IP address
- User ID
- API endpoint

## Request/Response Examples

### Complete Vector Search Flow

1. **Health Check**
```bash
curl -X GET "http://localhost:8000/health/health"
```

2. **Vector Search**
```bash
curl -X POST "http://localhost:8000/api/vector_search" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "experienced python developer",
    "k": 5,
    "user_id": "user123",
    "filters": {
      "total_experience_min": 3,
      "skills": ["python"]
    }
  }'
```

3. **Save Search**
```bash
curl -X POST "http://localhost:8000/api/save_search" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user123",
    "search_query": "experienced python developer",
    "search_type": "vector",
    "search_name": "Python Developers - Jan 2025"
  }'
```

## SDK Examples

### Python SDK Usage
```python
import requests

class ResumeSearchAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def vector_search(self, query, k=10, filters=None):
        payload = {
            "query": query,
            "k": k,
            "filters": filters or {}
        }
        response = requests.post(f"{self.base_url}/api/vector_search", json=payload)
        return response.json()
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health/health")
        return response.json()

# Usage
api = ResumeSearchAPI()
results = api.vector_search("python developer", k=5)
```

### JavaScript SDK Usage
```javascript
class ResumeSearchAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async vectorSearch(query, k = 10, filters = {}) {
        const response = await fetch(`${this.baseUrl}/api/vector_search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, k, filters })
        });
        return await response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health/health`);
        return await response.json();
    }
}

// Usage
const api = new ResumeSearchAPI();
const results = await api.vectorSearch('python developer', 5);
```

## Postman Collection

A Postman collection is available for testing all endpoints. Import the collection using the following URL:
```
http://localhost:8000/docs
```

The interactive Swagger UI provides a complete testing interface for all endpoints.
