# Project Architecture - Resume Search API

## Overview

The Resume Search API is a sophisticated, scalable application built with FastAPI that provides intelligent resume search capabilities using vector embeddings, traditional search methods, and AI-powered contextual search.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
│  (Web Apps, Mobile Apps, CLI Tools, Third-party Integrations)   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ HTTP/HTTPS Requests
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                     Nginx (Reverse Proxy)                      │
│            (Load Balancing, SSL Termination, CORS)             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ HTTP Requests
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                    FastAPI Application                         │
│                     (main.py + APIs)                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Health    │ │   Vector    │ │   Manual    │ │     RAG     │ │
│  │   Check     │ │   Search    │ │   Search    │ │   Search    │ │
│  │    APIs     │ │    APIs     │ │    APIs     │ │    APIs     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   User      │ │ Autocomplete│ │   Masking   │ │    LLM      │ │
│  │   Data      │ │     APIs    │ │    APIs     │ │   Config    │ │
│  │    APIs     │ │             │ │             │ │    APIs     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ Database & Service Calls
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                     Core Services Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Database   │ │ Vectorizer  │ │  LLM Core   │ │   Logger    │ │
│  │  Manager    │ │  Service    │ │  Services   │ │  Service    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Config    │ │ Exception   │ │   Helper    │ │    Cache    │ │
│  │  Manager    │ │  Handler    │ │   Utils     │ │  Service    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ External Service Calls
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                   External Services                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  MongoDB    │ │   Ollama    │ │ Groq Cloud  │ │ Embeddings  │ │
│  │   Atlas     │ │   Local     │ │    LLM      │ │   Models    │ │
│  │             │ │    LLM      │ │             │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Application Layer (`main.py`)

The main FastAPI application that orchestrates all components:

**Responsibilities:**
- Application lifecycle management
- Router registration and organization
- Middleware configuration (CORS, exception handling)
- Global exception handling
- Health monitoring integration

**Key Features:**
- Async context manager for startup/shutdown
- Comprehensive error handling
- Static file serving
- API documentation generation

### 2. API Layer (`apis/`)

RESTful API endpoints organized by functionality:

#### Search APIs
- **`vector_search.py`**: Vector similarity search using embeddings
- **`vectore_search_v2.py`**: Enhanced vector search with improved algorithms
- **`manual_search.py`**: Traditional filtering and query-based search
- **`rag_search.py`**: AI-powered contextual search using RAG

#### Data Management APIs
- **`add_userdata.py`**: Resume data ingestion and management
- **`resumerpaser.py`**: Resume parsing and text extraction
- **`autocomplete_skills_titiles.py`**: Real-time autocomplete suggestions
- **`citys.py`**: Location-based search and suggestions

#### Utility APIs
- **`healthcheck.py`**: System health monitoring and diagnostics
- **`llm_config_api.py`**: LLM provider configuration management
- **`retriever_api.py`**: Document retrieval services

#### History and Persistence
- **`manual_recent_search_save.py`**: Search history management
- **`ai_recent_saved_searchs.py`**: AI search history tracking

### 3. Core Services Layer (`core/`)

Foundational services and utilities:

#### Configuration Management (`config.py`)
- Centralized configuration using environment variables
- Provider-agnostic LLM configuration
- Database connection management
- Feature flag management

#### Custom Logging (`custom_logger.py`)
- Structured logging with rotation
- Component-specific log files
- Configurable log levels
- Production-ready logging

#### Database Abstraction (`database.py`)
- MongoDB connection pooling
- Connection health monitoring
- Error handling and retries

#### Utilities
- **`helpers.py`**: Common utility functions
- **`exceptions.py`**: Custom exception classes
- **`llm_factory.py`**: LLM provider factory pattern

### 4. Database Layer (`mangodatabase/`)

MongoDB operations and management:

#### Client Management (`client.py`)
- Connection management
- Collection getters
- Connection pooling

#### Operations (`operations.py`)
- CRUD operations
- Bulk operations
- Transaction management

#### Search Index Management (`search_indexes.py`)
- Atlas Search index creation and management
- Vector index configuration
- Index health monitoring

### 5. AI/ML Layer

#### Embeddings (`embeddings/`)
- **`vectorizer.py`**: Text to vector conversion
- Sentence transformer integration
- Embedding caching and optimization

#### LLM Integration
- **`GroqcloudLLM/`**: Groq Cloud API integration
- **`Rag/`**: Retrieval-Augmented Generation implementation
- **`core/llm_factory.py`**: Provider abstraction

#### Text Processing
- **`textextractors/`**: Document text extraction
- **`multipleresumepraser/`**: Batch resume processing

### 6. Data Models (`schemas/`)

Pydantic models for request/response validation:
- **`vector_search_scehma.py`**: Vector search models
- **`add_user_schemas.py`**: User data models
- **`search_index_api_schemas.py`**: Search index models

## Data Flow

### 1. Resume Ingestion Flow

```
User Upload → Text Extraction → Data Validation → Vector Generation → Database Storage → Index Update
     ↓              ↓              ↓               ↓                 ↓               ↓
   File/Data    Clean Text    Validated Data   384-dim Vector   MongoDB Doc    Search Index
```

### 2. Vector Search Flow

```
User Query → Query Vectorization → Vector Similarity Search → Result Ranking → Response Formatting
     ↓              ↓                      ↓                      ↓                ↓
Text Query     384-dim Vector         Atlas Vector Search    Sorted Results    JSON Response
```

### 3. AI Search Flow

```
User Query → Context Retrieval → LLM Processing → Response Generation → Result Formatting
     ↓              ↓               ↓               ↓                  ↓
Natural Language  Relevant Docs   AI Processing   Generated Text   JSON Response
```

## Database Design

### Collections

#### `resumes` Collection
```javascript
{
  "_id": ObjectId,
  "user_id": "unique_user_identifier",
  "contact_details": {
    "name": "Full Name",
    "email": "email@domain.com",
    "phone_number": "+1-234-567-8900",
    "current_city": "City Name",
    "pan_card": "PAN123456",
    "addhar_number": "123456789012"
  },
  "skills": ["Python", "JavaScript", "MongoDB"],
  "may_also_known_skills": ["Docker", "AWS"],
  "total_experience": 5,
  "current_salary": 80000,
  "expected_salary": 95000,
  "notice_period": 30,
  "experience": [
    {
      "title": "Software Engineer",
      "company": "Tech Corp",
      "duration": "2 years",
      "description": "Developed web applications"
    }
  ],
  "education": [
    {
      "degree": "B.Tech Computer Science",
      "institution": "University Name",
      "year": "2020"
    }
  ],
  "combined_resume_vector": [0.1, 0.2, ...], // 384 dimensions
  "created_at": ISODate,
  "updated_at": ISODate
}
```

#### `skills_titles` Collection
```javascript
{
  "_id": ObjectId,
  "type": "skill" | "title" | "city",
  "value": "Python" | "Software Engineer" | "Bangalore",
  "frequency": 1500,
  "created_at": ISODate
}
```

#### Search History Collections
- `ai_recent_search`: AI search history
- `ai_saved_searches`: Saved AI searches
- `manual_recent_search`: Manual search history
- `manual_saved_searches`: Saved manual searches

### Indexes

#### Primary Indexes
- User identification: `user_id`, `contact_details.email`, `contact_details.pan_card`
- Search optimization: `skills`, `total_experience`, `current_salary`
- Location: `contact_details.current_city`

#### Compound Indexes
- Search combinations: `(skills, total_experience)`, `(current_city, expected_salary)`
- User lookup: `(pan_card, email)`, `(phone_number, email)`

#### Vector Index
- Atlas Vector Search index on `combined_resume_vector` field
- 384 dimensions, cosine similarity

## Security Architecture

### 1. Data Protection
- PII masking capabilities
- Configurable data anonymization
- Audit logging for data access

### 2. API Security
- Input validation using Pydantic
- SQL injection prevention
- Rate limiting (configurable)

### 3. Infrastructure Security
- Environment variable configuration
- Secure connection strings
- Production security headers

## Scalability Considerations

### 1. Horizontal Scaling
- Stateless application design
- Database connection pooling
- Load balancer ready

### 2. Performance Optimization
- Efficient database indexing
- Vector search optimization
- Caching strategies (Redis integration ready)

### 3. Resource Management
- Configurable batch sizes
- Memory-efficient processing
- Graceful degradation

## Monitoring and Observability

### 1. Health Checks
- Application health endpoints
- Database connectivity monitoring
- External service health checks

### 2. Logging
- Structured logging with correlation IDs
- Component-specific log files
- Error tracking and alerting

### 3. Metrics
- Response time monitoring
- Error rate tracking
- Resource utilization metrics

## Deployment Architecture

### 1. Development Environment
```
Local Machine → Python Virtual Environment → Local MongoDB → Local LLM (Ollama)
```

### 2. Production Environment
```
Load Balancer → Multiple App Instances → MongoDB Atlas → Groq Cloud LLM
```

### 3. Container Deployment
```
Docker Containers → Kubernetes Pods → Managed Services → External APIs
```

## Configuration Management

### 1. Environment-based Configuration
- Development, staging, production environments
- Feature flags and toggles
- Provider-specific configurations

### 2. Runtime Configuration
- LLM provider switching
- Model selection
- Performance tuning parameters

## Error Handling Strategy

### 1. Graceful Degradation
- Fallback LLM models
- Alternative search methods
- Partial service availability

### 2. Error Recovery
- Automatic retries with exponential backoff
- Circuit breaker patterns
- Health check based recovery

## Future Architecture Considerations

### 1. Microservices Migration
- Service decomposition strategy
- API gateway implementation
- Inter-service communication

### 2. Event-Driven Architecture
- Asynchronous processing
- Event sourcing
- Message queues

### 3. AI/ML Pipeline Enhancement
- Model versioning
- A/B testing framework
- Continuous learning system

This architecture provides a solid foundation for a scalable, maintainable, and extensible resume search system that can grow with business requirements while maintaining performance and reliability.
