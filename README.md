# Resume Search API

A comprehensive FastAPI-based application for intelligent resume search and management using vector embeddings, AI-powered search, and MongoDB Atlas.

## 🚀 Features

- **Vector-based Resume Search**: Advanced similarity search using sentence transformers
- **AI-Powered Search**: Integration with multiple LLM providers (Ollama, Groq Cloud)
- **Resume Parsing**: Automated extraction and processing of resume data
- **MongoDB Atlas Integration**: Scalable document storage with search indices
- **RAG (Retrieval-Augmented Generation)**: Context-aware AI responses
- **Multiple Search Modes**: Manual search, vector search, and AI-enhanced search
- **Real-time Autocomplete**: Skills, titles, and location suggestions
- **Search History**: Track and save recent searches
- **Health Monitoring**: Comprehensive health checks and monitoring
- **Masking & Privacy**: Data protection and anonymization features

## 🏗️ Architecture

### Core Components

```
├── apis/                     # API endpoints
│   ├── vector_search.py     # Vector-based search
│   ├── manual_search.py     # Traditional search
│   ├── rag_search.py        # AI-powered search
│   ├── add_userdata.py      # User data management
│   └── healthcheck.py       # System health monitoring
├── core/                    # Core application logic
│   ├── config.py           # Configuration management
│   ├── database.py         # Database connections
│   └── custom_logger.py    # Logging system
├── mangodatabase/          # MongoDB operations
│   ├── client.py           # Database client
│   ├── operations.py       # CRUD operations
│   └── search_indexes.py   # Search index management
├── embeddings/             # Vector embedding logic
├── GroqcloudLLM/          # Groq Cloud LLM integration
├── Rag/                   # RAG implementation
└── schemas/               # Pydantic models
```

### Technology Stack

- **Backend**: FastAPI (Python 3.8+)
- **Database**: MongoDB Atlas
- **Vector Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Providers**: Ollama, Groq Cloud
- **Search**: MongoDB Atlas Search, Vector Search
- **Validation**: Pydantic
- **Logging**: Custom logging system

## 📋 Prerequisites

- Python 3.8 or higher
- MongoDB Atlas cluster (or local MongoDB with Atlas Search)
- Ollama (optional, for local LLM)
- Groq Cloud API key (optional, for cloud LLM)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd uphires_v1
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the environment template and configure:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Required: MongoDB Connection
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=cluster-name
DB_NAME=resume_db

# Optional: LLM Provider (choose one)
LLM_PROVIDER=ollama  # or groq_cloud

# If using Groq Cloud
GROQ_API_KEYS=your_groq_api_key_here

# If using Ollama
OLLAMA_API_URL=http://localhost:11434
```

### 5. Database Setup

The application will automatically create necessary collections and indexes on first run.

#### MongoDB Atlas Search Index

Create a search index in MongoDB Atlas with the following configuration:

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "combined_resume_vector": {
        "type": "knnVector",
        "dimensions": 384,
        "similarity": "cosine"
      },
      "skills": {
        "type": "string"
      },
      "experience": {
        "type": "document",
        "fields": {
          "title": {"type": "string"},
          "company": {"type": "string"}
        }
      },
      "contact_details": {
        "type": "document",
        "fields": {
          "current_city": {"type": "string"},
          "email": {"type": "string"}
        }
      }
    }
  }
}
```

## 🚀 Running the Application

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --port 8000
```

### Production Mode

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker (Optional)

```bash
# Build image
docker build -t resume-search-api .

# Run container
docker run -p 8000:8000 --env-file .env resume-search-api
```

## 📚 API Documentation

Once running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Health Check
- `GET /health/health` - Basic health check
- `GET /health/detailed` - Detailed system status
- `GET /health/database` - Database connectivity check

#### Search Operations
- `POST /api/vector_search` - Vector-based resume search
- `POST /api/manual_search` - Traditional search with filters
- `POST /api/rag_search` - AI-powered contextual search

#### Data Management
- `POST /api/add_user_data` - Add/update resume data
- `GET /api/autocomplete_skills` - Skill suggestions
- `GET /api/cities` - Location suggestions

#### Search History
- `GET /api/recent_searches` - Recent search history
- `POST /api/save_search` - Save search for later

## ⚙️ Configuration Guide

### LLM Provider Setup

#### Ollama (Local)
1. Install Ollama: https://ollama.ai/
2. Pull required models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ```
3. Set `LLM_PROVIDER=ollama` in `.env`

#### Groq Cloud
1. Get API key from https://console.groq.com/
2. Set `LLM_PROVIDER=groq_cloud` in `.env`
3. Add `GROQ_API_KEYS=your_key_here` in `.env`

### MongoDB Atlas Configuration

1. Create MongoDB Atlas cluster
2. Create database user with read/write permissions
3. Whitelist your IP address
4. Create search index (see Database Setup section)
5. Update `MONGODB_URI` in `.env`

### Vector Embeddings

The application uses `all-MiniLM-L6-v2` model for generating embeddings:
- **Dimensions**: 384
- **Language**: English
- **Performance**: Fast inference, good quality

## 🔧 Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check MongoDB URI format
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=cluster-name

# Verify network access
ping cluster.mongodb.net
```

#### 2. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve
```

#### 3. Vector Search Not Working
- Ensure Atlas Search index is created
- Check vector field name matches configuration
- Verify embeddings are generated correctly

#### 4. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

### Debugging

Enable debug logging:
```bash
LOG_LEVEL=DEBUG
ENABLE_DEBUG_LOGGING=true
```

Check logs in the `logs/` directory for detailed error information.

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health/health
```

### Basic Search
```bash
curl -X POST "http://localhost:8000/api/manual_search" \\
  -H "Content-Type: application/json" \\
  -d '{"skills": ["python", "fastapi"], "total_experience_min": 2}'
```

### Vector Search
```bash
curl -X POST "http://localhost:8000/api/vector_search" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "experienced python developer with ML skills", "k": 10}'
```

## 📈 Monitoring

### Health Endpoints
- `/health/health` - Basic status
- `/health/detailed` - Component health
- `/health/database` - DB connectivity
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe

### Logging
Logs are stored in the `logs/` directory:
- `main.log` - Application logs
- `database_manager.log` - Database operations
- `vector_search_api.log` - Search operations

## 🔒 Security Considerations

### Production Deployment
1. **Environment Variables**: Never commit `.env` files
2. **Database Access**: Use strong passwords and restrict IP access
3. **API Keys**: Rotate API keys regularly
4. **CORS**: Configure specific origins in production
5. **HTTPS**: Use SSL/TLS in production
6. **Rate Limiting**: Implement API rate limiting

### Data Privacy
- PII masking features available in `/masking` endpoints
- Audit logging for data access
- Configurable data retention policies

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review API documentation at `/docs`

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
  - Vector search implementation
  - MongoDB Atlas integration
  - Multiple LLM provider support
  - Comprehensive API documentation
