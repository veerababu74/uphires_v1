# Setup and Configuration Guide

This comprehensive guide will walk you through setting up the Resume Search API from scratch, including all dependencies and configuration options.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [MongoDB Atlas Configuration](#mongodb-atlas-configuration)
4. [LLM Provider Setup](#llm-provider-setup)
5. [Application Configuration](#application-configuration)
6. [Database Initialization](#database-initialization)
7. [Testing Setup](#testing-setup)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet connection for MongoDB Atlas and LLM providers

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16GB (for local LLM with Ollama)
- **CPU**: Multi-core processor
- **Storage**: SSD with 10GB free space

## Environment Setup

### 1. Python Installation

#### Windows
```powershell
# Download from python.org or use winget
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Using pyenv (recommended)
brew install pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv
```

### 2. Clone Repository
```bash
git clone <repository-url>
cd uphires_v1
```

### 3. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
which python
```

### 4. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
pip list | grep fastapi
pip list | grep pymongo
```

## MongoDB Atlas Configuration

### 1. Create MongoDB Atlas Account
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up for a free account
3. Create a new project

### 2. Create Cluster
1. Click "Create a Cluster"
2. Choose "Shared" for free tier
3. Select cloud provider and region
4. Choose cluster tier (M0 for free)
5. Name your cluster (e.g., "resume-search-cluster")
6. Click "Create Cluster"

### 3. Configure Database Access
1. Go to "Database Access" in the sidebar
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Create username and strong password
5. Set permissions to "Read and write to any database"
6. Click "Add User"

### 4. Configure Network Access
1. Go to "Network Access" in the sidebar
2. Click "Add IP Address"
3. For development: Click "Allow Access from Anywhere"
4. For production: Add specific IP addresses
5. Click "Confirm"

### 5. Get Connection String
1. Go to "Clusters" and click "Connect"
2. Choose "Connect your application"
3. Select "Python" and version "3.6 or later"
4. Copy the connection string
5. Replace `<password>` with your database user password

Example connection string:
```
mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=cluster-name
```

### 6. Create Search Index

#### Using MongoDB Atlas UI
1. Go to your cluster and click "Collections"
2. Click "Create Database" if needed:
   - Database name: `resume_db`
   - Collection name: `resumes`
3. Go to "Atlas Search" tab
4. Click "Create Search Index"
5. Choose "Visual Editor"
6. Select database `resume_db` and collection `resumes`
7. Use the following index definition:

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
        "type": "string",
        "analyzer": "lucene.standard"
      },
      "may_also_known_skills": {
        "type": "string",
        "analyzer": "lucene.standard"
      },
      "experience": {
        "type": "document",
        "fields": {
          "title": {
            "type": "string",
            "analyzer": "lucene.standard"
          },
          "company": {
            "type": "string",
            "analyzer": "lucene.standard"
          },
          "description": {
            "type": "string",
            "analyzer": "lucene.standard"
          }
        }
      },
      "contact_details": {
        "type": "document",
        "fields": {
          "current_city": {
            "type": "string"
          },
          "email": {
            "type": "string"
          },
          "name": {
            "type": "string",
            "analyzer": "lucene.standard"
          }
        }
      },
      "total_experience": {
        "type": "number"
      },
      "current_salary": {
        "type": "number"
      },
      "expected_salary": {
        "type": "number"
      },
      "notice_period": {
        "type": "number"
      }
    }
  }
}
```

8. Name the index: `vector_search_index`
9. Click "Next" and then "Create Search Index"

#### Using MongoDB Compass (Alternative)
1. Download and install MongoDB Compass
2. Connect using the connection string
3. Navigate to `resume_db.resumes` collection
4. Go to "Indexes" tab
5. Create the search index manually

## LLM Provider Setup

### Option 1: Ollama (Local LLM - Recommended for Development)

#### Installation

**Windows:**
1. Download from [Ollama.ai](https://ollama.ai/)
2. Run the installer
3. Open Command Prompt or PowerShell

**macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from ollama.ai
curl https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

#### Pull Required Models
```bash
# Primary model (3B parameters, fast)
ollama pull llama3.2:3b

# Backup models
ollama pull qwen2.5:3b
ollama pull qwen:4b

# Verify models
ollama list
```

#### Start Ollama Service
```bash
# Start Ollama (runs on http://localhost:11434)
ollama serve

# Test connection
curl http://localhost:11434/api/version
```

#### Configuration for Ollama
In your `.env` file:
```bash
LLM_PROVIDER=ollama
OLLAMA_API_URL=http://localhost:11434
OLLAMA_PRIMARY_MODEL=llama3.2:3b
OLLAMA_BACKUP_MODEL=qwen2.5:3b
OLLAMA_FALLBACK_MODEL=qwen:4b
```

### Option 2: Groq Cloud (Cloud LLM - Recommended for Production)

#### Get API Key
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up for an account
3. Go to "API Keys" section
4. Create a new API key
5. Copy the API key

#### Configuration for Groq Cloud
In your `.env` file:
```bash
LLM_PROVIDER=groq_cloud
GROQ_API_KEYS=your_api_key_here,your_backup_key_here
GROQ_PRIMARY_MODEL=gemma2-9b-it
GROQ_BACKUP_MODEL=llama-3.1-70b-versatile
GROQ_FALLBACK_MODEL=mixtral-8x7b-32768
```

## Application Configuration

### 1. Environment File Setup

Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Basic Configuration

Edit `.env` with your values:

```bash
# ==========================================
# REQUIRED CONFIGURATION
# ==========================================

# MongoDB Connection (REQUIRED)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=cluster-name
DB_NAME=resume_db
COLLECTION_NAME=resumes

# LLM Provider (REQUIRED - choose one)
LLM_PROVIDER=ollama  # or groq_cloud

# ==========================================
# OLLAMA CONFIGURATION (if using Ollama)
# ==========================================
OLLAMA_API_URL=http://localhost:11434
OLLAMA_PRIMARY_MODEL=llama3.2:3b
OLLAMA_BACKUP_MODEL=qwen2.5:3b

# ==========================================
# GROQ CLOUD CONFIGURATION (if using Groq)
# ==========================================
GROQ_API_KEYS=your_groq_api_key_here

# ==========================================
# OPTIONAL CONFIGURATION
# ==========================================

# Application Settings
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Vector Search
MODEL_NAME=all-MiniLM-L6-v2
DIMENSIONS=384
VECTOR_FIELD=combined_resume_vector

# Atlas Search
ENABLE_ATLAS_SEARCH=true
ATLAS_SEARCH_INDEX=vector_search_index

# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG_LOGGING=true
```

### 3. Advanced Configuration Options

#### Performance Tuning
```bash
# MongoDB Settings
DEFAULT_MONGODB_LIMIT=50
DEFAULT_MAX_RESULTS=20

# LLM Settings
MAX_CONTEXT_LENGTH=8000
OLLAMA_TEMPERATURE=0.1
GROQ_TEMPERATURE=0.1

# RAG Settings
RAG_RETRIEVAL_K=10
RAG_TEMPERATURE=0.0
RAG_MAX_RETRIES=3
```

#### Security Settings
```bash
# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
CORS_METHODS=["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS=["*"]

# For production, set DEBUG=false
DEBUG=false
ENABLE_DEBUG_LOGGING=false
```

## Database Initialization

### 1. Automatic Initialization
The application automatically creates collections and indexes on first run:

```bash
# Start the application
python main.py

# Check logs for initialization status
tail -f logs/main.log
```

### 2. Manual Database Setup (Optional)

If you need to set up the database manually:

```python
# Create a script: setup_database.py
from mangodatabase.client import get_collection
from main_functions import initialize_application_startup

async def setup():
    try:
        search_manager, success = await initialize_application_startup()
        if success:
            print("Database setup completed successfully!")
        else:
            print("Database setup failed!")
    except Exception as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(setup())
```

### 3. Verify Database Setup

```python
# test_database.py
from mangodatabase.client import get_collection
from pymongo.errors import ServerSelectionTimeoutError

def test_connection():
    try:
        collection = get_collection()
        # Test connection
        collection.database.client.admin.command("ping")
        print("✅ Database connection successful!")
        
        # Check collections
        collections = collection.database.list_collection_names()
        print(f"📋 Available collections: {collections}")
        
        # Check indexes
        indexes = list(collection.list_indexes())
        print(f"🗂️ Indexes created: {len(indexes)}")
        
        return True
    except ServerSelectionTimeoutError:
        print("❌ Database connection failed!")
        return False
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

## Testing Setup

### 1. Basic Health Check
```bash
# Start the application
python main.py

# In another terminal, test health endpoint
curl http://localhost:8000/health/health
```

### 2. Test Database Connection
```bash
curl http://localhost:8000/health/database
```

### 3. Test LLM Provider
```bash
# Test Ollama
curl http://localhost:11434/api/version

# Test application LLM endpoint
curl -X GET "http://localhost:8000/api/llm_config"
```

### 4. Test Vector Search
```bash
curl -X POST "http://localhost:8000/api/vector_search" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "python developer",
    "k": 5
  }'
```

### 5. Load Test Data (Optional)

Create a test script to add sample data:

```python
# add_test_data.py
import requests
import json

def add_sample_resume():
    url = "http://localhost:8000/api/add_user_data"
    data = {
        "user_id": "test_user_001",
        "contact_details": {
            "name": "Test Developer",
            "email": "test@example.com",
            "phone_number": "+91-9876543210",
            "current_city": "Bangalore"
        },
        "skills": ["Python", "FastAPI", "MongoDB"],
        "total_experience": 5,
        "current_salary": 80000,
        "expected_salary": 95000,
        "notice_period": 30
    }
    
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    add_sample_resume()
```

## Production Deployment

### 1. Environment Preparation

#### Production .env file:
```bash
# Production environment settings
DEBUG=false
ENABLE_DEBUG_LOGGING=false
LOG_LEVEL=WARNING

# Security settings
CORS_ORIGINS=["https://yourdomain.com"]

# Performance settings
DEFAULT_MONGODB_LIMIT=20
DEFAULT_MAX_RESULTS=10
```

### 2. Using Gunicorn

Install Gunicorn:
```bash
pip install gunicorn
```

Run with Gunicorn:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

Build and run:
```bash
docker build -t resume-search-api .
docker run -p 8000:8000 --env-file .env resume-search-api
```

### 4. Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with Docker Compose:
```bash
docker-compose up -d
```

### 5. Reverse Proxy with Nginx

Nginx configuration:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Module not found" errors
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. MongoDB connection fails
```bash
# Check connection string format
# Ensure IP whitelist includes your IP
# Verify username/password

# Test connection manually
python -c "
from pymongo import MongoClient
client = MongoClient('your_connection_string')
client.admin.command('ping')
print('Connected successfully!')
"
```

#### 3. Ollama connection fails
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Check available models
ollama list
```

#### 4. Vector search not working
```bash
# Ensure Atlas Search index is created and active
# Check index name matches configuration
# Verify vector field exists in documents
```

#### 5. Import errors with sentence-transformers
```bash
# Update PyTorch and transformers
pip install --upgrade torch transformers sentence-transformers

# For M1/M2 Macs
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```

#### 6. Memory issues with local LLM
```bash
# Use smaller models
ollama pull llama3.2:1b  # Instead of 3b

# Adjust Ollama settings
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_PARALLEL=1
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_DEBUG_LOGGING=true

# Check logs
tail -f logs/main.log
tail -f logs/vector_search_api.log
```

### Performance Monitoring

Monitor application performance:

```bash
# Check health endpoints
curl http://localhost:8000/health/detailed

# Monitor system resources
top  # Linux/Mac
Get-Process python | Select-Object * | Format-Table  # Windows PowerShell
```

### Getting Help

If you encounter issues:

1. Check the logs in the `logs/` directory
2. Verify your `.env` configuration
3. Test individual components (database, LLM, embeddings)
4. Review the troubleshooting section above
5. Create an issue in the repository with:
   - Error message
   - Configuration (without sensitive data)
   - Steps to reproduce
   - System information

## Next Steps

After successful setup:

1. **Read the API Documentation**: Review `API_DOCUMENTATION.md`
2. **Test All Endpoints**: Use the Swagger UI at `/docs`
3. **Load Sample Data**: Add test resumes through the API
4. **Monitor Performance**: Set up logging and monitoring
5. **Security Review**: Implement authentication and rate limiting for production
6. **Backup Strategy**: Set up regular database backups
