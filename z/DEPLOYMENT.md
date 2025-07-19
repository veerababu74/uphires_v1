# Deployment Guide - Resume Search API

This guide covers different deployment strategies for the Resume Search API, from development to production environments.

## Table of Contents

1. [Development Deployment](#development-deployment)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Development Deployment

### Local Development Setup

#### Prerequisites
- Python 3.8+
- Git
- MongoDB Atlas account (or local MongoDB)
- Ollama (optional, for local LLM)

#### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd uphires_v1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your configuration

# Run application
python main.py
```

#### Development Server Options

**Option 1: Direct Python execution**
```bash
python main.py
```

**Option 2: Uvicorn with reload**
```bash
uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

**Option 3: With custom configuration**
```bash
uvicorn main:app --reload --port 8000 --log-level debug
```

### Development Tools

#### Code Formatting and Linting
```bash
# Install development tools
pip install black isort flake8 mypy

# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

#### Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Production Deployment

### Option 1: Traditional Server Deployment

#### System Requirements
- Ubuntu 20.04+ / CentOS 8+ / Amazon Linux 2
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- 20GB+ storage
- SSL certificate for HTTPS

#### Step 1: Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.11 python3.11-pip python3.11-venv nginx

# Create application user
sudo useradd --system --shell /bin/bash --home-dir /opt/resume-api resume-api
sudo mkdir -p /opt/resume-api
sudo chown resume-api:resume-api /opt/resume-api
```

#### Step 2: Application Setup
```bash
# Switch to application user
sudo su - resume-api

# Clone and setup application
git clone <repository-url> /opt/resume-api/app
cd /opt/resume-api/app

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
```

#### Step 3: Configuration
```bash
# Create production environment file
cp .env.example .env

# Edit configuration for production
nano .env
```

Production `.env` example:
```bash
# Production settings
DEBUG=false
LOG_LEVEL=INFO
ENABLE_DEBUG_LOGGING=false

# MongoDB Atlas (production)
MONGODB_URI=mongodb+srv://prod_user:password@prod-cluster.mongodb.net/?retryWrites=true&w=majority

# LLM Provider (Groq for production)
LLM_PROVIDER=groq_cloud
GROQ_API_KEYS=your_production_groq_key

# Security settings
CORS_ORIGINS=["https://yourdomain.com"]
CORS_METHODS=["GET", "POST", "PUT", "DELETE"]
```

#### Step 4: Systemd Service
```bash
# Create systemd service file
sudo nano /etc/systemd/system/resume-api.service
```

Service file content:
```ini
[Unit]
Description=Resume Search API
After=network.target

[Service]
Type=notify
User=resume-api
Group=resume-api
WorkingDirectory=/opt/resume-api/app
Environment=PATH=/opt/resume-api/app/venv/bin
EnvironmentFile=/opt/resume-api/app/.env
ExecStart=/opt/resume-api/app/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 --timeout 120
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

#### Step 5: Start Service
```bash
# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable resume-api
sudo systemctl start resume-api

# Check status
sudo systemctl status resume-api
```

#### Step 6: Nginx Configuration
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/resume-api
```

Nginx configuration:
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Configuration
    ssl_certificate /path/to/your/certificate.pem;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # API Configuration
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 100M;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
        access_log off;
    }
}
```

#### Step 7: Enable Nginx Site
```bash
# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/resume-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Option 2: Cloud Platform Deployment

#### AWS Deployment with ECS

**Step 1: Create ECR Repository**
```bash
# Create ECR repository
aws ecr create-repository --repository-name resume-search-api

# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com
```

**Step 2: Build and Push Docker Image**
```bash
# Build image
docker build -t resume-search-api .

# Tag for ECR
docker tag resume-search-api:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/resume-search-api:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/resume-search-api:latest
```

**Step 3: Create ECS Task Definition**
```json
{
  "family": "resume-search-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account-id:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "resume-search-api",
      "image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/resume-search-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MONGODB_URI",
          "value": "mongodb+srv://..."
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/resume-search-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Platform Deployment

**Step 1: Build and Deploy with Cloud Run**
```bash
# Install Google Cloud SDK
# Configure authentication
gcloud auth login
gcloud config set project your-project-id

# Build and deploy
gcloud run deploy resume-search-api \\
  --source . \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated \\
  --set-env-vars MONGODB_URI="mongodb+srv://..." \\
  --memory 2Gi \\
  --cpu 2 \\
  --timeout 300 \\
  --max-instances 10
```

## Docker Deployment

### Single Container Deployment

#### Build Image
```bash
# Build the Docker image
docker build -t resume-search-api:latest .

# Run container
docker run -d \\
  --name resume-search-api \\
  -p 8000:8000 \\
  --env-file .env \\
  -v ./logs:/app/logs \\
  --restart unless-stopped \\
  resume-search-api:latest
```

### Docker Compose Deployment

#### Development Compose
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - DEBUG=true
    command: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: resume-search-api:latest
    ports:
      - "8000:8000"
    volumes:
      - logs:/app/logs
      - uploads:/app/dummy_data_save
    env_file:
      - .env.production
    environment:
      - DEBUG=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  logs:
  uploads:
```

#### Deploy with Compose
```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Production
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

## Kubernetes Deployment

### Namespace and ConfigMap
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: resume-search

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: resume-api-config
  namespace: resume-search
data:
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  HOST: "0.0.0.0"
  PORT: "8000"
```

### Secret for Sensitive Data
```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: resume-api-secrets
  namespace: resume-search
type: Opaque
stringData:
  MONGODB_URI: "mongodb+srv://user:password@cluster.mongodb.net/"
  GROQ_API_KEYS: "your-groq-api-key"
```

### Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resume-search-api
  namespace: resume-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: resume-search-api
  template:
    metadata:
      labels:
        app: resume-search-api
    spec:
      containers:
      - name: api
        image: resume-search-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: resume-api-config
        - secretRef:
            name: resume-api-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service and Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: resume-search-api-service
  namespace: resume-search
spec:
  selector:
    app: resume-search-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: resume-search-api-ingress
  namespace: resume-search
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: resume-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: resume-search-api-service
            port:
              number: 80
```

### Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n resume-search
kubectl get services -n resume-search
kubectl get ingress -n resume-search
```

## Monitoring and Maintenance

### Health Monitoring
```bash
# Check application health
curl https://yourdomain.com/health/detailed

# Check logs
sudo journalctl -u resume-api -f

# Docker logs
docker logs resume-search-api -f

# Kubernetes logs
kubectl logs -l app=resume-search-api -n resume-search -f
```

### Performance Monitoring

#### Using Prometheus and Grafana
```yaml
# monitoring.yaml
apiVersion: v1
kind: Service
metadata:
  name: prometheus
spec:
  ports:
  - port: 9090
  selector:
    app: prometheus

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus
        ports:
        - containerPort: 9090
```

### Backup Strategy

#### Database Backup
```bash
# MongoDB Atlas automatic backups
# Configure backup retention and frequency in Atlas console

# Manual backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mongodump --uri="$MONGODB_URI" --out="/backup/resume_db_$DATE"
tar -czf "/backup/resume_db_$DATE.tar.gz" "/backup/resume_db_$DATE"
rm -rf "/backup/resume_db_$DATE"
```

#### Application Backup
```bash
# Backup configuration and logs
tar -czf "/backup/app_backup_$(date +%Y%m%d).tar.gz" \\
  /opt/resume-api/app/.env \\
  /opt/resume-api/app/logs/
```

### Update and Maintenance

#### Rolling Updates
```bash
# Build new image with version tag
docker build -t resume-search-api:v1.1.0 .

# Update Kubernetes deployment
kubectl set image deployment/resume-search-api \\
  api=resume-search-api:v1.1.0 \\
  -n resume-search

# Rollback if needed
kubectl rollout undo deployment/resume-search-api -n resume-search
```

#### Database Maintenance
```bash
# Update MongoDB indexes
python -c "
from main_functions import initialize_application_startup
import asyncio
asyncio.run(initialize_application_startup())
"

# Check index usage
# Use MongoDB Compass or Atlas UI to analyze index performance
```

### Security Maintenance

#### Regular Security Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python dependencies
pip install --upgrade -r requirements.txt

# Scan for vulnerabilities
pip-audit
```

#### SSL Certificate Renewal
```bash
# For Let's Encrypt certificates
sudo certbot renew --dry-run

# For Kubernetes with cert-manager
kubectl get certificates -n resume-search
```

### Troubleshooting Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
docker stats

# Optimize Gunicorn workers
# Adjust worker count in systemd service
ExecStart=... -w 2 -k uvicorn.workers.UvicornWorker
```

#### Database Connection Issues
```bash
# Check MongoDB Atlas network access
# Verify connection string format
# Test connection manually
python -c "
from pymongo import MongoClient
client = MongoClient('$MONGODB_URI')
print(client.admin.command('ping'))
"
```

#### LLM Provider Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Test Groq API
curl -H "Authorization: Bearer $GROQ_API_KEY" \\
  https://api.groq.com/openai/v1/models
```

This deployment guide provides comprehensive coverage for different deployment scenarios, from development to production environments, ensuring reliable and scalable deployment of the Resume Search API.
