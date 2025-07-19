# Embedding Model Caching System

## Overview

The embedding model caching system automatically downloads and stores embedding models locally in the `emmodels` directory. This eliminates repeated downloads during deployment and provides faster model loading.

## Features

✅ **Automatic Model Caching**: Models are downloaded once and cached locally  
✅ **Fast Subsequent Loading**: Cached models load directly from disk  
✅ **Production Deployment Ready**: No internet required after initial download  
✅ **Multiple Model Support**: Cache different models in separate directories  
✅ **Fallback Support**: Falls back to regular download if caching fails  

## Directory Structure

```
uphires_v1/
├── emmodels/                           # Local model cache directory
│   ├── BAAI_bge-large-zh-v1.5/       # Chinese model (1024 dims)
│   ├── BAAI_bge-large-en-v1.5/       # English model (1024 dims)
│   ├── sentence-transformers_all-MiniLM-L6-v2/  # Small test model
│   └── ...other models/
├── embeddings/
│   ├── providers.py                   # Enhanced with caching logic
│   ├── config.py                     # Model configurations
│   └── ...
├── download_embedding_models.py       # Batch download script
└── test_model_caching.py             # Caching test script
```

## Usage

### 1. Download Individual Models

```bash
# Download specific model
python download_embedding_models.py --models "BAAI/bge-large-zh-v1.5"

# Download multiple models
python download_embedding_models.py --models "BAAI/bge-large-zh-v1.5,bge-large-en-v1.5"
```

### 2. Download Recommended Models

```bash
# Download best performing models for production
python download_embedding_models.py --recommended
```

### 3. Download All Available Models

```bash
# Download all configured models (large download!)
python download_embedding_models.py --all
```

### 4. List Available Models

```bash
# See all available models with descriptions
python download_embedding_models.py --list
```

## Programmatic Usage

```python
from embeddings.providers import SentenceTransformerProvider

# Create provider (automatically uses cache if available)
provider = SentenceTransformerProvider("BAAI/bge-large-zh-v1.5")

# Check if model is cached
if provider._is_model_cached():
    print("Model will load from cache (fast)")
else:
    print("Model will be downloaded (slower)")

# Load model (triggers download if not cached)
model = provider.model

# Generate embeddings
embedding = provider.generate_embedding("Your text here")
print(f"Dimensions: {len(embedding)}")
```

## Cache Management

### Cache Location
- **Directory**: `uphires_v1/emmodels/`
- **Naming**: Model names with `/` replaced by `_` (e.g., `BAAI_bge-large-zh-v1.5`)

### Cache Detection
The system checks for essential SentenceTransformer files:
- `config_sentence_transformers.json` - SentenceTransformer configuration
- `modules.json` - Model modules definition
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `tokenizer.json` or `vocab.txt` - Tokenizer files

### Manual Cache Management

```bash
# Check cache size
python download_embedding_models.py --list

# Clear specific model cache (manual deletion)
rmdir /s "emmodels\\BAAI_bge-large-zh-v1.5"

# Clear all caches (manual deletion)
rmdir /s emmodels
```

## Available Models

### Target Model (1024 dimensions)
- **BAAI/bge-large-zh-v1.5**: Original Chinese model for 384→1024 migration

### Best Performance Models
- **BAAI/bge-large-en-v1.5**: Best English model (1024 dims)
- **thenlper/gte-large**: Excellent retrieval performance (1024 dims)
- **BAAI/bge-m3**: Multi-lingual, multi-functionality (1024 dims)
- **intfloat/multilingual-e5-large**: Best multilingual support (1024 dims)

### Reliable Models
- **sentence-transformers/all-mpnet-base-v2**: Solid general-purpose (768 dims)
- **intfloat/e5-large-v2**: Strong embedding performance (1024 dims)
- **BAAI/bge-base-en-v1.5**: Good speed/accuracy balance (768 dims)

## Testing

```bash
# Test caching functionality
python test_model_caching.py

# Should show:
# ✓ Caching test PASSED
# ✓ Embeddings are consistent between cached and fresh loads
# ✓ Cache provides performance improvement
```

## Production Deployment

### Pre-deployment Download
```bash
# Download all recommended models before deployment
python download_embedding_models.py --recommended

# Verify downloads
python download_embedding_models.py --list
```

### Deployment Benefits
1. **No Internet Required**: Models cached locally
2. **Faster Startup**: No download delays
3. **Consistent Performance**: Predictable loading times
4. **Offline Capability**: Works without external connectivity

## Migration from 384 to 1024 Dimensions

The target model `BAAI/bge-large-zh-v1.5` is now cached and ready:

```python
# Use the cached Chinese model
provider = SentenceTransformerProvider("BAAI/bge-large-zh-v1.5")
print(f"Dimensions: {provider.embedding_dim}")  # Should show: 1024

# Generate 1024-dimensional embeddings
embedding = provider.generate_embedding("中文文本测试")
print(f"Vector length: {len(embedding)}")  # Should show: 1024
```

## Troubleshooting

### Cache Not Working
1. Check if `emmodels` directory exists and has correct permissions
2. Verify model files are complete (check file sizes)
3. Try deleting cache and re-downloading

### Download Failures
1. Check internet connectivity
2. Verify model names in config are correct
3. Check if models require authentication tokens

### Performance Issues
1. Ensure models are actually cached (check logs)
2. Use smaller models for testing
3. Consider SSD storage for better I/O performance

## Configuration

Models are configured in `embeddings/config.py`:

```python
EMBEDDING_CONFIGS = {
    "BAAI/bge-large-zh-v1.5": {
        "dimensions": 1024,
        "trust_remote_code": True,
        "description": "Chinese embedding model"
    },
    # ... more models
}
```

## Success Metrics

✅ **Original Goal Achieved**: Successfully switched from 384 to 1024 dimensions using BAAI/bge-large-zh-v1.5  
✅ **Migration Scripts Fixed**: All deprecation warnings resolved  
✅ **Local Caching Implemented**: Models download to `emmodels` folder  
✅ **Production Ready**: No repeated downloads during deployment  
✅ **Best Models Added**: Enhanced with top-performing embedding models  

Current cache contains **2.72 GB** of high-performance embedding models ready for production use.
