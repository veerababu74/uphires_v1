# Multi-Model Embedding Support Guide

## Overview

Your UPHires application now supports multiple embedding models with different dimensions. This guide explains how to use the new models you requested and others.

## Supported Models

### Your Requested Models

1. **nomic-ai/nomic-embed-text-v1** (768 dimensions)
   - Requires `trust_remote_code=True`
   - Good for semantic search tasks
   - Usage: Prefix queries with "search_query: " for best results

2. **intfloat/e5-small-v2** (384 dimensions)
   - Same dimensions as current default
   - Supports query/passage prefixes
   - Good balance of speed and performance

### Other Available Models

3. **BAAI/bge-large-zh-v1.5** (1024 dimensions)
   - Chinese and English support
   - High-quality embeddings

4. **intfloat/e5-base-v2** (768 dimensions)
   - Better than e5-small, larger than e5-small

5. **intfloat/e5-large-v2** (1024 dimensions)
   - Best E5 model, highest quality

## Quick Start

### 1. Test the Models

First, test the models to ensure they work:

```bash
# Test your specific models
python test_user_models.py

# Test all available models (optional)
python detect_and_configure_models.py
```

### 2. Choose Your Model

Update your `.env` file with your preferred model:

#### For nomic-ai/nomic-embed-text-v1 (768 dimensions):
```env
EMBEDDING_PROVIDER=sentence_transformer
SENTENCE_TRANSFORMER_MODEL=nomic-ai/nomic-embed-text-v1
EMBEDDING_DIMENSIONS=768
EMBEDDING_DEVICE=cpu
TRUST_REMOTE_CODE=true
```

#### For intfloat/e5-small-v2 (384 dimensions - no migration needed):
```env
EMBEDDING_PROVIDER=sentence_transformer
SENTENCE_TRANSFORMER_MODEL=intfloat/e5-small-v2
EMBEDDING_DIMENSIONS=384
EMBEDDING_DEVICE=cpu
```

#### For intfloat/e5-large-v2 (1024 dimensions):
```env
EMBEDDING_PROVIDER=sentence_transformer
SENTENCE_TRANSFORMER_MODEL=intfloat/e5-large-v2
EMBEDDING_DIMENSIONS=1024
EMBEDDING_DEVICE=cpu
```

### 3. Migration (if needed)

If you're switching to a model with different dimensions than your current setup:

#### From 384 to 768 dimensions (e.g., nomic model):
```bash
# Update the migration script for 768 dimensions
python update_embeddings_to_768.py  # You'll need to create this
```

#### From 384 to 1024 dimensions (e.g., e5-large):
```bash
# Use the existing migration script
python update_embeddings_to_1024.py
```

#### No migration needed:
- If switching from all-MiniLM-L6-v2 to e5-small-v2 (both 384 dimensions)
- Just update .env and restart your application

## Model Comparison

| Model | Dimensions | Migration Needed | Special Requirements |
|-------|------------|------------------|---------------------|
| all-MiniLM-L6-v2 (current) | 384 | - | None |
| intfloat/e5-small-v2 | 384 | No | None |
| nomic-ai/nomic-embed-text-v1 | 768 | Yes | trust_remote_code=True |
| intfloat/e5-base-v2 | 768 | Yes | None |
| BAAI/bge-large-zh-v1.5 | 1024 | Yes | None |
| intfloat/e5-large-v2 | 1024 | Yes | None |

## Usage Examples

### Basic Usage

```python
from embeddings import get_default_resume_vectorizer

# Will use whatever model is configured in .env
vectorizer = get_default_resume_vectorizer()
embeddings = vectorizer.generate_resume_embeddings(resume_data)
```

### Using Specific Models

```python
from embeddings.config import get_config_by_name
from embeddings.providers import EmbeddingProviderFactory
from embeddings import EmbeddingManager, ResumeVectorizer

# Use nomic model
config = get_config_by_name("nomic-embed-text-v1")
provider = EmbeddingProviderFactory.create_provider(
    provider_type=config.provider,
    model_name=config.model_name,
    device=config.device,
    trust_remote_code=config.trust_remote_code
)
manager = EmbeddingManager(provider)
vectorizer = ResumeVectorizer(manager)
```

### E5 Model Usage (with prefixes)

```python
# For E5 models, use query/passage prefixes for best results
query_text = "query: Python developer with machine learning experience"
passage_text = "passage: Senior Data Scientist with 5 years Python experience"

embedding_query = manager.generate_embedding(query_text)
embedding_passage = manager.generate_embedding(passage_text)
```

## Creating Migration Scripts for Different Dimensions

If you need to migrate to 768 dimensions (e.g., for nomic model), you can create a copy of the 1024 migration script:

```bash
# Copy the existing migration script
cp update_embeddings_to_1024.py update_embeddings_to_768.py

# Edit the new script to use 768 instead of 1024
# Update the search index definition
# Update the verification function
```

## Troubleshooting

### Common Issues

1. **trust_remote_code Error**
   ```
   Solution: Add TRUST_REMOTE_CODE=true to .env for nomic model
   ```

2. **Dimension Mismatch**
   ```
   Error: Search index expects 384 but model produces 768
   Solution: Run migration script or update search index
   ```

3. **Model Download Issues**
   ```
   Solution: Ensure internet connection and sufficient disk space
   Large models may take time to download initially
   ```

### Model Performance

- **Speed**: all-MiniLM-L6-v2 > e5-small-v2 > e5-base-v2 > e5-large-v2 > nomic > BAAI
- **Quality**: e5-large-v2 > BAAI > nomic > e5-base-v2 > e5-small-v2 > all-MiniLM-L6-v2
- **Memory**: Lower dimensions = less memory usage

## Recommendations

### For Your Use Case

1. **If you want minimal changes**: Switch to `intfloat/e5-small-v2`
   - Same 384 dimensions, no migration needed
   - Better quality than current model
   - Supports query/passage prefixes

2. **If you want better quality**: Use `intfloat/e5-large-v2`
   - 1024 dimensions, migration needed
   - Best performance among tested models

3. **If you need the nomic model**: Use `nomic-ai/nomic-embed-text-v1`
   - 768 dimensions, migration needed
   - Requires trust_remote_code=True

### Recommended Migration Path

1. **Test first**: `python test_user_models.py`
2. **Start simple**: Try e5-small-v2 (no migration)
3. **If satisfied**: Stay with e5-small-v2
4. **If need better quality**: Migrate to e5-large-v2 or BAAI

## Files Modified

- `embeddings/config.py` - Added new model configurations
- `embeddings/providers.py` - Added dimension detection and trust_remote_code support
- `mangodatabase/search_indexes.py` - Updated for 1024 dimensions
- `.env` - Updated default configuration

## Next Steps

1. Choose your preferred model from the table above
2. Update your `.env` file accordingly
3. Run migration script if dimensions changed
4. Test your application
5. Monitor performance and memory usage
