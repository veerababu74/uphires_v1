# Embedding Module Centralization Guide

## Overview
The embedding functionality has been centralized under the `embeddings` module to provide better organization, maintainability, and flexibility. This guide explains the changes and how to use the new structure.

## What Changed

### Before (Old Structure)
```python
# Each module created its own SentenceTransformer instance
from embeddings.vectorizer import Vectorizer, AddUserDataVectorizer

vectorizer = Vectorizer()  # Creates new SentenceTransformer instance
embedding = vectorizer.generate_embedding("some text")
```

### After (New Centralized Structure)
```python
# Centralized embedding management
from embeddings import EmbeddingManager, ResumeVectorizer, AddUserDataVectorizer

# Single embedding manager for the entire application
manager = EmbeddingManager()
vectorizer = ResumeVectorizer(manager)
embedding = vectorizer.generate_embedding("some text")

# Or use singleton instances for convenience
from embeddings import get_default_resume_vectorizer
vectorizer = get_default_resume_vectorizer()
```

## New Module Structure

```
embeddings/
├── __init__.py              # Module exports and convenience functions
├── base.py                  # Abstract base classes
├── providers.py             # Embedding provider implementations
├── manager.py               # Centralized embedding management
└── vectorizer.py            # Backward compatibility (deprecated)
```

## Key Benefits

1. **Single Model Instance**: One SentenceTransformer instance shared across the application
2. **Memory Efficiency**: Reduces memory usage by avoiding multiple model instances
3. **Provider Flexibility**: Easy to switch between different embedding providers
4. **Configuration**: Centralized configuration through environment variables
5. **Backward Compatibility**: Existing code continues to work with deprecation warnings

## Migration Guide

### For New Code (Recommended)

#### Basic Usage
```python
from embeddings import EmbeddingManager, ResumeVectorizer

# Create manager with default provider (SentenceTransformer)
manager = EmbeddingManager()
vectorizer = ResumeVectorizer(manager)

# Generate embeddings
resume_with_vectors = vectorizer.generate_resume_embeddings(resume_data)
```

#### Using Singleton Pattern
```python
from embeddings import get_default_resume_vectorizer, get_default_add_user_data_vectorizer

# Get singleton instances (recommended for most use cases)
resume_vectorizer = get_default_resume_vectorizer()
user_data_vectorizer = get_default_add_user_data_vectorizer()
```

#### Custom Provider Configuration
```python
from embeddings import EmbeddingManager, ResumeVectorizer
from embeddings.providers import EmbeddingProviderFactory

# Create custom provider
provider = EmbeddingProviderFactory.create_provider(
    provider_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

manager = EmbeddingManager(provider)
vectorizer = ResumeVectorizer(manager)
```

### For Existing Code (Backward Compatibility)

Existing code will continue to work but will show deprecation warnings:

```python
# This still works but shows deprecation warning
from embeddings.vectorizer import Vectorizer, AddUserDataVectorizer

vectorizer = Vectorizer()  # Shows deprecation warning
embedding = vectorizer.generate_embedding("text")
```

To remove warnings, update imports:

```python
# Old import
from embeddings.vectorizer import Vectorizer, AddUserDataVectorizer

# New import (recommended)
from embeddings import get_default_resume_vectorizer, get_default_add_user_data_vectorizer

# Update usage
vectorizer = get_default_resume_vectorizer()  # Instead of Vectorizer()
user_vectorizer = get_default_add_user_data_vectorizer()  # Instead of AddUserDataVectorizer()
```

## Environment Configuration

Set these environment variables in your `.env` file:

```bash
# Embedding Provider Configuration
EMBEDDING_PROVIDER=sentence_transformer
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
EMBEDDING_DEVICE=cpu

# Alternative models (uncomment to use)
# SENTENCE_TRANSFORMER_MODEL=Qwen/Qwen3-Embedding-0.6B
# EMBEDDING_DIMENSIONS=1024

# OpenAI Embeddings (if using OpenAI provider)
# EMBEDDING_PROVIDER=openai
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_API_KEY=your_api_key
```

## Files Using Embeddings

The following files have been updated to use the centralized embedding system:

### Already Updated (Using Centralized Imports)
- `Rag/rag_application.py`
- `mangodatabase/operations.py`
- `apis/autocomplete_skills_titiles.py`
- `Rag/ragapp.py`
- `Rag/embeddings.py`
- `multipleresumepraser/routes.py`
- `Retrivers/retriver.py`
- `GroqcloudLLM/routes.py`
- `core/vectorizer.py`
- `apisofmango/resume.py`
- `apis/vector_search.py`
- `apis/vectore_search_v2.py`
- `apis/resumerpaser.py`
- `apis/add_userdata.py`

### Migration Steps for Each Module

1. **Replace old imports**:
   ```python
   # Old
   from embeddings.vectorizer import Vectorizer
   
   # New
   from embeddings import get_default_resume_vectorizer
   ```

2. **Update instance creation**:
   ```python
   # Old
   vectorizer = Vectorizer()
   
   # New
   vectorizer = get_default_resume_vectorizer()
   ```

3. **API remains the same**: All existing methods work identically

## Testing the Migration

Run the existing tests to ensure backward compatibility:

```bash
python test_multi_llm.py
python test_rag_fallback.py
python test_complete_setup.py
```

## Performance Benefits

- **Memory Usage**: Reduced by ~200-500MB (depending on model size)
- **Startup Time**: Faster application startup (model loaded once)
- **Consistency**: Same model instance ensures consistent embeddings

## Future Enhancements

The centralized structure enables:
- Easy addition of new embedding providers (OpenAI, Cohere, etc.)
- Dynamic model switching
- Embedding caching
- Model warm-up strategies
- Performance monitoring

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to update import statements
2. **Model Loading**: Check CUDA availability for GPU usage
3. **Memory Issues**: Monitor memory usage with centralized instance

### Debug Information

```python
from embeddings import get_default_embedding_manager

manager = get_default_embedding_manager()
print(f"Provider: {manager.get_provider_info()}")
print(f"Embedding Dimension: {manager.get_embedding_dimension()}")
```
