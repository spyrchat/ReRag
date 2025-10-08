# Embedding Module

Production-ready embedding generation with multiple providers, caching, and hybrid dense+sparse support.

## 📋 Overview

The embedding module provides a unified interface for generating vector embeddings from text using various providers. It supports:

- **Multiple Providers**: Google, OpenAI, Voyage, HuggingFace, Bedrock
- **Hybrid Embeddings**: Dense semantic + sparse keyword vectors
- **Intelligent Caching**: Persistent caching to avoid re-computation
- **Batch Processing**: Efficient handling of large document sets
- **Production Ready**: Error handling, rate limiting, monitoring

## 🏗️ Architecture

```
embedding/
├── __init__.py
├── base_embedder.py        # Abstract interfaces
├── factory.py             # Provider factory
├── bedrock_embeddings.py  # AWS Bedrock implementation
├── hf_embedder.py         # HuggingFace implementation
├── processor.py           # Text preprocessing
├── recursive_splitter.py  # Advanced text splitting
├── splitter.py           # Basic text splitting
└── utils.py              # Utility functions
```

### Provider Support

| Provider | Dense | Sparse | API Key Required | Notes |
|----------|-------|--------|------------------|--------|
| **Google** | ✅ | ❌ | `GOOGLE_API_KEY` | text-embedding-004 |
| **OpenAI** | ✅ | ❌ | `OPENAI_API_KEY` | text-embedding-3-large |
| **Voyage** | ✅ | ❌ | `VOYAGE_API_KEY` | voyage-large-2 |
| **HuggingFace** | ✅ | ✅ | Optional | Local/remote models |
| **Bedrock** | ✅ | ❌ | AWS credentials | titan-embed-text-v1 |
| **SPLADE** | ❌ | ✅ | No | Sparse embeddings only |

## 🚀 Quick Start

### Basic Usage

```python
from embedding.factory import get_embedder

# Dense embeddings (semantic similarity)
dense_embedder = get_embedder(
    provider="google",
    model="text-embedding-004",
    api_key="your-api-key"
)

# Generate embeddings
texts = ["Solar energy is renewable", "Wind power generates electricity"]
embeddings = dense_embedder.embed_documents(texts)
print(f"Shape: {len(embeddings)}x{len(embeddings[0])}")  # 2x768
```

### Hybrid Embeddings

```python
from embedding.factory import get_embedder

# Dense embedder for semantic similarity
dense_embedder = get_embedder(
    provider="google",
    model="text-embedding-004"
)

# Sparse embedder for keyword matching
sparse_embedder = get_embedder(
    provider="sparse-splade",
    model="prithivida/Splade_PP_en_v1"
)

# Use both in retrieval pipeline
documents = ["Text about renewable energy..."]
dense_vectors = dense_embedder.embed_documents(documents)
sparse_vectors = sparse_embedder.embed_documents(documents)
```

### With Configuration

```python
config = {
    "provider": "google",
    "model": "text-embedding-004",
    "api_key_env": "GOOGLE_API_KEY",
    "batch_size": 32,
    "dimensions": 768
}

embedder = get_embedder(**config)
```

## ⚙️ Provider Configuration

### Google AI

```python
# Google text-embedding-004
embedder = get_embedder(
    provider="google",
    model="text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY"),
    dimensions=768,
    batch_size=32
)
```

Environment setup:
```bash
export GOOGLE_API_KEY=your_google_api_key
```

### OpenAI

```python
# OpenAI text-embedding-3-large
embedder = get_embedder(
    provider="openai", 
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=3072,
    batch_size=16
)
```

### HuggingFace

```python
# Local HuggingFace model
embedder = get_embedder(
    provider="hf",
    model="BAAI/bge-large-en-v1.5",
    device="cuda",  # or "cpu"
    normalize_embeddings=True
)

# Remote HuggingFace Inference API
embedder = get_embedder(
    provider="hf",
    model="sentence-transformers/all-MiniLM-L6-v2",
    api_key=os.getenv("HF_API_KEY"),
    use_api=True
)
```

### Voyage AI

```python
embedder = get_embedder(
    provider="voyage",
    model="voyage-large-2",
    api_key=os.getenv("VOYAGE_API_KEY")
)
```

### AWS Bedrock

```python
embedder = get_embedder(
    provider="bedrock",
    model="amazon.titan-embed-text-v1",
    region="us-east-1"
)
# Requires AWS credentials configured
```

### Sparse/SPLADE

```python
# For sparse keyword embeddings
sparse_embedder = get_embedder(
    provider="sparse-splade",
    model="prithivida/Splade_PP_en_v1",
    device="cuda"
)
```

## 🔧 Advanced Features

### Caching

```python
# Enable persistent caching
embedder = get_embedder(
    provider="google",
    model="text-embedding-004",
    cache_dir="cache/embeddings/",
    cache_enabled=True
)

# First call - generates and caches
embeddings = embedder.embed_documents(["Text to embed"])

# Second call - loads from cache (much faster)
embeddings = embedder.embed_documents(["Text to embed"])
```

### Batch Processing

```python
# Large dataset processing
large_texts = ["Document " + str(i) for i in range(10000)]

embedder = get_embedder(
    provider="google",
    batch_size=64,  # Process 64 documents at once
    rate_limit_delay=0.1  # Small delay between batches
)

embeddings = embedder.embed_documents(large_texts)
# Automatically handles batching and rate limiting
```

### Text Preprocessing

```python
from embedding.processor import TextProcessor

processor = TextProcessor(
    max_length=512,
    clean_html=True,
    normalize_whitespace=True,
    remove_special_chars=False
)

# Preprocess before embedding
processed_texts = processor.process_texts(raw_texts)
embeddings = embedder.embed_documents(processed_texts)
```

### Chunking Integration

```python
from embedding.recursive_splitter import RecursiveCharacterTextSplitter

# Advanced chunking for long documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Split document and embed chunks
document = "Very long document text..."
chunks = splitter.split_text(document)
embeddings = embedder.embed_documents(chunks)
```

## 📊 Monitoring & Performance

### Performance Metrics

```python
import time

# Timing embeddings
start_time = time.time()
embeddings = embedder.embed_documents(texts)
duration = time.time() - start_time

print(f"Embedded {len(texts)} docs in {duration:.2f}s")
print(f"Rate: {len(texts)/duration:.1f} docs/sec")
```

### Error Handling

```python
from embedding.factory import get_embedder
import logging

logging.basicConfig(level=logging.INFO)

try:
    embedder = get_embedder(
        provider="google",
        model="text-embedding-004",
        api_key="invalid-key",
        retry_count=3,
        retry_delay=1.0
    )
    embeddings = embedder.embed_documents(["test"])
except Exception as e:
    logging.error(f"Embedding failed: {e}")
```

### Memory Management

```python
# For large datasets, process in chunks
def embed_large_dataset(texts, embedder, chunk_size=1000):
    all_embeddings = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        chunk_embeddings = embedder.embed_documents(chunk)
        all_embeddings.extend(chunk_embeddings)
        
        # Optional: garbage collection
        import gc
        gc.collect()
    
    return all_embeddings
```

## 🔌 Extension Points

### Adding New Providers

1. **Implement Base Interface**
   ```python
   from embedding.base_embedder import BaseEmbedder
   
   class MyCustomEmbedder(BaseEmbedder):
       def __init__(self, model: str, api_key: str, **kwargs):
           self.model = model
           self.api_key = api_key
       
       def embed_documents(self, texts: List[str]) -> List[List[float]]:
           # Your implementation here
           return embeddings
       
       def embed_query(self, text: str) -> List[float]:
           return self.embed_documents([text])[0]
   ```

2. **Register in Factory**
   ```python
   # embedding/factory.py
   from .my_custom_embedder import MyCustomEmbedder
   
   EMBEDDER_REGISTRY["my_provider"] = MyCustomEmbedder
   ```

3. **Use Your Provider**
   ```python
   embedder = get_embedder(
       provider="my_provider",
       model="my-model",
       api_key="my-key"
   )
   ```

### Custom Text Processing

```python
from embedding.processor import TextProcessor

class MyCustomProcessor(TextProcessor):
    def preprocess_text(self, text: str) -> str:
        # Custom preprocessing logic
        text = super().preprocess_text(text)
        text = self.custom_cleaning(text)
        return text
    
    def custom_cleaning(self, text: str) -> str:
        # Your custom logic here
        return text
```

## 🧪 Testing

### Unit Tests

```bash
# Test embedding functionality
pytest tests/unit/test_embedding.py -v

# Test specific provider
pytest tests/unit/test_embedding.py::test_google_embedder -v
```

### Integration Tests

```bash
# Test with real APIs (requires keys)
export GOOGLE_API_KEY=your_key
pytest tests/integration/test_embedding_integration.py -v
```

### Performance Tests

```bash
# Benchmark different providers
python -m embedding.benchmark \
  --providers google,openai,voyage \
  --texts 1000 \
  --batch_sizes 16,32,64
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Issues**
   ```
   Error: Invalid API key
   ```
   **Solution**: Verify environment variables and API key validity

2. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   **Solution**: Reduce `batch_size` or increase `rate_limit_delay`

3. **Memory Issues**
   ```
   Error: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU for local models

4. **Model Not Found**
   ```
   Error: Model 'xyz' not found
   ```
   **Solution**: Check model name and provider compatibility

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enables detailed logging
embedder = get_embedder(provider="google", model="text-embedding-004")
```

### Performance Optimization

```python
# GPU optimization for HuggingFace
embedder = get_embedder(
    provider="hf",
    model="BAAI/bge-large-en-v1.5",
    device="cuda",
    model_kwargs={
        "torch_dtype": "float16",  # Half precision
        "device_map": "auto"
    }
)

# Batch size optimization
optimal_batch_size = embedder.find_optimal_batch_size(sample_texts)
```

## 📈 Best Practices

### Production Deployment

1. **Use Environment Variables**
   ```python
   embedder = get_embedder(
       provider="google",
       api_key=os.getenv("GOOGLE_API_KEY"),  # Never hardcode
       rate_limit_delay=0.1  # Respect API limits
   )
   ```

2. **Enable Caching**
   ```python
   embedder = get_embedder(
       provider="google",
       cache_enabled=True,
       cache_dir="/persistent/cache/"  # Persistent storage
   )
   ```

3. **Monitor Performance**
   ```python
   from logs.utils.logger import get_logger
   
   logger = get_logger(__name__)
   
   start_time = time.time()
   embeddings = embedder.embed_documents(texts)
   duration = time.time() - start_time
   
   logger.info(f"Embedded {len(texts)} docs in {duration:.2f}s", 
               extra={"component": "embedding", "provider": "google"})
   ```

### Cost Optimization

- **Use caching** to avoid re-computing embeddings
- **Choose appropriate models** (smaller for development, larger for production)
- **Batch requests** to maximize API efficiency
- **Monitor usage** to stay within budget limits

### Quality Assurance

```python
# Validate embedding quality
def validate_embeddings(embeddings):
    assert len(embeddings) > 0, "No embeddings generated"
    assert all(len(emb) > 0 for emb in embeddings), "Empty embeddings found"
    assert all(isinstance(val, float) for emb in embeddings for val in emb), "Non-float values"
    
validate_embeddings(embeddings)
```

---

## 🔗 Related Documentation

- **[Database README](../database/README.md)**: Vector storage
- **[Retrievers README](../retrievers/README.md)**: Search and retrieval
- **[Pipelines README](../pipelines/README.md)**: Data ingestion
- **[Main README](../readme.md)**: System overview

## 📞 Support

For embedding-specific issues:
1. Check API key configuration and validity
2. Verify model names and provider compatibility  
3. Monitor rate limits and adjust batch sizes
4. Review provider documentation for specific features
