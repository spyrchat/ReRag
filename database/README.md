# Database Module

Vector database abstraction layer with production-ready Qdrant integration for hybrid dense+sparse retrieval.

## 📋 Overview

The database module provides a clean abstraction over vector databases, currently implementing Qdrant as the primary backend. It supports:

- **Hybrid Indexing**: Dense + sparse vectors in the same collection
- **Production Configuration**: Environment variables, API keys, cloud deployment
- **Automatic Collection Management**: Schema creation, versioning, cleanup
- **LangChain Integration**: Seamless compatibility with LangChain VectorStore

## 🏗️ Architecture

```
database/
├── base.py                 # Abstract interfaces
├── qdrant_controller.py    # Qdrant implementation
└── README.md              # This file
```

### Class Hierarchy

```python
BaseVectorDB (Abstract)
    ↓
QdrantVectorDB (Concrete)
    ↓
LangChain VectorStore Integration
```

## 🚀 Quick Start

### Basic Usage

```python
from database.qdrant_controller import QdrantVectorDB

# Initialize with defaults (localhost)
db = QdrantVectorDB(strategy="hybrid")

# Initialize with custom config
config = {
    "qdrant": {
        "host": "your-qdrant-cloud.com",
        "api_key": "your-api-key",
        "collection": "my_collection"
    }
}
db = QdrantVectorDB(strategy="hybrid", config=config)

# Initialize collection for 1024-dimensional vectors
db.init_collection(dense_vector_size=1024)
```

### Document Insertion

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Renewable energy is sustainable...",
        metadata={"source": "energy_paper.pdf", "page": 1}
    ),
    Document(
        page_content="Solar panels convert sunlight...",
        metadata={"source": "solar_guide.pdf", "page": 3}
    )
]

# Insert with embeddings
db.insert_documents(
    documents=documents,
    dense_embedder=your_dense_embedder,
    sparse_embedder=your_sparse_embedder
)
```

### LangChain Integration

```python
# Get LangChain-compatible vectorstore
vectorstore = db.as_langchain_vectorstore(
    dense_embedding=dense_embedder,
    sparse_embedding=sparse_embedder,
    strategy="hybrid"  # or "dense", "sparse"
)

# Use with LangChain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.get_relevant_documents("your query")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `QDRANT_HOST` | Qdrant server host | `localhost` | ✅ |
| `QDRANT_PORT` | Qdrant server port | `6333` | No |
| `QDRANT_API_KEY` | API key (for cloud) | `None` | Cloud only |
| `QDRANT_COLLECTION` | Collection name | `default_collection` | ✅ |
| `DENSE_VECTOR_NAME` | Dense vector field name | `dense` | No |
| `SPARSE_VECTOR_NAME` | Sparse vector field name | `sparse` | No |

### Config Object

```python
config = {
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "api_key": None,  # Optional
        "collection": "my_collection",
        "dense_vector_name": "dense",
        "sparse_vector_name": "sparse"
    }
}
```

### Retrieval Strategies

- **`dense`**: Dense vector search only (semantic similarity)
- **`sparse`**: Sparse vector search only (keyword matching)  
- **`hybrid`**: Combined dense + sparse with score fusion

## 🔧 Advanced Usage

### Custom Collection Management

```python
# Check if collection exists
if db.client.collection_exists("my_collection"):
    print("Collection exists")

# Recreate collection (deletes existing data)
db.init_collection(dense_vector_size=1024)

# Get raw Qdrant client for advanced operations
client = db.get_client()
collections = client.get_collections()
```

### Batch Operations

```python
# Large batch insertion
large_documents = [...]  # 10,000+ documents

db.insert_documents(
    documents=large_documents,
    dense_embedder=embedder,
    sparse_embedder=sparse_embedder
)
# Automatically handles batching and memory management
```

### External ID Management

```python
# Documents with external IDs (for updates/deduplication)
documents = [
    Document(
        page_content="Content here",
        metadata={
            "external_id": "doc_123",  # Will be used as vector ID
            "source": "file.pdf"
        }
    )
]

db.insert_documents(documents, dense_embedder=embedder)
# Uses "doc_123" as the vector ID in Qdrant
```

## 🏥 Health & Monitoring

### Connection Testing

```python
try:
    db = QdrantVectorDB()
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
```

### Collection Statistics

```python
client = db.get_client()
collection_info = client.get_collection("my_collection")
print(f"Vectors: {collection_info.vectors_count}")
print(f"Status: {collection_info.status}")
```

## 🐳 Deployment

### Local Development

```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant:latest

# Or use docker-compose
docker-compose up -d qdrant
```

### Production Deployment

```python
# Cloud configuration
config = {
    "qdrant": {
        "host": "xyz-abc.qdrant.tech",
        "port": 6333,
        "api_key": "your-api-key",
        "collection": "production_collection"
    }
}

db = QdrantVectorDB(config=config)
```

### Environment Variables (Production)

```bash
export QDRANT_HOST=your-qdrant-instance.com
export QDRANT_API_KEY=your-api-key
export QDRANT_COLLECTION=production_collection

# No .env file needed - uses environment variables directly
```

## 🧪 Testing

### Unit Tests

```bash
# Run database unit tests
pytest tests/unit/test_database.py -v
```

### Integration Tests

```bash
# Start Qdrant first
docker-compose up -d qdrant

# Run integration tests
pytest tests/integration/test_qdrant_integration.py -v
```

### Health Check

```bash
# Quick connectivity test
python -c "
from database.qdrant_controller import QdrantVectorDB
try:
    db = QdrantVectorDB()
    print('✅ Database OK')
except Exception as e:
    print(f'❌ Database Error: {e}')
"
```

## 🔌 Extension Points

### Adding New Vector Databases

1. **Implement Base Interface**
   ```python
   from database.base import BaseVectorDB
   
   class MyVectorDB(BaseVectorDB):
       def init_collection(self, dense_vector_size: int) -> None:
           # Implementation here
           pass
       
       def insert_documents(self, documents, dense_embedder, sparse_embedder) -> None:
           # Implementation here
           pass
   ```

2. **Register in Factory** (if using factory pattern)
   ```python
   DATABASE_REGISTRY["my_db"] = MyVectorDB
   ```

### Custom Metadata Schemas

```python
# Add custom metadata processing
class CustomQdrantDB(QdrantVectorDB):
    def insert_documents(self, documents, dense_embedder, sparse_embedder):
        # Custom preprocessing
        for doc in documents:
            doc.metadata["processed_at"] = datetime.now().isoformat()
            doc.metadata["vector_version"] = "v2.0"
        
        # Call parent implementation
        super().insert_documents(documents, dense_embedder, sparse_embedder)
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: Connection refused to localhost:6333
   ```
   **Solution**: Start Qdrant with `docker-compose up -d qdrant`

2. **API Key Authentication**
   ```
   Error: Unauthorized access
   ```
   **Solution**: Set `QDRANT_API_KEY` environment variable

3. **Collection Already Exists**
   ```
   Error: Collection 'my_collection' already exists
   ```
   **Solution**: Use `init_collection()` to recreate or choose different name

4. **Vector Dimension Mismatch**
   ```
   Error: Vector dimension mismatch
   ```
   **Solution**: Ensure embedder output matches `dense_vector_size` in collection

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enables detailed Qdrant operation logging
db = QdrantVectorDB()
```

### Performance Tuning

```python
# For large datasets
config = {
    "qdrant": {
        "collection": "large_collection",
        # Add Qdrant-specific optimizations here
    }
}
```

## 📊 Monitoring

### Key Metrics

- **Connection Health**: Regular connectivity checks
- **Collection Size**: Number of vectors stored
- **Query Performance**: Average response time
- **Memory Usage**: Vector storage efficiency

### Logging

```python
from logs.utils.logger import get_logger

logger = get_logger(__name__)

# Database operations are automatically logged
db.insert_documents(...)  # Logs: "Inserted 100 documents"
```

---

## 🔗 Related Documentation

- **[Pipelines README](../pipelines/README.md)**: Data ingestion pipeline
- **[Embedding README](../embedding/README.md)**: Embedding generation
- **[Retrievers README](../retrievers/README.md)**: Search and retrieval
- **[Main README](../readme.md)**: System overview

## 📞 Support

For database-specific issues:
1. Check Qdrant logs: `docker logs <qdrant-container>`
2. Verify configuration with health checks
3. Review connection parameters and API keys
4. Check [Qdrant documentation](https://qdrant.tech/documentation/) for advanced features
