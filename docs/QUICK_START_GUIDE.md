# Quick Start Guide: Understanding the MLOps Pipeline for RAG

**⚠️ Important Note**: This guide provides a simplified tutorial for understanding the MLOps concepts. The actual project has a much more sophisticated implementation with advanced features like hybrid embeddings, multiple chunking strategies, agent workflows, and comprehensive benchmarking.

**To use the actual project:**
- See `README.md` for setup instructions
- Use the CLI: `python bin/ingest.py --help`
- Check `docs/SOSUM_INGESTION.md` for real dataset examples
- Review `docs/MLOPS_PIPELINE_ARCHITECTURE.md` for detailed architecture

This guide provides a step-by-step walkthrough for implementing a simplified version of the MLOps pipeline architecture.

## Prerequisites

- Python 3.9+
- Docker (for vector database)
- Git (for version control)
- Basic understanding of ML and RAG concepts

## 1. Project Initialization (15 minutes)

### Create Project Structure
```bash
mkdir my-rag-project
cd my-rag-project

# Create directory structure
mkdir -p {pipelines/{adapters,ingest,configs,eval},bin,docs,tests}
mkdir -p {embedding,database,logs/utils,examples,scripts}

# Initialize git repository
git init
```

### Setup Python Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install core dependencies
pip install pydantic langchain langchain-core qdrant-client sentence-transformers pandas pyyaml python-dotenv
```

## 2. Implement Core Contracts (30 minutes)

### Create Base Contracts (`pipelines/contracts.py`)
```python
"""Core contracts for the RAG pipeline."""
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.documents import Document

class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    ALL = "all"

class BaseRow(BaseModel):
    """Base schema for dataset rows."""
    external_id: str = Field(..., description="Unique identifier from source")
    
    class Config:
        extra = "allow"

class ChunkMeta(BaseModel):
    """Metadata for processed chunks."""
    # Identity
    doc_id: str
    chunk_id: str
    doc_sha256: str
    text: str
    
    # Source
    source: str
    dataset_version: str
    external_id: str
    
    # Processing
    chunk_index: int
    num_chunks: int
    char_count: int
    split: DatasetSplit
    
    # Pipeline metadata
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    
    # Embeddings
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    dense_embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None
    
    # Additional metadata
    labels: Dict[str, Any] = Field(default_factory=dict)

class DatasetAdapter(ABC):
    """Abstract adapter for datasets."""
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        pass
    
    @abstractmethod
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[BaseRow]:
        pass
    
    @abstractmethod
    def to_documents(self, rows: List[BaseRow], split: DatasetSplit) -> List[Document]:
        pass
    
    @abstractmethod
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        pass

# Utility functions
def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def build_doc_id(source: str, external_id: str, content_hash: str) -> str:
    """Build deterministic document ID."""
    return f"{source}:{external_id}:{content_hash[:12]}"

def build_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Build deterministic chunk ID."""
    return f"{doc_id}#c{chunk_index:04d}"
```

## 3. Implement Your First Dataset Adapter (45 minutes)

### Example: CSV Dataset Adapter (`pipelines/adapters/csv_dataset.py`)
```python
"""CSV dataset adapter example."""
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Dict, Any

from langchain_core.documents import Document
from pipelines.contracts import DatasetAdapter, BaseRow, DatasetSplit

class CSVRow(BaseRow):
    """Schema for CSV dataset rows."""
    title: str
    content: str
    category: Optional[str] = None

class CSVDatasetAdapter(DatasetAdapter):
    """Adapter for CSV-based datasets."""
    
    def __init__(self, data_path: str, text_column: str = "content", id_column: str = "id"):
        self.data_path = Path(data_path)
        self.text_column = text_column
        self.id_column = id_column
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
    
    @property
    def source_name(self) -> str:
        return "csv_dataset"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[CSVRow]:
        """Read rows from CSV file."""
        if self.data_path.is_file():
            # Single CSV file
            df = pd.read_csv(self.data_path)
        else:
            # Directory with split files
            split_file = self.data_path / f"{split.value}.csv"
            if not split_file.exists() and split == DatasetSplit.ALL:
                # Try common filenames
                for filename in ["data.csv", "dataset.csv", "train.csv"]:
                    split_file = self.data_path / filename
                    if split_file.exists():
                        break
            
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")
            
            df = pd.read_csv(split_file)
        
        for _, row in df.iterrows():
            yield CSVRow(
                external_id=str(row[self.id_column]),
                title=row.get("title", ""),
                content=row[self.text_column],
                category=row.get("category")
            )
    
    def to_documents(self, rows: List[CSVRow], split: DatasetSplit) -> List[Document]:
        """Convert rows to LangChain documents."""
        documents = []
        
        for row in rows:
            # Combine title and content
            if row.title:
                content = f"{row.title}\n\n{row.content}"
            else:
                content = row.content
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": self.source_name,
                    "external_id": row.external_id,
                    "split": split.value,
                    "title": row.title,
                    "category": row.category,
                    "dataset_version": self.version
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Generate evaluation queries."""
        # Simple approach: use titles as queries
        queries = []
        for row in self.read_rows(split):
            if row.title:
                queries.append({
                    "query": row.title,
                    "expected_doc_id": row.external_id,
                    "category": row.category
                })
        
        return queries[:100]  # Limit for testing
```

## 4. Create Configuration System (20 minutes)

### Configuration Schema (`pipelines/configs/config_schema.py`)
```python
"""Configuration schema validation."""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class DatasetConfig(BaseModel):
    name: str
    version: str
    adapter: str
    path: str

class ChunkingConfig(BaseModel):
    strategy: str = "recursive_character"
    chunk_size: int = 1000
    chunk_overlap: int = 200

class EmbeddingConfig(BaseModel):
    strategy: str = "dense"  # dense, sparse, hybrid
    provider: str = "hf"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    collection_name: str
    host: str = "localhost"
    port: int = 6333

class PipelineConfig(BaseModel):
    dataset: DatasetConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    
    # Optional experiment settings
    experiment: Optional[Dict[str, Any]] = None
    max_documents: Optional[int] = None
    dry_run: bool = False
```

### Example Configuration (`pipelines/configs/csv_example.yml`)
```yaml
dataset:
  name: "my_csv_dataset"
  version: "1.0.0"
  adapter: "csv_dataset"
  path: "/path/to/your/data.csv"

chunking:
  strategy: "recursive_character"
  chunk_size: 1000
  chunk_overlap: 200

embedding:
  strategy: "dense"
  provider: "hf"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 16

vector_store:
  provider: "qdrant"
  collection_name: "my_csv_dataset_v1"
  host: "localhost"
  port: 6333

experiment:
  name: "baseline"
  description: "Initial baseline with MiniLM embeddings"
  canary: false

max_documents: null  # null = no limit
dry_run: false
```

## 5. Implement Core Processing Components (60 minutes)

### Simple Chunker (`pipelines/ingest/chunker.py`)
```python
"""Document chunking functionality."""
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Chunks documents using configurable strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = config.get("strategy", "recursive_character")
        
        if self.strategy == "recursive_character":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
                separators=config.get("separators", ["\n\n", "\n", " ", ""])
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        logger.info(f"Chunking {len(documents)} documents with {self.strategy} strategy")
        
        chunked_docs = []
        
        for doc in documents:
            chunks = self.splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "num_chunks": len(chunks),
                    "chunk_strategy": self.strategy,
                    "original_doc_id": doc.metadata.get("external_id")
                })
                chunked_docs.append(chunk)
        
        logger.info(f"Generated {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
```

**Note**: The actual implementation (`pipelines/ingest/chunker.py`) has multiple advanced chunking strategies:
- `RecursiveChunkingStrategy`: Basic recursive character splitting
- `SemanticChunkingStrategy`: Sentence-boundary aware chunking  
- `CodeAwareChunkingStrategy`: Preserves code blocks and functions
- `TableAwareChunkingStrategy`: Preserves table structure
- `ChunkingStrategyFactory`: Factory for strategy selection

### Simple Embedder (`pipelines/ingest/embedder.py`)
```python
"""Embedding generation."""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from pipelines.contracts import ChunkMeta, compute_content_hash, build_doc_id, build_chunk_id, DatasetSplit

logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """Generate embeddings for documents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = config.get("strategy", "dense")
        
        if self.strategy in ["dense", "hybrid"]:
            model_name = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
    
    def process_documents(self, documents: List[Document]) -> List[ChunkMeta]:
        """Convert documents to ChunkMeta with embeddings."""
        logger.info(f"Processing {len(documents)} documents for embeddings")
        
        chunk_metas = []
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batch
        if self.strategy in ["dense", "hybrid"]:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Convert to ChunkMeta
        for i, doc in enumerate(documents):
            chunk_meta = self._document_to_chunk_meta(doc)
            
            if self.strategy in ["dense", "hybrid"]:
                chunk_meta.dense_embedding = embeddings[i].tolist()
                chunk_meta.embedding_dim = len(embeddings[i])
                chunk_meta.embedding_model = self.config.get("model")
            
            chunk_metas.append(chunk_meta)
        
        return chunk_metas
    
    def _document_to_chunk_meta(self, doc: Document) -> ChunkMeta:
        """Convert Document to ChunkMeta."""
        text = doc.page_content
        metadata = doc.metadata
        
        # Generate deterministic IDs
        doc_sha256 = compute_content_hash(text)
        source = metadata.get("source", "unknown")
        external_id = metadata.get("external_id", "unknown")
        
        doc_id = build_doc_id(source, external_id, doc_sha256)
        chunk_index = metadata.get("chunk_index", 0)
        chunk_id = build_chunk_id(doc_id, chunk_index)
        
        return ChunkMeta(
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_sha256=doc_sha256,
            text=text,
            source=source,
            dataset_version=metadata.get("dataset_version", "unknown"),
            external_id=external_id,
            chunk_index=chunk_index,
            num_chunks=metadata.get("num_chunks", 1),
            char_count=len(text),
            split=DatasetSplit(metadata.get("split", "all")),
            labels=metadata
        )
```

**Note**: The actual implementation (`pipelines/ingest/embedder.py`) is more sophisticated with:
- Support for dense, sparse, and hybrid embedding strategies
- Caching and error handling
- Batch processing with progress bars
- Integration with multiple embedding providers (HuggingFace, Google, AWS Bedrock)

## 6. Use the Actual CLI Interface (15 minutes)

The actual project has a sophisticated CLI with subcommands. Here's how to use it:

### CLI Usage Examples
```bash
# View available commands
python bin/ingest.py --help

# Ingest a dataset (dry run)
python bin/ingest.py ingest natural_questions /path/to/data --config config.yml --dry-run --max-docs 100

# Ingest Stack Overflow dataset
python bin/ingest.py ingest stackoverflow /path/to/sosum --config config.yml

# Run in canary mode for testing
python bin/ingest.py ingest energy_papers /path/to/papers --canary --max-docs 50

# Check collection status
python bin/ingest.py status --config config.yml

# Evaluate retrieval performance
python bin/ingest.py evaluate natural_questions /path/to/data --output-dir results/

# Batch ingestion
python bin/ingest.py batch-ingest batch_config.json
```

### Batch Configuration Example (`batch_config.json`)
```json
{
  "datasets": [
    {"type": "natural_questions", "path": "/path/to/nq", "version": "1.0.0"},
    {"type": "stackoverflow", "path": "/path/to/sosum", "version": "1.0.0"}
  ]
}
```

### Available Adapter Types
- `natural_questions`: Natural Questions dataset
- `stackoverflow`: Stack Overflow (SOSum format) dataset  
- `energy_papers`: Energy research papers dataset

## 7. Test with Actual Implementation (15 minutes)

### Use Real Configuration Files
The actual project has several pre-configured YAML files you can use:

```bash
# List available configurations
ls pipelines/configs/retrieval/

# Available configs:
# - modern_dense.yml: Dense embeddings with neural reranking
# - modern_hybrid.yml: Hybrid dense+sparse with reranking  
# - fast_hybrid.yml: Fast hybrid retrieval
# - ci_google_gemini.yml: CI configuration with Google embeddings
```

### Test with Stack Overflow Dataset
```bash
# Download SOSum dataset
mkdir -p datasets/sosum
cd datasets/sosum
# Download from https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools

# Test the adapter (dry run)
python bin/ingest.py ingest stackoverflow datasets/sosum/data --config config.yml --dry-run --max-docs 10 --verbose

# Check what was ingested
python bin/ingest.py status --config config.yml
```

### Actual Configuration Structure
The real `config.yml` looks like this:
```yaml
# Main configuration file
agent:
  retrieval_pipeline_config: "pipelines/configs/retrieval/modern_dense.yml"

database:
  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "sosum_stackoverflow_hybrid_v1"

# The retrieval configs contain detailed chunking and embedding settings
```

## 8. Next Steps and Extensions

### Immediate Improvements
1. **Add Vector Store Integration**: Implement actual upload to Qdrant
2. **Add Validation**: Input validation and quality checks
3. **Add Error Handling**: Robust error handling and recovery
4. **Add Logging**: Structured logging with metrics

### Advanced Features
1. **Evaluation Framework**: Implement retrieval evaluation
2. **Configuration Validation**: Schema validation for configs
3. **Experiment Tracking**: MLflow or W&B integration
4. **Monitoring**: Prometheus metrics and Grafana dashboards
5. **Streaming**: Process large datasets without loading into memory

### Production Readiness
1. **Docker Containers**: Containerize the application
2. **CI/CD Pipeline**: Automated testing and deployment
3. **Infrastructure as Code**: Terraform for cloud resources
4. **Security**: Authentication, authorization, and encryption
5. **Backup and Recovery**: Data backup and disaster recovery

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Make sure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing Dependencies**:
```bash
# Install additional packages as needed
pip install sentence-transformers pandas pyyaml
```

**Configuration Errors**:
- Check YAML syntax
- Verify file paths exist
- Ensure adapter names match module names

**Memory Issues**:
- Reduce batch_size in embedding config
- Use --max-docs to limit dataset size
- Consider streaming implementation for large datasets

This quick start guide should get you up and running with a basic MLOps pipeline for RAG systems. Start with this foundation and gradually add more sophisticated features as your needs grow.
