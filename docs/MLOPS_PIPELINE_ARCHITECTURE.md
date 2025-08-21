# MLOps Pipeline Architecture for RAG Systems

## Table of Contents
1. [Introduction to MLOps for RAG](#introduction)
2. [Overall Architecture](#overall-architecture)
3. [Core Pipeline Components](#core-components)
4. [Data Flow and Processing](#data-flow)
5. [MLOps Principles Implementation](#mlops-principles)
6. [Configuration Management](#configuration)
7. [Reproducibility and Versioning](#reproducibility)
8. [Monitoring and Observability](#monitoring)
9. [Advantages and Trade-offs](#advantages-tradeoffs)
10. [How to Reproduce in Other Projects](#reproduction-guide)

## 1. Introduction to MLOps for RAG {#introduction}

### What is MLOps?
MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

### Why MLOps for RAG Systems?
Retrieval-Augmented Generation (RAG) systems have unique challenges:
- **Data Pipeline Complexity**: Multiple data sources, formats, and processing steps
- **Model Dependencies**: Embedding models, chunking strategies, retrieval algorithms
- **Evaluation Complexity**: Measuring retrieval quality and generation performance
- **Version Management**: Dataset versions, model versions, configuration versions
- **Experimentation**: Comparing different embedding models and retrieval strategies

Our pipeline addresses these challenges through systematic MLOps practices.

## 2. Overall Architecture {#overall-architecture}

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RAG MLOps Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   Raw Data  │───▶│  Adapters   │───▶│ Validation  │             │
│  │ (Multiple   │    │ (Dataset    │    │ & Quality   │             │
│  │  Sources)   │    │ Specific)   │    │   Checks)   │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                │                     │
│                                                ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ Vector Store│◀───│  Embedder   │◀───│   Chunker   │             │
│  │ (Qdrant)    │    │ (Multiple   │    │ (Strategy   │             │
│  │             │    │ Strategies) │    │  Based)     │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                │                     │
│                                                ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ Evaluation  │◀───│ Smoke Tests │◀───│  Lineage    │             │
│  │ Framework   │    │ & Quality   │    │ Tracking    │             │
│  │             │    │ Assurance   │    │             │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles:
1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new datasets, embedders, or evaluation metrics
3. **Reproducibility**: Deterministic IDs, versioning, and configuration management
4. **Observability**: Comprehensive logging, metrics, and lineage tracking
5. **Safety**: Dry-run modes, canary deployments, and validation checks

## 3. Core Pipeline Components {#core-components}

### 3.1 Dataset Adapters (`pipelines/adapters/`)

**Purpose**: Convert raw dataset formats into standardized LangChain Documents

**Architecture**:
```python
class DatasetAdapter(ABC):
    @abstractmethod
    def read_rows(self, split: DatasetSplit) -> Iterable[BaseRow]:
        """Read raw dataset rows"""
    
    @abstractmethod
    def to_documents(self, rows: List[BaseRow], split: DatasetSplit) -> List[Document]:
        """Convert to standardized documents"""
    
    @abstractmethod
    def get_evaluation_queries(self, split: DatasetSplit) -> List[Dict[str, Any]]:
        """Provide evaluation queries"""
```

**Advantages**:
- ✅ **Dataset Agnostic**: Same pipeline works for any dataset
- ✅ **Type Safety**: Pydantic schemas ensure data validity
- ✅ **Extensibility**: Easy to add new datasets
- ✅ **Evaluation Integration**: Built-in evaluation query generation

**Trade-offs**:
- ❌ **Initial Overhead**: Requires implementing adapter for each dataset
- ❌ **Memory Usage**: Loads entire dataset into memory (can be optimized with streaming)

**Example Implementation** (StackOverflow):
```python
class StackOverflowAdapter(DatasetAdapter):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        # Load questions and answers from CSV files
        
    def to_documents(self, rows: List[StackOverflowRow], split: DatasetSplit) -> List[Document]:
        documents = []
        for row in rows:
            # Create question document
            doc = Document(
                page_content=f"Question: {row.title}\n\n{row.body}",
                metadata={
                    "source": self.source_name,
                    "external_id": f"q_{row.question_id}",
                    "split": split.value,
                    "type": "question",
                    "tags": row.tags
                }
            )
            documents.append(doc)
        return documents
```

### 3.2 Validation System (`pipelines/ingest/validator.py`)

**Purpose**: Ensure data quality before processing

**Components**:
- **Character Validation**: Check for problematic characters
- **Content Validation**: Ensure minimum content requirements
- **Metadata Validation**: Verify required fields exist

**Advantages**:
- ✅ **Early Error Detection**: Catch issues before expensive embedding generation
- ✅ **Data Quality Assurance**: Consistent quality across datasets
- ✅ **Configurable Rules**: Different validation rules per dataset type

**Trade-offs**:
- ❌ **Processing Overhead**: Additional validation step
- ❌ **False Positives**: May flag valid content (e.g., HTML in code examples)

**Example Validation Rules**:
```python
def validate_document(self, doc: Document) -> ValidationResult:
    errors = []
    
    # Check minimum length
    if len(doc.page_content.strip()) < self.min_length:
        errors.append(f"Content too short: {len(doc.page_content)} < {self.min_length}")
    
    # Check for required metadata
    if not doc.metadata.get("external_id"):
        errors.append("Missing external_id in metadata")
    
    return ValidationResult(
        valid=len(errors) == 0,
        doc_id=doc.metadata.get("external_id", "unknown"),
        errors=errors
    )
```

### 3.3 Chunking System (`pipelines/ingest/chunker.py`)

**Purpose**: Split documents into optimal chunks for embedding and retrieval

**Strategies**:
- **Recursive Character Splitting**: Split by paragraphs, then sentences, then characters
- **Token-based Splitting**: Split based on token count for transformer models
- **Semantic Splitting**: Future enhancement for content-aware splitting

**Advantages**:
- ✅ **Strategy Flexibility**: Multiple chunking approaches
- ✅ **Deterministic Results**: Same input always produces same chunks
- ✅ **Metadata Preservation**: Chunk metadata tracks source document

**Trade-offs**:
- ❌ **Context Loss**: Splitting may break semantic coherence
- ❌ **Parameter Sensitivity**: Chunk size affects retrieval quality

**Configuration Example**:
```yaml
chunking:
  strategy: "recursive_character"
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", " ", ""]
```

### 3.4 Embedding System (`pipelines/ingest/embedder.py`)

**Purpose**: Generate dense and sparse embeddings for semantic search

**Strategies**:
- **Dense Embeddings**: Sentence transformers (e.g., MiniLM, BGE, E5)
- **Sparse Embeddings**: TF-IDF, BM25 (future: SPLADE)
- **Hybrid Embeddings**: Combination of dense and sparse

**Advantages**:
- ✅ **Multiple Strategies**: Compare different embedding approaches
- ✅ **Caching**: Avoid recomputing embeddings
- ✅ **Batch Processing**: Efficient GPU utilization
- ✅ **Error Handling**: Graceful fallbacks for failed embeddings

**Trade-offs**:
- ❌ **Computational Cost**: Embedding generation is expensive
- ❌ **Model Dependencies**: Different models require different environments
- ❌ **Storage Requirements**: Embeddings consume significant space

**Example Configuration**:
```yaml
embedding:
  strategy: "hybrid"  # dense, sparse, or hybrid
  dense:
    provider: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
  sparse:
    provider: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  cache_enabled: true
```

### 3.5 Vector Store Integration (`pipelines/ingest/uploader.py`)

**Purpose**: Upload embeddings to vector database with proper indexing

**Features**:
- **Collection Management**: Automatic collection creation and configuration
- **Batch Uploads**: Efficient bulk operations
- **Canary Deployments**: Safe testing with temporary collections
- **Metadata Storage**: Rich metadata for filtering and retrieval

**Advantages**:
- ✅ **Scalability**: Handles large datasets efficiently
- ✅ **Safety**: Canary mode prevents affecting production
- ✅ **Flexibility**: Multiple vector stores supported (Qdrant primary)

**Trade-offs**:
- ❌ **Infrastructure Dependency**: Requires vector database setup
- ❌ **Network Overhead**: Upload time depends on network and data size

### 3.6 Lineage Tracking (`pipelines/contracts.py`)

**Purpose**: Track complete provenance of processed data

**Information Tracked**:
- **Data Provenance**: Source dataset, version, and split
- **Processing History**: Chunking strategy, embedding model, configuration
- **Code Provenance**: Git commit hash, configuration hash
- **Quality Metrics**: Success/failure counts, validation results

**Advantages**:
- ✅ **Reproducibility**: Complete history for debugging and reproduction
- ✅ **Compliance**: Audit trail for data governance
- ✅ **Debugging**: Easy to trace issues to specific configurations

**Trade-offs**:
- ❌ **Storage Overhead**: Additional metadata storage
- ❌ **Complexity**: More fields to maintain and track

## 4. Data Flow and Processing {#data-flow}

### Step-by-Step Processing Flow:

```
1. Configuration Loading
   ├── Load YAML config file
   ├── Validate configuration schema
   └── Initialize components with config

2. Data Ingestion
   ├── Adapter reads raw data files
   ├── Convert to standardized BaseRow objects
   └── Generate LangChain Documents

3. Validation
   ├── Check document content quality
   ├── Validate required metadata fields
   └── Filter out invalid documents

4. Chunking
   ├── Split documents using configured strategy
   ├── Generate deterministic chunk IDs
   └── Preserve metadata and provenance

5. Embedding Generation
   ├── Process chunks in batches
   ├── Generate dense/sparse embeddings
   └── Cache results for efficiency

6. Vector Store Upload
   ├── Create/configure collection
   ├── Upload chunks with embeddings
   └── Verify upload success

7. Quality Assurance
   ├── Run smoke tests
   ├── Validate retrieval functionality
   └── Generate quality reports

8. Lineage Recording
   ├── Save complete processing history
   ├── Record configuration and results
   └── Enable reproduction and debugging
```

### ID Generation Strategy:

```python
# Deterministic Document ID
doc_hash = sha256(normalized_content).hexdigest()[:12]
doc_id = f"{source}:{external_id}:{doc_hash}"

# Deterministic Chunk ID  
chunk_id = f"{doc_id}#c{chunk_index:04d}"
```

**Benefits**:
- ✅ **Idempotency**: Rerunning pipeline produces same IDs
- ✅ **Deduplication**: Same content gets same ID across runs
- ✅ **Traceability**: Easy to trace chunks back to source documents

## 5. MLOps Principles Implementation {#mlops-principles}

### 5.1 Reproducibility

**Implementation**:
- **Deterministic IDs**: Content-based hashing ensures same results
- **Configuration Versioning**: YAML configs tracked in git
- **Environment Specification**: Requirements.txt pins exact versions
- **Data Versioning**: Dataset versions tracked in metadata

**Example**:
```yaml
# Configuration is versioned and tracked
dataset:
  name: "stackoverflow"
  version: "1.0.0"
  path: "/data/sosum"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  # Exact model version ensures reproducibility
```

### 5.2 Experimentation

**A/B Testing Support**:
```yaml
# Different configs for comparing embedding models
collection_name: "sosum_stackoverflow_minilm_v1"    # MiniLM experiment
collection_name: "sosum_stackoverflow_bge_large_v1" # BGE Large experiment
collection_name: "sosum_stackoverflow_e5_large_v1"  # E5 Large experiment
```

**Benefits**:
- ✅ **Safe Comparison**: Each experiment uses separate collection
- ✅ **Parallel Testing**: Multiple configurations can run simultaneously
- ✅ **Easy Rollback**: Keep previous versions available

### 5.3 Monitoring and Observability

**Logging Strategy**:
```python
logger.info(f"Processing {len(documents)} documents with {strategy} strategy")
logger.warning(f"Validation errors found: {validation_errors}")
logger.error(f"Embedding generation failed: {error}")
```

**Metrics Tracked**:
- Processing times per component
- Success/failure rates
- Data quality metrics
- Embedding generation statistics

### 5.4 Quality Assurance

**Multi-layer Validation**:
1. **Input Validation**: Check raw data quality
2. **Processing Validation**: Verify each transformation step
3. **Output Validation**: Smoke tests on final results
4. **End-to-end Testing**: Retrieval quality evaluation

## 6. Configuration Management {#configuration}

### Configuration Schema:
```yaml
# Dataset Configuration
dataset:
  name: "stackoverflow"
  version: "1.0.0" 
  adapter: "stackoverflow"
  path: "/path/to/data"

# Processing Configuration
chunking:
  strategy: "recursive_character"
  chunk_size: 1000
  chunk_overlap: 200

embedding:
  strategy: "dense"  # dense, sparse, hybrid
  provider: "hf"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

# Infrastructure Configuration
vector_store:
  provider: "qdrant"
  collection_name: "sosum_stackoverflow_minilm_v1"
  distance_metric: "cosine"

# Experiment Configuration
experiment:
  name: "minilm_baseline"
  description: "Baseline with MiniLM embeddings"
  canary: false
  max_documents: null  # null = no limit
```

### Configuration Benefits:
- ✅ **Declarative**: Infrastructure as code approach
- ✅ **Version Controlled**: Track configuration changes
- ✅ **Environment Specific**: Different configs for dev/staging/prod
- ✅ **Validation**: Schema validation prevents configuration errors

## 7. Reproducibility and Versioning {#reproducibility}

### Version Management Strategy:

```python
class ChunkMeta(BaseModel):
    # Identity and Content
    doc_id: str              # Deterministic based on content
    chunk_id: str            # Deterministic based on doc_id + index
    doc_sha256: str          # Content hash for integrity
    
    # Provenance Tracking
    source: str              # Dataset name
    dataset_version: str     # Dataset version
    git_commit: str          # Code version when processed
    config_hash: str         # Configuration hash
    
    # Processing Metadata
    embedding_model: str     # Exact model used
    chunk_strategy: dict     # Chunking parameters used
```

### Reproduction Steps:
1. **Checkout Code**: Use git commit from lineage record
2. **Load Configuration**: Use exact config from lineage record  
3. **Install Dependencies**: Use requirements.txt from that commit
4. **Run Pipeline**: Should produce identical results

### Benefits:
- ✅ **Full Traceability**: Know exactly how any chunk was created
- ✅ **Bug Investigation**: Reproduce issues from production
- ✅ **Compliance**: Meet audit requirements for data processing

## 8. Monitoring and Observability {#monitoring}

### Observability Stack:

```python
# Structured Logging
logger.info(
    "Embedding generation completed",
    extra={
        "component": "embedder",
        "strategy": "dense",
        "model": "all-MiniLM-L6-v2",
        "batch_size": 32,
        "chunks_processed": 1500,
        "processing_time": 45.2,
        "success_rate": 0.998
    }
)
```

### Key Metrics:
- **Throughput**: Documents/chunks processed per minute
- **Quality**: Validation success rates, embedding generation success
- **Performance**: Processing time per component
- **Resource Usage**: Memory, CPU, GPU utilization
- **Error Rates**: Failed documents, failed embeddings

### Alerts and Monitoring:
- High failure rates in validation or embedding
- Processing time exceeding thresholds
- Resource utilization issues
- Data quality degradation

## 9. Advantages and Trade-offs {#advantages-tradeoffs}

### Overall Architecture Advantages:

✅ **Modularity**: 
- Easy to replace components (e.g., switch from Qdrant to Pinecone)
- Test components in isolation
- Parallel development by different team members

✅ **Reproducibility**:
- Deterministic results across runs
- Complete provenance tracking
- Easy debugging and issue reproduction

✅ **Scalability**:
- Batch processing for efficiency
- Horizontal scaling of individual components
- Streaming support for large datasets (future enhancement)

✅ **Experimentation**:
- A/B testing different configurations
- Safe canary deployments
- Easy comparison of approaches

✅ **Quality Assurance**:
- Multi-layer validation
- Automated testing and smoke tests
- Continuous monitoring

### Trade-offs and Limitations:

❌ **Complexity**:
- More complex than simple scripts
- Requires understanding of MLOps concepts
- More code to maintain

❌ **Initial Setup Cost**:
- Significant upfront investment
- Infrastructure dependencies (vector database, etc.)
- Learning curve for team members

❌ **Resource Requirements**:
- Embedding generation requires computational resources
- Vector storage requires significant disk space
- Caching increases memory usage

❌ **Vendor Dependencies**:
- Qdrant for vector storage
- HuggingFace for embedding models
- Specific Python version and libraries

### When to Use This Architecture:

**Good Fit**:
- Multiple datasets to process
- Need for experimentation and comparison
- Production RAG systems requiring reliability
- Teams needing reproducibility and compliance
- Long-term projects requiring maintenance

**Not Ideal For**:
- One-off experiments or prototypes
- Very small datasets (< 1000 documents)
- Teams without MLOps experience
- Projects with tight deadlines
- Limited computational resources

## 10. How to Reproduce in Other Projects {#reproduction-guide}

### 10.1 Project Structure Setup

```
your_project/
├── pipelines/
│   ├── contracts.py          # Base schemas and interfaces
│   ├── adapters/             # Dataset-specific adapters
│   │   ├── your_dataset.py
│   │   └── another_dataset.py
│   ├── ingest/               # Core processing components
│   │   ├── validator.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── uploader.py
│   │   └── pipeline.py
│   ├── configs/              # Configuration files
│   │   ├── baseline.yml
│   │   └── experiment.yml
│   └── eval/                 # Evaluation framework
│       └── evaluator.py
├── bin/
│   └── ingest.py            # CLI interface
├── docs/
│   └── architecture.md
└── requirements.txt
```

### 10.2 Implementation Steps

#### Step 1: Define Core Contracts
```python
# pipelines/contracts.py
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum

class DatasetSplit(str, Enum):
    TRAIN = "train"
    TEST = "test"
    ALL = "all"

class BaseRow(BaseModel):
    external_id: str
    class Config:
        extra = "allow"

class DatasetAdapter(ABC):
    @abstractmethod
    def read_rows(self, split: DatasetSplit) -> Iterable[BaseRow]:
        pass
    
    @abstractmethod  
    def to_documents(self, rows: List[BaseRow], split: DatasetSplit) -> List[Document]:
        pass
```

#### Step 2: Implement Dataset Adapter
```python
# pipelines/adapters/your_dataset.py
class YourDatasetAdapter(DatasetAdapter):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    @property
    def source_name(self) -> str:
        return "your_dataset"
    
    def read_rows(self, split: DatasetSplit) -> Iterable[YourDatasetRow]:
        # Load your data format (CSV, JSON, etc.)
        for item in self._load_data():
            yield YourDatasetRow(**item)
    
    def to_documents(self, rows: List[YourDatasetRow], split: DatasetSplit) -> List[Document]:
        documents = []
        for row in rows:
            doc = Document(
                page_content=row.content,
                metadata={
                    "source": self.source_name,
                    "external_id": row.id,
                    "split": split.value,
                    # Add your specific metadata
                }
            )
            documents.append(doc)
        return documents
```

#### Step 3: Configure Processing Pipeline
```yaml
# pipelines/configs/your_config.yml
dataset:
  name: "your_dataset"
  version: "1.0.0"
  adapter: "your_dataset"
  path: "/path/to/your/data"

chunking:
  strategy: "recursive_character"
  chunk_size: 1000
  chunk_overlap: 200

embedding:
  strategy: "dense"
  provider: "hf"
  model: "sentence-transformers/all-MiniLM-L6-v2"

vector_store:
  provider: "qdrant"
  collection_name: "your_dataset_v1"
```

#### Step 4: Run Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Start vector database (if using Qdrant)
docker run -p 6333:6333 qdrant/qdrant

# Run ingestion
python bin/ingest.py --config pipelines/configs/your_config.yml \
    ingest your_dataset /path/to/data --dry-run --max-docs 100

# Run without dry-run when ready
python bin/ingest.py --config pipelines/configs/your_config.yml \
    ingest your_dataset /path/to/data
```

### 10.3 Customization Points

#### Custom Validation Rules:
```python
def validate_document(self, doc: Document) -> ValidationResult:
    errors = []
    
    # Your domain-specific validation
    if "required_field" not in doc.metadata:
        errors.append("Missing required field")
    
    # Custom content checks
    if len(doc.page_content.split()) < 10:
        errors.append("Content too short")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors
    )
```

#### Custom Embedding Provider:
```python
class CustomEmbedder:
    def __init__(self, config: Dict[str, Any]):
        self.model = load_your_model(config["model_path"])
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

#### Custom Evaluation Metrics:
```python
def evaluate_retrieval(self, queries: List[str], ground_truth: List[List[str]]) -> Dict[str, float]:
    # Implement your evaluation logic
    results = {}
    for k in [1, 3, 5, 10]:
        results[f"recall_at_{k}"] = compute_recall_at_k(predictions, ground_truth, k)
    return results
```

### 10.4 Best Practices for Adaptation

1. **Start Simple**: Begin with basic adapter and gradually add features
2. **Use Type Hints**: Leverage Pydantic for data validation and documentation  
3. **Test Components**: Write unit tests for each component
4. **Configuration First**: Make everything configurable from YAML files
5. **Log Everything**: Add comprehensive logging for debugging
6. **Version Everything**: Track dataset versions, model versions, and code versions
7. **Validate Early**: Catch data quality issues as early as possible
8. **Plan for Scale**: Consider memory and compute requirements
9. **Document Decisions**: Explain why certain approaches were chosen
10. **Monitor in Production**: Set up alerts and monitoring for production systems

### 10.5 Common Pitfalls to Avoid

1. **Hard-coded Paths**: Use configuration files instead
2. **Missing Error Handling**: Plan for failures in each component
3. **No Rollback Strategy**: Always have a way to revert changes
4. **Insufficient Testing**: Test with small datasets first
5. **Ignoring Resource Limits**: Monitor memory and disk usage
6. **No Backup Strategy**: Plan for data and model backup
7. **Vendor Lock-in**: Design for portability between providers
8. **Poor Documentation**: Document configuration options and troubleshooting

This architecture provides a solid foundation for MLOps in RAG systems. The key is to start with the core components and gradually add complexity based on your specific needs. The modular design ensures you can adapt it to different domains, datasets, and requirements while maintaining the benefits of reproducibility, scalability, and quality assurance.
