# Advanced RAG System - Complete Codebase Documentation

**Version**: 1.0.0  
**Date**: September 11, 2025  
**Author**: Comprehensive Analysis of Advanced RAG MLOps Pipeline

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Component Deep Dive](#component-deep-dive)
4. [Data Insertion Pipeline](#data-insertion-pipeline)
5. [Data Retrieval Pipeline](#data-retrieval-pipeline)
6. [Agent Workflow System](#agent-workflow-system)
7. [Configuration Management](#configuration-management)
8. [Vector Database Integration](#vector-database-integration)
9. [Embedding Systems](#embedding-systems)
10. [Evaluation & Benchmarking](#evaluation--benchmarking)
11. [CLI Tools & Utilities](#cli-tools--utilities)
12. [Error Handling & Monitoring](#error-handling--monitoring)
13. [Extension Points](#extension-points)

---

## System Overview

### What This System Does

This is a **production-ready Advanced RAG (Retrieval-Augmented Generation) system** with sophisticated MLOps capabilities. It provides:

- **Intelligent Document Processing**: Multi-strategy chunking, validation, and quality assurance
- **Hybrid Retrieval**: Dense, sparse, and hybrid vector search with intelligent fusion
- **LangGraph Agent Workflows**: Configurable AI agents with query interpretation and response generation
- **Comprehensive Benchmarking**: Built-in evaluation framework with multiple metrics
- **MLOps Pipeline**: Complete lineage tracking, reproducibility, and monitoring

### Core Capabilities

```
📊 Data Ingestion → 🔍 Retrieval → 🤖 Agent Processing → 📈 Evaluation
     ↓                ↓               ↓                    ↓
   Qdrant DB      Vector Search    LLM Generation     Benchmarks
```

### Key Technologies

- **Vector Database**: Qdrant (primary), supports hybrid dense+sparse indexing
- **Embeddings**: Voyage AI, Google Gemini, HuggingFace, AWS Bedrock
- **Agent Framework**: LangGraph with OpenAI GPT models
- **Configuration**: YAML-based, environment-aware
- **Data Processing**: LangChain, Pydantic schemas, deterministic IDs

---

## Architecture & Data Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG MLOps System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Raw Data  │──▶│  Adapters   │--▶│ Validation  │              │
│  │ (Multiple   │    │ (Dataset    │    │ & Quality   │              │
│  │  Sources)   │    │ Specific)   │    │   Checks    │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                │                    │
│                                                ▼                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Chunking  │◀──│ Config Mgmt │──> | Embeddings  │              │
│  │ (Multiple   │    │ (YAML-based │    │ (Dense +    │              │
│  │ Strategies) │    │ Hierarchical│    │  Sparse)    │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                │                    │
│                                                ▼                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Qdrant    │◀──│ Uploader &  │◀──│ ChunkMeta   │              │
│  │ Vector DB   │    │ Versioning  │    │ Generation  │              │
│  │ (Hybrid)    │    │             │    │             │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                     │
│                         RETRIEVAL LAYER                             │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │Query Input  │───>│ Retrieval   │───>│ Reranking & │              │
│  │             │    │ Pipeline    │    │ Filtering   │              │
│  │             │    │ (Configur.) │    │             │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                │                    │
│                                                ▼                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │ LangGraph   │◀──│ Agent Nodes │◀─ │ Retrieved   │              │
│  │ Agent       │    │ (Query      │    │ Context     │              │
│  │ Workflow    │    │ Interpreter)│    │             │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

#### 1. **Ingestion Flow** (Left to Right)
```
Raw Data → Dataset Adapter → Document Validation → Chunking → Embedding Generation → Vector Store Upload
```

#### 2. **Retrieval Flow** (Right to Left)  
```
User Query → Query Interpretation → Vector Search → Reranking → Context Assembly → LLM Generation
```

#### 3. **Configuration Flow** (Top to Bottom)
```
Main config.yml → Custom Dataset Configs → Component Initialization → Runtime Parameters
```

---

## Component Deep Dive

### Core Directory Structure

```
/home/spiros/Desktop/Thesis/
├── pipelines/              # Core ingestion pipeline
│   ├── contracts.py        # Base schemas & interfaces  
│   ├── adapters/           # Dataset-specific adapters
│   ├── ingest/             # Core processing components
│   └── configs/            # Configuration files
├── retrievers/             # Modern retrieval implementations
├── components/             # Retrieval pipeline components
├── agent/                  # LangGraph agent system
├── database/               # Vector & traditional DB controllers
├── embedding/              # Embedding factories & processors
├── bin/                    # CLI tools & utilities
├── config/                 # Configuration management
└── docs/                   # Documentation
```

### Key Components Explained

#### **`pipelines/contracts.py`** - System Schemas
```python
# Core data structures that define the entire system
class BaseRow(BaseModel):          # Raw data interface
class ChunkMeta(BaseModel):        # Processed chunk with embeddings  
class IngestionRecord(BaseModel):  # Complete processing lineage
class DatasetAdapter(ABC):         # Dataset integration interface
```

**Purpose**: Provides type safety, data validation, and consistent interfaces across all components.

#### **`pipelines/adapters/`** - Dataset Integration
- **`stackoverflow.py`**: Processes Stack Overflow Q&A data
- **`energy_papers.py`**: Handles research papers
- **`natural_questions.py`**: Manages Q&A datasets

**Pattern**: Each adapter implements `DatasetAdapter` interface:
```python
def read_rows(self, split: DatasetSplit) -> Iterator[BaseRow]
def to_documents(self, rows: List[BaseRow]) -> List[Document]  
```

#### **`pipelines/ingest/`** - Core Processing Engine

**`pipeline.py`** - Main Orchestrator
- Coordinates all ingestion steps
- Handles error recovery and lineage tracking
- Supports dry-run and canary deployments

**`validator.py`** - Quality Assurance
- Content length validation
- HTML cleaning and sanitization  
- Duplicate detection
- Language filtering

**`chunker.py`** - Text Segmentation
- **Recursive Strategy**: Character-based splitting with hierarchy
- **Semantic Strategy**: Sentence-boundary aware
- **Code-Aware Strategy**: Preserves code blocks
- **Table-Aware Strategy**: Maintains table structure

**`embedder.py`** - Vector Generation
- Supports dense, sparse, and hybrid strategies
- Batch processing with progress tracking
- Automatic caching and error handling
- Multiple provider integration

**`uploader.py`** - Vector Store Management
- Idempotent uploads with versioning
- Collection creation and configuration
- Batch uploads with verification
- Canary deployment support

---

## Data Insertion Pipeline

### Complete Insertion Flow

#### Step 1: Configuration Loading
```yaml
# Example: stackoverflow_voyage_premium.yml
dataset:
  name: "stackoverflow_sosum"
  version: "v1.0.0"
  adapter: "stackoverflow"

embedding:
  strategy: "hybrid"
  dense:
    provider: "voyage"
    model: "voyage-3.5"
    dimensions: 1024
  sparse:
    provider: "sparse"
    model: "Qdrant/bm25"
```

#### Step 2: Data Reading & Validation
```python
# In IngestionPipeline.ingest_dataset()
adapter = StackOverflowAdapter(dataset_path)
rows = adapter.read_rows(split=DatasetSplit.ALL)
documents = adapter.to_documents(rows, split)

# Document validation
validator = DocumentValidator(config["validation"])
valid_docs = validator.validate_documents(documents)
```

#### Step 3: Document Chunking
```python
# ChunkingStrategyFactory creates appropriate chunker
chunker = ChunkingStrategyFactory.create_chunker(config["chunking"])
chunks = chunker.chunk_documents(valid_docs)

# Each chunk gets deterministic ID
chunk_id = f"{doc_id}#c{chunk_index:04d}"
```

#### Step 4: Embedding Generation
```python
# EmbeddingPipeline processes chunks
embedding_pipeline = EmbeddingPipeline(config)
chunk_metas = embedding_pipeline.process_documents(chunks)

# For hybrid strategy:
dense_embeddings = dense_embedder.embed_documents(texts)
sparse_embeddings = sparse_embedder.embed_documents(texts)
```

#### Step 5: Vector Store Upload
```python
# VectorStoreUploader handles Qdrant operations
uploader = VectorStoreUploader(config)
record = uploader.upload_chunks(chunk_metas)

# Creates/configures collection if needed
uploader._ensure_collection_exists(chunk_metas)
```

#### Step 6: Quality Assurance
```python
# Smoke tests verify successful upload
smoke_runner = SmokeTestRunner(config)
test_results = smoke_runner.run_smoke_tests(
    collection_name=collection, 
    chunk_metas=chunk_metas
)
```

#### Step 7: Lineage Recording
```python
# Complete processing history saved
lineage_data = {
    "ingestion_record": record.dict(),
    "config": config,
    "environment": {
        "git_commit": get_git_commit(),
        "python_version": get_python_version(),
        "timestamp": str(datetime.utcnow())
    }
}
```

### When Collections Are Created

**Collections are created during the upload phase** in `VectorStoreUploader._ensure_collection_exists()`:

```python
def _ensure_collection_exists(self, chunk_metas: List[ChunkMeta]):
    """Collection creation happens here"""
    if not client.collection_exists(collection_name):
        # Create with hybrid vector configuration
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
                "sparse": SparseVectorParams()
            }
        )
```

### Deterministic ID Generation

**Document IDs**: `{source}:{external_id}:{content_hash[:12]}`
**Chunk IDs**: `{doc_id}#c{chunk_index:04d}`

This ensures:
- Same content always gets same ID
- Reruns don't create duplicates
- Easy debugging and reproduction

---

## Data Retrieval Pipeline

### Retrieval Architecture

```
Query Input → Configurable Pipeline → Multiple Retrievers → Fusion → Reranking → Results
```

### Component Structure

#### **`components/retrieval_pipeline.py`** - Pipeline Framework
```python
class RetrievalPipeline:
    """Configurable pipeline with multiple stages"""
    def run(self, query: str, k: int = 5) -> List[RetrievalResult]
    
class BaseRetriever(RetrievalComponent):
    """Base class for all retrievers"""
    
class RerankerComponent(RetrievalComponent): 
    """Base class for reranking components"""
```

#### **`retrievers/`** - Modern Retriever Implementations

**Dense Retriever** (`dense_retriever.py`)
```python
class QdrantDenseRetriever(ModernBaseRetriever):
    """Semantic similarity search using dense vectors"""
    
    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        # Embed query
        query_vector = self.embedding.embed_query(query)
        
        # Direct Qdrant search
        search_result = qdrant_db.client.search(
            collection_name=collection_name,
            query_vector=NamedVector(name="dense", vector=query_vector),
            limit=k,
            with_payload=True
        )
        
        # Convert to RetrievalResult objects
        return self._create_retrieval_results(search_result)
```

**Sparse Retriever** (`sparse_retriever.py`)
```python
class QdrantSparseRetriever(ModernBaseRetriever):
    """Keyword-based search using sparse vectors (BM25)"""
    
    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        # Generate sparse query vector
        query_vector = self.embedding.embed_query(query)  # Returns dict
        
        # Search with sparse vectors
        search_result = qdrant_db.client.search(
            collection_name=collection_name,
            query_vector=NamedSparseVector(
                name="sparse",
                vector={
                    "indices": list(query_vector.keys()),
                    "values": list(query_vector.values())
                }
            ),
            limit=k
        )
```

**Hybrid Retriever** (`hybrid_retriever.py`)
```python
class QdrantHybridRetriever(ModernBaseRetriever):
    """Combines dense + sparse with Reciprocal Rank Fusion"""
    
    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        # Perform both searches
        dense_results = self._perform_dense_search(query, k)
        sparse_results = self._perform_sparse_search(query, k) 
        
        # Fuse using RRF (Reciprocal Rank Fusion)
        return self._fuse_results(dense_results, sparse_results, k)
    
    def _fuse_results(self, dense: List, sparse: List, k: int) -> List:
        """RRF Score = 1/(rank + k) for each result"""
        rrf_scores = {}
        for rank, result in enumerate(dense):
            doc_id = result.document.metadata.get('external_id')
            rrf_scores[doc_id] = 1.0 / (rank + 1 + self.rrf_k)
            
        for rank, result in enumerate(sparse):
            doc_id = result.document.metadata.get('external_id')
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += 1.0 / (rank + 1 + self.rrf_k)
```

### Retrieval Configuration

**Example**: `pipelines/configs/retrieval/modern_hybrid.yml`
```yaml
retrieval_pipeline:
  components:
    - type: retriever
      config:
        retriever_type: hybrid
        top_k: 20
        
    - type: score_filter  
      config:
        min_score: 0.01
        
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
        top_k: 10
```

### Search Process Flow

#### 1. Query Processing
```python
# In RetrievalPipeline.run()
query = "How to handle Python exceptions?"
k = 5
```

#### 2. Initial Retrieval
```python
# Retriever component (e.g., hybrid)
initial_results = retriever.retrieve(query, k=20)  # Get more for reranking
```

#### 3. Score Filtering
```python
# Filter low-score results
filtered_results = [r for r in initial_results if r.score >= 0.01]
```

#### 4. Reranking
```python
# CrossEncoder reranking
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
final_results = reranker.rerank(query, filtered_results, top_k=10)
```

#### 5. Result Assembly
```python
# Convert to final format
documents = [result.document for result in final_results]
context = "\n\n".join([doc.page_content for doc in documents])
```

---

## Agent Workflow System

### LangGraph Agent Architecture

```
User Query → Query Interpreter → [Retriever OR Direct Generator] → Response Generator → Memory Update
```

#### **`agent/graph.py`** - Main Agent Definition
```python
# Load configuration
config = load_config("config.yml")
retrieval_config_path = config["agent_retrieval"]["config_path"]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

# Create nodes
query_interpreter = make_query_interpreter(llm)
retriever = make_configurable_retriever(config_path=retrieval_config_path)
generator = make_generator(llm)

# Build workflow graph
builder = StateGraph(AgentState)
builder.add_node("query_interpreter", query_interpreter)
builder.add_node("retriever", retriever) 
builder.add_node("generator", generator)
builder.add_node("memory_updater", memory_updater)

# Define routing logic
builder.add_conditional_edges("query_interpreter", 
    lambda state: state["next_node"], 
    {
        "retriever": "retriever",
        "generator": "generator"
    }
)
```

### Agent Node Implementations

#### **Query Interpreter** (`agent/nodes/query_interpreter.py`)
```python
def make_query_interpreter(llm):
    def query_interpreter(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["question"]
        
        # LLM analyzes query intent
        prompt = f"""
        Decide if this query needs document retrieval or can be answered directly:
        Query: {query}
        
        Respond with JSON: {{"query_type": "text"|"none", "next_node": "retriever"|"generator"}}
        """
        
        response = llm.invoke(prompt)
        decision = json.loads(response.content)
        
        return {
            **state,
            "query_type": decision["query_type"],
            "next_node": decision["next_node"]
        }
```

#### **Configurable Retriever** (`agent/nodes/retriever.py`)
```python
def make_configurable_retriever(config_path: str):
    # Initialize retrieval agent with YAML config
    agent = ConfigurableRetrieverAgent(config_path)
    
    def retriever(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["question"]
        top_k = state.get("retrieval_top_k", 5)
        
        # Use configurable pipeline for retrieval
        docs_info = agent.retrieve(query, top_k=top_k)
        
        # Convert to context string
        context = "\n\n".join([doc["content"] for doc in docs_info])
        
        return {
            **state,
            "context": context,
            "retrieved_documents": docs_info,
            "retrieval_metadata": {
                "num_results": len(docs_info),
                "retrieval_method": docs_info[0]["retrieval_method"] if docs_info else "none"
            }
        }
```

#### **Response Generator** (`agent/nodes/generator.py`)
```python
def make_generator(llm):
    def generator(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["question"]
        context = state.get("context", "")
        
        if context:
            prompt = f"""
            Context: {context}
            
            Question: {query}
            
            Provide a comprehensive answer based on the context.
            """
        else:
            prompt = f"Question: {query}\n\nProvide a direct answer."
            
        response = llm.invoke(prompt)
        
        return {
            **state,
            "answer": response.content
        }
```

### Agent State Management

#### **`agent/schema.py`** - State Definition
```python
class AgentState(TypedDict):
    question: str                    # User query
    query_type: str                  # "text" or "none" 
    next_node: str                   # Routing decision
    context: str                     # Retrieved context
    retrieved_documents: List[Dict]  # Full document metadata
    retrieval_metadata: Dict         # Retrieval statistics
    answer: str                      # Final response
    chat_history: List[Dict]         # Conversation memory
    error: Optional[str]             # Error handling
```

### Agent Execution Flow

#### 1. **Query Analysis**
```python
state = {"question": "How to handle Python exceptions?"}
interpreted_state = query_interpreter(state)
# Result: {"next_node": "retriever", "query_type": "text"}
```

#### 2. **Conditional Routing**
```python
if interpreted_state["next_node"] == "retriever":
    # Need document retrieval
    retrieved_state = retriever(interpreted_state)
else:
    # Direct generation
    retrieved_state = {"context": ""}
```

#### 3. **Response Generation**
```python
final_state = generator(retrieved_state)
# Includes: answer, context, retrieval_metadata
```

#### 4. **Memory Update**
```python
updated_state = memory_updater(final_state)
# Updates chat_history for multi-turn conversations
```

---

## Configuration Management

### Configuration Hierarchy

```
1. Main config.yml (Global settings)
2. Custom dataset configs (Override specifics) 
3. Retrieval pipeline configs (Retrieval behavior)
4. Environment variables (Secrets & runtime)
```

#### **Main Configuration** (`config.yml`)
```yaml
# Global embeddings configuration
embedding:
  dense:
    provider: voyage
    model: voyage-3.5-lite
    dimensions: 1024
    api_key_env: VOYAGE_API_KEY
  sparse:
    provider: sparse
    model: Qdrant/bm25
  strategy: hybrid

# Vector database settings  
qdrant:
  collection: sosum_stackoverflow_hybrid_v1
  host: localhost
  port: 6333
  dense_vector_name: dense
  sparse_vector_name: sparse

# Agent configuration
agent_retrieval:
  config_path: pipelines/configs/retrieval/modern_hybrid.yml
  
# LLM settings
llm:
  model: gpt-4.1-mini
  provider: openai
  temperature: 0.0
```

#### **Dataset-Specific Config** (`pipelines/configs/datasets/stackoverflow_voyage_premium.yml`)
```yaml
# Override embedding settings for this dataset
embedding:
  strategy: hybrid
  dense:
    provider: voyage
    model: voyage-3.5          # Premium model
    dimensions: 1024
    batch_size: 32
    
# Dataset-specific chunking
chunking:
  strategy: recursive
  chunk_size: 512             # Smaller chunks for Q&A
  chunk_overlap: 50
  
# Custom collection name
qdrant:
  collection: sosum_stackoverflow_voyage_premium_v1
  
# Smoke test queries for this dataset
smoke_tests:
  golden_queries:
    - query: "Python list comprehension example"
      min_recall: 0.1
    - query: "JavaScript async function" 
      min_recall: 0.1
```

#### **Retrieval Pipeline Config** (`pipelines/configs/retrieval/modern_hybrid.yml`)
```yaml
retrieval_pipeline:
  default_retriever: hybrid
  components:
    - type: retriever
      config:
        retriever_type: hybrid
        top_k: 20
        fusion:
          rrf_k: 60
          dense_weight: 0.6
          sparse_weight: 0.4
          
    - type: score_filter
      config:
        min_score: 0.01
        
    - type: reranker  
      config:
        model_type: cross_encoder
        model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
        top_k: 10
```

### Configuration Loading Logic

#### **`config/config_loader.py`** - Configuration Management
```python
def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """Load main configuration file"""
    
def load_config_with_overrides(config_path: str, overrides: Dict) -> Dict[str, Any]:
    """Merge config with overrides using deep merge"""
    config = load_config(config_path)
    return _deep_merge(config, overrides)
    
def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge dictionaries"""
```

#### **How Custom Configs Work**
```python
# In bin/ingest.py
if args.config:
    # Load ONLY the custom config (no merging)
    config = load_config(args.config)
else:
    # Load main config.yml
    config = load_config()
```

**Key Insight**: When you specify `--config custom.yml`, it loads **only** that file. For merging behavior, you'd need to use `load_config_with_overrides()`.

---

## Vector Database Integration

### Qdrant Controller Architecture

#### **`database/qdrant_controller.py`** - Database Interface
```python
class QdrantVectorDB(BaseVectorDB):
    """Qdrant database controller with hybrid vector support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = config["qdrant"]["collection"]
        self.dense_vector_name = config["qdrant"]["dense_vector_name"]
        self.sparse_vector_name = config["qdrant"]["sparse_vector_name"]
    
    def insert_documents(self, documents: List[Document], 
                        dense_embedder: Embeddings, 
                        sparse_embedder: Embeddings):
        """Insert with both dense and sparse vectors"""
        
    def as_langchain_vectorstore(self, strategy: str) -> QdrantVectorStore:
        """Return LangChain-compatible interface"""
```

### Collection Structure

#### **Hybrid Collection Schema**
```python
# Collection configuration in Qdrant
vectors_config = {
    "dense": VectorParams(
        size=1024,                    # Voyage AI dimensions
        distance=Distance.COSINE      # Similarity metric
    ),
    "sparse": SparseVectorParams()    # BM25 sparse vectors
}

# Document payload structure
payload = {
    "page_content": "document text...",
    "metadata": {
        "external_id": "stackoverflow:123456:abc123",
        "source": "stackoverflow_sosum", 
        "split": "all",
        "chunk_index": 0,
        "labels": {
            "title": "Python Exception Handling",
            "tags": ["python", "exceptions"],
            "enhanced": True
        }
    }
}
```

### Vector Search Operations

#### **Dense Search** (Semantic Similarity)
```python
def dense_search(query: str, k: int) -> List[RetrievalResult]:
    # 1. Embed query
    query_vector = dense_embedder.embed_query(query)
    
    # 2. Search dense vectors
    results = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name="dense", vector=query_vector),
        limit=k,
        with_payload=True
    )
    
    # 3. Convert to results
    return [create_retrieval_result(r) for r in results]
```

#### **Sparse Search** (Keyword Matching)  
```python
def sparse_search(query: str, k: int) -> List[RetrievalResult]:
    # 1. Generate sparse vector (BM25)
    sparse_vector = sparse_embedder.embed_query(query)  # Returns dict
    
    # 2. Search sparse vectors
    results = client.search(
        collection_name=collection_name,
        query_vector=NamedSparseVector(
            name="sparse",
            vector={
                "indices": list(sparse_vector.keys()),
                "values": list(sparse_vector.values())
            }
        ),
        limit=k
    )
```

#### **Hybrid Search** (Fusion)
```python
def hybrid_search(query: str, k: int) -> List[RetrievalResult]:
    # 1. Perform both searches
    dense_results = dense_search(query, k)
    sparse_results = sparse_search(query, k)
    
    # 2. Apply Reciprocal Rank Fusion
    rrf_scores = {}
    
    # Dense contributions
    for rank, result in enumerate(dense_results):
        doc_id = result.document.metadata["external_id"]
        rrf_scores[doc_id] = 1.0 / (rank + 1 + 60)  # RRF constant = 60
    
    # Sparse contributions  
    for rank, result in enumerate(sparse_results):
        doc_id = result.document.metadata["external_id"]
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += 1.0 / (rank + 1 + 60)
        else:
            rrf_scores[doc_id] = 1.0 / (rank + 1 + 60)
    
    # 3. Sort by combined score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]
```

---

## Embedding Systems

### Embedding Factory Pattern

#### **`embedding/factory.py`** - Provider Abstraction
```python
def get_embedder(cfg: dict):
    """Factory to return embedder based on configuration"""
    provider = cfg.get("provider", "hf").lower()
    
    if provider == "voyage":
        model_name = cfg.get("model", "voyage-3.5-lite")
        api_key = os.getenv("VOYAGE_API_KEY")
        return VoyageAIEmbeddings(model=model_name, voyage_api_key=api_key)
        
    elif provider == "google":
        model_name = cfg.get("model", "models/embedding-001")
        api_key = os.getenv("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        
    elif provider == "hf":
        model_name = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
        
    elif provider == "sparse":
        model_name = cfg.get("model", "Qdrant/bm25")
        return SparseEmbedder(model_name=model_name)
```

### Supported Embedding Providers

#### **1. Voyage AI** (Premium Dense Embeddings)
```python
# Configuration
embedding:
  dense:
    provider: voyage
    model: voyage-3.5          # or voyage-3.5-lite
    dimensions: 1024           # Native dimension
    api_key_env: VOYAGE_API_KEY
    batch_size: 32

# Features:
# - High-quality semantic embeddings
# - Optimized for RAG applications  
# - Rate limits: 3 RPM (free), higher with billing
```

#### **2. Google Gemini** (Dense Embeddings)
```python
# Configuration  
embedding:
  dense:
    provider: google
    model: models/embedding-001
    dimensions: 768            # or full dimension
    api_key_env: GOOGLE_API_KEY

# Features:
# - Good semantic understanding
# - Configurable output dimensions
# - Generous free tier
```

#### **3. HuggingFace** (Local Dense Embeddings)
```python
# Configuration
embedding:
  dense:
    provider: hf
    model: sentence-transformers/all-MiniLM-L6-v2
    device: cuda              # or cpu

# Features:
# - No API costs
# - Local processing
# - Many model options
```

#### **4. Sparse/BM25** (Keyword Embeddings)
```python
# Configuration
embedding:
  sparse:
    provider: sparse
    model: Qdrant/bm25
    vector_name: sparse

# Features:
# - Keyword-based search
# - Fast and interpretable
# - Complements dense embeddings
```

### Embedding Processing Pipeline

#### **`pipelines/ingest/embedder.py`** - Processing Engine
```python
class EmbeddingPipeline:
    """Processes documents through embedding generation"""
    
    def process_documents(self, documents: List[Document]) -> List[ChunkMeta]:
        strategy = self.config.get("strategy", "dense")
        
        if strategy == "hybrid":
            return self._process_hybrid(documents)
        elif strategy == "dense":
            return self._process_dense(documents)
        elif strategy == "sparse":
            return self._process_sparse(documents)
    
    def _process_hybrid(self, documents: List[Document]) -> List[ChunkMeta]:
        """Generate both dense and sparse embeddings"""
        
        # Convert to text
        texts = [doc.page_content for doc in documents]
        
        # Generate dense embeddings
        dense_embeddings = self._batch_embed_dense(texts)
        
        # Generate sparse embeddings  
        sparse_embeddings = self._batch_embed_sparse(texts)
        
        # Create ChunkMeta objects
        chunk_metas = []
        for i, doc in enumerate(documents):
            chunk_meta = ChunkMeta(
                chunk_id=self._generate_chunk_id(doc),
                doc_id=self._generate_doc_id(doc),
                text=doc.page_content,
                dense_embedding=dense_embeddings[i],
                sparse_embedding=sparse_embeddings[i],
                metadata=doc.metadata
            )
            chunk_metas.append(chunk_meta)
            
        return chunk_metas
```

---

## Evaluation & Benchmarking

### Benchmarking Architecture

#### **`benchmarks/`** - Evaluation Framework
```python
# benchmarks/benchmarks_runner.py
class BenchmarkRunner:
    """Orchestrates benchmark evaluation"""
    
    def run_benchmark(self, scenario_config: Dict) -> BenchmarkResult:
        # 1. Load evaluation dataset
        adapter = self._get_adapter(scenario_config["dataset"])
        eval_queries = adapter.get_evaluation_queries()
        
        # 2. Initialize retrieval pipeline
        pipeline = self._create_pipeline(scenario_config["retrieval"])
        
        # 3. Run evaluation
        results = []
        for query_data in eval_queries:
            retrieved_docs = pipeline.run(query_data["query"])
            metrics = self._compute_metrics(retrieved_docs, query_data["expected"])
            results.append(metrics)
            
        # 4. Aggregate results
        return self._aggregate_results(results)
```

#### **Evaluation Metrics** (`benchmarks/benchmarks_metrics.py`)
```python
class RetrievalMetrics:
    """Standard retrieval evaluation metrics"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Precision@K = |relevant ∩ retrieved@k| / k"""
        retrieved_at_k = retrieved[:k]
        return len(set(retrieved_at_k) & set(relevant)) / k
    
    @staticmethod 
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Recall@K = |relevant ∩ retrieved@k| / |relevant|"""
        retrieved_at_k = retrieved[:k]
        return len(set(retrieved_at_k) & set(relevant)) / len(relevant)
    
    @staticmethod
    def mrr(retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> float:
        """Mean Reciprocal Rank"""
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            rr = 0.0
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

#### **Benchmark Scenarios** (`benchmark_scenarios/`)

**Dense Baseline** (`dense_baseline.yml`)
```yaml
name: "Dense Baseline"
description: "Pure dense retrieval with Google embeddings"

dataset:
  name: "stackoverflow_sosum"
  split: "test"

retrieval:
  strategy: "dense"
  top_k: 10
  embedding:
    provider: "google"
    model: "models/embedding-001"

evaluation:
  metrics: ["precision", "recall", "mrr", "ndcg"]
  k_values: [1, 5, 10]
```

**Hybrid Advanced** (`hybrid_advanced.yml`)  
```yaml
name: "Hybrid with Reranking"
description: "Hybrid retrieval + CrossEncoder reranking"

retrieval:
  strategy: "hybrid"
  components:
    - type: retriever
      config:
        retriever_type: hybrid
        top_k: 20
        
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
        top_k: 10
```

### Running Benchmarks

#### **Command Line Usage**
```bash
# Run single scenario
python benchmarks/run_benchmark_optimization.py \
  --scenario benchmark_scenarios/hybrid_advanced.yml \
  --output-dir results/

# Run multiple scenarios  
python benchmarks/run_real_benchmark.py \
  --scenarios benchmark_scenarios/ \
  --datasets stackoverflow,energy_papers \
  --output-dir results/
```

#### **Benchmark Results** 
```json
{
  "scenario": "hybrid_advanced",
  "dataset": "stackoverflow_sosum", 
  "metrics": {
    "precision@5": 0.78,
    "recall@5": 0.65,
    "mrr": 0.82,
    "ndcg@10": 0.75
  },
  "timing": {
    "avg_query_time": 0.45,
    "total_time": 120.3
  },
  "configuration": {
    "retrieval_strategy": "hybrid",
    "reranker": "cross-encoder"
  }
}
```

---

## CLI Tools & Utilities

### Primary CLI Tools

#### **`bin/ingest.py`** - Data Ingestion CLI
```bash
# Basic ingestion
python bin/ingest.py ingest stackoverflow datasets/sosum/ \
  --config pipelines/configs/datasets/stackoverflow_voyage_premium.yml

# Dry run for testing
python bin/ingest.py ingest stackoverflow datasets/sosum/ \
  --dry-run --max-docs 10 --verbose

# Canary deployment
python bin/ingest.py ingest stackoverflow datasets/sosum/ \
  --canary --verify

# Check pipeline status
python bin/ingest.py status

# Run evaluation
python bin/ingest.py evaluate stackoverflow datasets/sosum/ \
  --output-dir results/
```

**Command Structure**:
```
python bin/ingest.py [--config CONFIG] COMMAND [ARGS]

Commands:
  ingest           # Ingest single dataset
  batch-ingest     # Process multiple datasets  
  evaluate         # Run retrieval evaluation
  status           # Show collection status
  cleanup          # Remove canary collections
```

#### **`bin/qdrant_inspector.py`** - Database Inspection
```bash
# List all collections
python bin/qdrant_inspector.py list

# Inspect specific collection
python bin/qdrant_inspector.py inspect sosum_stackoverflow_hybrid_v1

# Search collection
python bin/qdrant_inspector.py search sosum_stackoverflow_hybrid_v1 \
  "Python exception handling" --limit 5

# Collection statistics  
python bin/qdrant_inspector.py stats sosum_stackoverflow_hybrid_v1
```

#### **`bin/agent_retriever.py`** - Retrieval Testing
```bash
# Test retrieval pipeline
python bin/agent_retriever.py

# Interactive mode with configuration switching
python -c "
from bin.agent_retriever import ConfigurableRetrieverAgent
agent = ConfigurableRetrieverAgent('pipelines/configs/retrieval/modern_hybrid.yml')
results = agent.retrieve('Python exceptions', top_k=5)
print(f'Found {len(results)} results')
"
```

#### **`bin/retrieval_pipeline.py`** - Direct Pipeline Usage
```bash
# Test retrieval pipeline directly
python bin/retrieval_pipeline.py \
  --config pipelines/configs/retrieval/modern_hybrid.yml \
  --query "How to handle Python exceptions?"
```

### Configuration Switching

#### **`bin/switch_agent_config.py`** - Runtime Configuration Changes
```bash
# Switch to different retrieval config
python bin/switch_agent_config.py fast_hybrid

# List available configurations
python bin/switch_agent_config.py --list

# Show current configuration
python bin/switch_agent_config.py --status
```

### Main Application Entry Point

#### **`main.py`** - Interactive Chat Interface
```bash
# Start interactive chat session
python main.py

# Example session:
You: How do I handle exceptions in Python?
---
Agent: Based on the retrieved context, here are the key ways to handle exceptions in Python:

1. **try-except blocks**: The fundamental exception handling mechanism...
[Full response with retrieved context]
---

You: exit
Goodbye!
```

---

## Error Handling & Monitoring

### Logging System

#### **`logs/utils/logger.py`** - Centralized Logging
```python
def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance"""
    logger = logging.getLogger(name)
    
    # File handler
    file_handler = logging.FileHandler(f"logs/{name}.log")
    file_handler.setFormatter(formatter)
    
    # Console handler  
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

#### **Log Files Generated**
```
logs/
├── agent.log              # Agent workflow logs
├── ingestion.log          # Data ingestion logs  
├── retrieval.log          # Retrieval pipeline logs
├── benchmark.log          # Evaluation logs
└── chat.log              # Interactive chat logs
```

### Error Recovery Mechanisms

#### **Ingestion Error Handling**
```python
# In IngestionPipeline.ingest_dataset()
try:
    # Main ingestion flow
    documents = self._read_and_validate_documents(adapter, split, record)
    chunks = self._chunk_documents(documents, record) 
    chunk_metas = self._process_chunks(chunks, record)
    
    if not dry_run:
        upload_record = self._upload_chunks(chunk_metas)
        
except Exception as e:
    logger.error(f"Ingestion failed: {e}")
    record.mark_complete()
    record.metadata = {"error": str(e)}
    self._save_lineage(record)  # Always save lineage
    raise
```

#### **Retrieval Error Handling**
```python
# In ModernBaseRetriever.retrieve()
try:
    results = self._perform_search(query, k)
    
    # Apply score filtering
    if self.score_threshold > 0:
        results = [r for r in results if r.score >= self.score_threshold]
        
    return results[:k]
    
except Exception as e:
    logger.error(f"Error during retrieval: {e}")
    return []  # Return empty results instead of crashing
```

#### **Agent Error Handling**  
```python
# In agent nodes
try:
    # Agent processing
    final_state = graph.invoke(state)
    answer = final_state.get("answer", "[No answer returned]")
    
except Exception as e:
    logger.error(f"Agent execution failed: {e}")
    return {
        **state,
        "answer": "I apologize, but I encountered an error processing your request.",
        "error": str(e)
    }
```

### Health Checks & Monitoring

#### **System Status Checks**
```python
# In bin/ingest.py status command
def cmd_status(args):
    """Check system health"""
    
    # Check Qdrant connection
    try:
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"✓ Qdrant: {len(collections.collections)} collections")
    except Exception as e:
        print(f"✗ Qdrant: Connection failed - {e}")
    
    # Check embedding providers
    try:
        # Test Voyage AI
        voyage_key = os.getenv("VOYAGE_API_KEY")
        print(f"✓ Voyage AI: {'Configured' if voyage_key else 'Missing key'}")
        
        # Test Google
        google_key = os.getenv("GOOGLE_API_KEY") 
        print(f"✓ Google: {'Configured' if google_key else 'Missing key'}")
        
    except Exception as e:
        print(f"✗ Embedding check failed: {e}")
```

#### **Smoke Tests** (`pipelines/ingest/smoke_tests.py`)
```python
class SmokeTestRunner:
    """Post-ingestion validation tests"""
    
    def run_smoke_tests(self, collection_name: str, chunk_metas: List[ChunkMeta]) -> List[SmokeTestResult]:
        tests = [
            self._test_collection_exists(collection_name),
            self._test_document_count(collection_name, len(chunk_metas)),
            self._test_vector_dimensions(collection_name),
            self._test_sample_retrieval(collection_name),
            self._test_golden_queries(collection_name)
        ]
        return tests
    
    def _test_golden_queries(self, collection_name: str) -> SmokeTestResult:
        """Test retrieval with known good queries"""
        golden_queries = self.config.get("smoke_tests", {}).get("golden_queries", [])
        
        passed = 0
        for query_config in golden_queries:
            query = query_config["query"]
            min_recall = query_config.get("min_recall", 0.1)
            
            # Perform retrieval
            results = self._perform_test_retrieval(collection_name, query)
            
            # Check if minimum recall achieved
            if len(results) >= min_recall * 10:  # Assuming top-10 search
                passed += 1
                
        return SmokeTestResult(
            test_name="golden_queries",
            passed=passed == len(golden_queries),
            details=f"{passed}/{len(golden_queries)} queries passed"
        )
```

---

## Extension Points

### Adding New Components

#### **1. New Dataset Adapter**
```python
# pipelines/adapters/my_dataset.py
class MyDatasetAdapter(DatasetAdapter):
    """Adapter for custom dataset format"""
    
    @property
    def source_name(self) -> str:
        return "my_dataset"
    
    def read_rows(self, split: DatasetSplit) -> Iterator[BaseRow]:
        """Read raw data files"""
        for file_path in self._get_files(split):
            with open(file_path) as f:
                data = json.load(f)
                for item in data:
                    yield MyDatasetRow(
                        external_id=item["id"],
                        content=item["text"],
                        metadata=item.get("meta", {})
                    )
    
    def to_documents(self, rows: List[BaseRow], split: DatasetSplit) -> List[Document]:
        """Convert to LangChain documents"""
        documents = []
        for row in rows:
            doc = Document(
                page_content=row.content,
                metadata={
                    "external_id": row.external_id,
                    "source": self.source_name,
                    "split": split.value,
                    **row.metadata
                }
            )
            documents.append(doc)
        return documents

# Register in bin/ingest.py get_adapter()
def get_adapter(adapter_type: str, dataset_path: str, version: str):
    if adapter_type == "my_dataset":
        return MyDatasetAdapter(dataset_path, version)
```

#### **2. New Embedding Provider**
```python
# embedding/factory.py - Add new provider
elif provider == "openai":
    model_name = cfg.get("model", "text-embedding-3-small")
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
```

#### **3. New Retrieval Component**
```python
# components/my_component.py
class MyReranker(RetrievalComponent):
    """Custom reranking component"""
    
    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        # Custom reranking logic
        reranked = self._custom_rerank(query, results)
        return reranked
        
    def _custom_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        # Implement custom ranking algorithm
        pass

# Register in components/retrieval_pipeline.py
COMPONENT_REGISTRY = {
    "retriever": ...,
    "reranker": ..., 
    "my_reranker": MyReranker
}
```

#### **4. New Chunking Strategy**
```python
# pipelines/ingest/chunker.py
class MyChunkingStrategy(ChunkingStrategy):
    """Custom chunking approach"""
    
    @property
    def strategy_name(self) -> str:
        return "my_strategy"
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            # Custom chunking logic
            doc_chunks = self._custom_chunk(doc)
            chunks.extend(doc_chunks)
        return chunks

# Register in ChunkingStrategyFactory.STRATEGIES
```

### Configuration Extension

#### **Custom Retrieval Pipeline**
```yaml
# pipelines/configs/retrieval/my_custom.yml
retrieval_pipeline:
  default_retriever: semantic
  components:
    - type: retriever
      config:
        retriever_type: semantic
        strategies:
          hybrid:
            enabled: true
            weight: 0.6
          dense:
            enabled: true
            weight: 0.4
        top_k: 15
        
    - type: my_reranker  
      config:
        algorithm: "custom"
        boost_factor: 1.2
        
    - type: score_filter
      config:
        min_score: 0.05
```

### Performance Optimization

#### **Caching Strategies**
```python
# Enable embedding caching
embedding_cache:
  enabled: true
  dir: "cache/embeddings/my_dataset"
  
# Enable pipeline caching  
retrieval_cache:
  enabled: true
  ttl: 3600  # 1 hour
```

#### **Batch Processing**
```python
# Optimize batch sizes
embedding:
  dense:
    batch_size: 64        # Larger batches for GPU
  sparse:  
    batch_size: 128       # Sparse can handle larger batches
    
upload:
  batch_size: 100         # Qdrant upload batching
```

#### **Hardware Optimization**
```python
# GPU acceleration
embedding:
  dense:
    device: "cuda"        # Use GPU for embeddings
    
# Parallel processing
processing:
  num_workers: 4          # Parallel document processing
  chunk_batch_size: 1000  # Process chunks in batches
```

---

## Summary: Complete System Understanding

### What Happens During Data Insertion

1. **Configuration Loading**: System loads main config + dataset-specific overrides
2. **Data Reading**: Dataset adapter reads raw files into standardized `BaseRow` objects
3. **Document Conversion**: Rows converted to LangChain `Document` objects with metadata
4. **Validation**: Documents checked for quality (length, language, duplicates)
5. **Chunking**: Documents split using configurable strategy (recursive, semantic, etc.)
6. **Embedding Generation**: Dense and/or sparse vectors created for each chunk
7. **ChunkMeta Creation**: Processed chunks with embeddings, IDs, and metadata
8. **Vector Store Upload**: ChunkMetas uploaded to Qdrant with hybrid indexing
9. **Smoke Testing**: Validation tests ensure successful ingestion
10. **Lineage Recording**: Complete processing history saved for reproducibility

### What Happens During Data Retrieval

1. **Query Input**: User provides natural language query
2. **Agent Interpretation**: LLM decides if retrieval is needed or direct answer suffices  
3. **Pipeline Initialization**: Configurable retrieval pipeline loaded from YAML
4. **Vector Search**: Query embedded and searched against dense/sparse/hybrid vectors
5. **Result Fusion**: Multiple search strategies combined using RRF or other methods
6. **Filtering**: Low-score results filtered out based on thresholds
7. **Reranking**: CrossEncoder or other rerankers improve result ordering
8. **Context Assembly**: Retrieved documents assembled into context string
9. **LLM Generation**: Context + query sent to LLM for final answer generation
10. **Response Delivery**: Structured response with metadata returned to user

### Individual Component Functions

- **`pipelines/contracts.py`**: Defines all data schemas and interfaces
- **`pipelines/adapters/`**: Dataset-specific readers that normalize different formats
- **`pipelines/ingest/`**: Core processing engine (validation, chunking, embedding, upload)
- **`retrievers/`**: Modern retrieval implementations (dense, sparse, hybrid, semantic)
- **`components/`**: Modular retrieval pipeline with configurable stages
- **`agent/`**: LangGraph workflow system with intelligent routing
- **`database/`**: Vector database abstraction with Qdrant implementation
- **`embedding/`**: Provider-agnostic embedding factory supporting multiple APIs
- **`config/`**: Hierarchical configuration system with YAML merging
- **`bin/`**: CLI tools for ingestion, inspection, testing, and administration

### System Strengths

✅ **Production Ready**: Comprehensive error handling, logging, and monitoring  
✅ **Highly Configurable**: YAML-based configuration with environment overrides  
✅ **Extensible**: Clean interfaces for adding datasets, embeddings, retrievers  
✅ **Theory-Backed**: Implements best practices (RRF fusion, deterministic IDs, etc.)  
✅ **Observable**: Complete lineage tracking and evaluation framework  
✅ **Scalable**: Batch processing, caching, and efficient vector operations  

This system represents a sophisticated, production-ready RAG implementation that balances flexibility, performance, and maintainability while providing comprehensive tooling for development, evaluation, and operations.
