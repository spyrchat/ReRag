# ReRag: a Reconfigurable Retrieval-Augmented-Generation Experimentation and Validation framework

**Version**: 2.0.0  
**Author**: Spiros Chatzigeorgiou

> Production-ready Retrieval-Augmented Generation (RAG) system with hybrid retrieval, Self-RAG agent workflows, cross-encoder reranking, and comprehensive benchmarking.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 16GB+ RAM recommended
- API keys: Google AI, OpenAI (optional: Voyage AI)

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd ReRag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env_example .env
# Edit .env and add your API keys:
#   GOOGLE_API_KEY=your_key_here
#   OPENAI_API_KEY=your_key_here
```

### 2. Start Vector Database
```bash
# Start Qdrant
docker-compose up -d

# Verify it's running
curl http://localhost:6333/healthz

#You can see the ingestion results in Qdrant's Web UI visiting the link below:
http://localhost:6333/dashboard#/collections
```

### 3. Run Your First Pipeline
```bash
#First download the dataset from the scripts folder
# Ingest documents (requires dataset - see Data Ingestion section)
python bin/ingest.py ingest --config pipelines/configs/datasets/stackoverflow_hybrid.yml

# Run agent in interactive mode
python main.py

# Run agent with single query
python main.py --query "What are Python best practices?"

# Run Self-RAG mode (with iterative refinement)
python main.py --mode self-rag --query "Explain how asyncio works"
```

---

## 📚 User Guide

### Data Ingestion

Ingest documents into the vector database:

```bash
# Basic ingestion from config
python bin/ingest.py ingest --config pipelines/configs/datasets/stackoverflow_hybrid.yml

# Test with dry run (no upload)
python bin/ingest.py ingest --config my_config.yml --dry-run --max-docs 100

# Check ingestion status
python bin/ingest.py status

# Cleanup canary collections
python bin/ingest.py cleanup
```

**Configuration File Format** (`pipelines/configs/datasets/*.yml`):
```yaml
dataset:
  name: "my_dataset"
  adapter: "stackoverflow"  # or full path: "pipelines.adapters.custom.MyAdapter"
  path: "datasets/sosum/data"

embedding:
  strategy: "hybrid"  # or "dense" or "sparse"
  dense:
    provider: "google"
    model: "text-embedding-004"
  sparse:
    provider: "sparse"
    model: "Qdrant/bm25"

qdrant:
  collection: "my_collection"
  host: "localhost"
  port: 6333
```

### Retrieval Testing

Test retrieval pipelines before using in agents:

```bash
# Use any retrieval configuration
python bin/retrieval_pipeline.py \
  --config pipelines/configs/retrieval/basic_dense.yml \
  --query "How to handle Python exceptions?" \
  --top-k 5
```

### Agent Workflows

Run the RAG agent with two available modes:

```bash
# Standard RAG mode (single-pass)
python main.py --query "Explain Python decorators"

# Self-RAG mode (iterative refinement with verification)
python main.py --mode self-rag --query "How does asyncio work?"

# Interactive chat
python main.py
# or
python main.py --mode self-rag
```

### Benchmarking

Run evaluation experiments:

```bash
# Run experiment with output directory
python -m benchmarks.experiment1 --output-dir results/exp1

# Run 2D grid optimization for hybrid search parameters
python -m benchmarks.optimize_2d_grid_alpha_rrfk \
  --scenario-yaml benchmark_scenarios/your_scenario.yml \
  --dataset-path datasets/sosum/data \
  --n-folds 5 \
  --output-dir results/optimization

# Generate ground truth for evaluation
python -m benchmarks.generate_ground_truth \
  --queries-file queries.json \
  --output-file ground_truth.json
```

See `benchmarks/README.md` for detailed documentation.

---

## 📖 System Architecture

### Overview

Modular RAG system with three main subsystems:

```
┌────────────────────────────────────────────────────────────┐
│                     RAG System                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  📊 INGESTION → 🔍 RETRIEVAL → 🤖 AGENT                    │
│                                                            │
│  Documents      Vector Search    LangGraph                 │
│  Chunking       Reranking        Response Gen              │
│  Embedding      Filtering        Verification              │
│  ↓                ↓                 ↓                      │
│  └───────────→ Qdrant ←───────────┘                        │
│                                                            │
│  📈 BENCHMARKS: Evaluation & Optimization                  │
└────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| **[pipelines/](pipelines/)** | Data ingestion & processing | [README](pipelines/README.md) |
| **[components/](components/)** | Retrieval pipeline (filters, rerankers) | [README](components/README.md) |
| **[embedding/](embedding/)** | Multi-provider embeddings | [README](embedding/README.md) |
| **[retrievers/](retrievers/)** | Dense/sparse/hybrid search | [README](retrievers/README.md) |
| **[agent/](agent/)** | LangGraph workflows (Standard + Self-RAG) | [README](agent/README.md) |
| **[database/](database/)** | Qdrant vector database interface | [README](database/README.md) |
| **[benchmarks/](benchmarks/)** | Evaluation framework | [README](benchmarks/README.md) |
| **[config/](config/)** | Configuration system | - |

---

## 🔧 Installation

### 1. Python Environment
```bash
# Clone repository
git clone <repository-url>
cd Thesis

# Create virtual environment (Python 3.11+ required)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys
```bash
# Create environment file
cp .env_example .env
```

Edit `.env` and add your API keys:
```env
# Required
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Optional
VOYAGE_API_KEY=your_key_here
```

### 3. Start Vector Database
```bash
# Start Qdrant using Docker
docker-compose up -d

# Verify it's running
curl http://localhost:6333/health
```



---

## 📁 Project Structure

```
Thesis/
├── readme.md                      # This file
├── main.py                        # Agent entry point (Standard & Self-RAG modes)
├── config.yml                     # Main configuration file
├── docker-compose.yml             # Qdrant database setup
├── requirements.txt               # Python dependencies
│
├── agent/                         # LangGraph agent workflows
│   ├── graph_refined.py          # Standard RAG workflow
│   ├── graph_self_rag.py         # Self-RAG workflow (iterative refinement)
│   ├── schema.py                 # State definitions
│   └── nodes/                    # Agent nodes (retriever, generator, grader)
│
├── pipelines/                     # Data ingestion
│   ├── adapters/                 # Dataset adapters (StackOverflow, custom)
│   ├── ingest/                   # Ingestion pipeline core
│   ├── eval/                     # Retrieval evaluator
│   └── configs/                  # Dataset configurations
│       └── datasets/             # Per-dataset configs
│
├── components/                    # Retrieval pipeline components
│   ├── retrieval_pipeline.py    # Pipeline orchestration
│   ├── rerankers.py             # CrossEncoder, Semantic, ColBERT, MultiStage
│   ├── filters.py               # Tag, duplicate, relevance filters
│   └── post_processors.py       # Result enhancement & limiting
│
├── retrievers/                    # Core retrieval implementations
│   ├── dense_retriever.py       # Dense/sparse/hybrid retrieval
│   └── base.py                  # Abstract interfaces
│
├── embedding/                     # Embedding providers
│   ├── factory.py               # Provider factory
│   ├── providers/               # Google, OpenAI, Voyage, HuggingFace
│   └── base_embedder.py        # Abstract interfaces
│
├── database/                      # Vector database
│   ├── qdrant_controller.py    # Qdrant integration
│   └── base.py                  # Abstract interfaces
│
├── config/                        # Configuration system
│   ├── config_loader.py         # YAML config loader
│   └── llm_factory.py           # LLM provider factory
│
├── benchmarks/                    # Evaluation framework
│   ├── experiment1.py           # Main experiment runner
│   ├── optimize_2d_grid_alpha_rrfk.py  # Grid search optimization
│   ├── llm_as_judge_eval.py     # LLM-based evaluation
│   ├── generate_ground_truth.py # Ground truth generation
│   ├── benchmarks_runner.py     # Core benchmark runner
│   ├── benchmarks_metrics.py    # Metrics (Recall, Precision, MRR, NDCG)
│   ├── report_generator.py      # Report generation (used by experiments)
│   └── statistical_analyzer.py  # Statistical analysis
│
├── bin/                          # CLI tools
│   ├── ingest.py                # Ingestion CLI
│   ├── retrieval_pipeline.py   # Retrieval testing CLI
│   ├── qdrant_inspector.py     # Database inspection
│   └── switch_agent_config.py  # Config switcher
│
├── logs/                         # Application logs
│   ├── agent.log                # Main agent log
│   ├── ingestion.log            # Ingestion log
│   └── utils/logger.py          # Custom logger
│
└── tests/                        # Test suite
    ├── test_self_rag_integration.py  # Self-RAG integration tests
    └── [other test files]
```

---

## ⚙️ Configuration

### Configuration Files

**Main Config** (`config.yml`):
- System-wide settings
- Loaded by `config/config_loader.py`

**Pipeline Configs** (`pipelines/configs/`):
- `datasets/` - Dataset-specific configs (ingestion)
- `retrieval/` - Retrieval pipeline configs

**Example: Ingestion Config**
```yaml
dataset:
  name: "stackoverflow"
  adapter: "stackoverflow"  # or full path
  path: "datasets/sosum/data"

embedding:
  strategy: "hybrid"  # dense, sparse, or hybrid
  dense:
    provider: "google"
    model: "text-embedding-004"
  sparse:
    provider: "sparse"
    model: "Qdrant/bm25"

qdrant:
  collection: "my_collection"
  host: "localhost"
  port: 6333
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key | No |
| `OPENAI_API_KEY` | OpenAI API key | No |
| `VOYAGE_API_KEY` | Voyage AI API key | No |

---

## 🔌 Extension Points

### Add Custom Dataset Adapter

1. Create adapter class:
   ```python
   # pipelines/adapters/my_adapter.py
   from pipelines.contracts import BaseAdapter, Document
   
   class MyAdapter(BaseAdapter):
       def load_documents(self) -> List[Document]:
           # Load your data
           return documents
   ```

2. Use in config:
   ```yaml
   dataset:
     adapter: "pipelines.adapters.my_adapter.MyAdapter"
     path: "path/to/data"
   ```

### Add Custom Reranker

Implement in `components/rerankers.py` or `components/advanced_rerankers.py`:
```python
from components.rerankers import BaseReranker

class MyReranker(BaseReranker):
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        # Your reranking logic
        return reranked_results
```

### Add Custom Agent Node

1. Create node in `agent/nodes/`:
   ```python
   from agent.schema import AgentState
   
   def my_node(state: AgentState) -> AgentState:
       # Process state
       return state
   ```

2. Add to graph in `agent/graph_refined.py` or `agent/graph_self_rag.py`

---

## 🎯 Key Features

### Retrieval Strategies
- **Dense Retrieval**: Semantic search using embeddings (Google, OpenAI, Voyage, HuggingFace)
- **Sparse Retrieval**: BM25-style keyword matching (Qdrant/bm25, SPLADE)
- **Hybrid Retrieval**: Combines dense + sparse with RRF (Reciprocal Rank Fusion)

### Reranking
- **Cross-Encoder**: ms-marco-MiniLM-L-6-v2 (default)
- **Semantic**: Sentence transformers for semantic similarity
- **ColBERT**: Token-level contextual matching
- **Multi-Stage**: Cascading rerankers for efficiency

### Agent Modes
- **Standard RAG**: Single-pass retrieval → generation
- **Self-RAG**: Iterative refinement with hallucination detection and context verification

### Benchmarking
- **Metrics**: Recall@K, Precision@K, MRR, NDCG@K
- **Optimization**: Grid search for hybrid parameters (alpha, RRF-k)
- **LLM-as-Judge**: Automated quality evaluation (faithfulness, relevance, helpfulness)
- **Statistical Analysis**: Cross-validation, significance testing

---

## 📊 Testing

### Run Integration Tests
```bash
# Self-RAG integration tests
pytest tests/test_self_rag_integration.py -v

# All tests
pytest tests/ -v
```

### Verify Components
See `components/LOGGING_GUIDE.md` for how to verify rerankers and filters are working correctly via logs.

---

## 🔍 CLI Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `bin/ingest.py` | Ingest datasets | `python bin/ingest.py ingest --config my_config.yml` |
| `bin/retrieval_pipeline.py` | Test retrieval | `python bin/retrieval_pipeline.py --config config.yml --query "test"` |
| `bin/qdrant_inspector.py` | Inspect database | `python bin/qdrant_inspector.py list` |
| `bin/switch_agent_config.py` | Switch configs | `python bin/switch_agent_config.py` |

---

## 📈 System Requirements

**Minimum:**
- Python 3.11+
- 8GB RAM
- 10GB storage

**Recommended:**
- 16GB+ RAM
- SSD storage
- 4+ CPU cores

---

## 📚 Documentation

- **Main README**: This file
- **Components**: `components/README.md` - Retrieval pipeline components
- **Pipelines**: `pipelines/README.md` - Data ingestion system
- **Benchmarks**: `benchmarks/README.md` - Evaluation framework
- **Agent**: `agent/README.md` - LangGraph workflows
- **CLI Reference**: `CLI_REFERENCE.md` - Command-line tools
- **Logging Guide**: `components/LOGGING_GUIDE.md` - Verify components work

---

## 🛠️ Technologies

- **LangGraph**: Agent workflow orchestration
- **Qdrant**: Vector database
- **LangChain**: Document processing
- **Sentence Transformers**: Embeddings and reranking
- **Pydantic**: Data validation

---

## 📧 Contact

**Author**: Spiros Chatzigeorgiou  
**Email**: spyrchat@ece.auth.gr

---

**Built for production RAG workflows with hybrid retrieval, advanced reranking, and comprehensive evaluation.**
