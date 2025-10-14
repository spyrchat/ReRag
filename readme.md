# Advanced RAG System - Complete Documentation

**Version**: 2.0.0  
**Date**: October 2025  
**Author**: Spiros Chatzigeorgiou

> 🎯 **Production-ready Advanced RAG System** with hybrid retrieval, LangGraph agents, and comprehensive benchmarking framework.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 16GB+ RAM recommended
- API keys for embedding providers (Google, OpenAI, etc.)

### 1. Setup Environment
```bash
# Clone and enter repository
git clone <repository-url>
cd thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env_example .env
# Edit .env with your API keys
```

### 2. Start Infrastructure
```bash
# Start Qdrant vector database
docker-compose up -d qdrant

# Verify Qdrant is running
curl http://localhost:6333/health
```

### 3. Run Your First Pipeline
```bash
# Ingest Stack Overflow dataset
python bin/ingest.py ingest --config pipelines/configs/datasets/stackoverflow_hybrid.yml

# Test agent workflow (interactive mode)
python main.py

# Test agent workflow (single query)
python main.py --query "What are Python best practices?"
```

---

## 📚 Complete User Guide

### Step-by-Step Tutorial

#### Step 1: Environment Setup
First, ensure you have the required API keys:

1. **Google AI API Key**: Get from [Google AI Studio](https://aistudio.google.com/)
2. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/)
3. **Voyage AI API Key** (optional): Get from [Voyage AI](https://www.voyageai.com/)

Create your `.env` file:
```bash
# Copy the example file
cp .env_example .env

# Edit with your actual keys
nano .env
```

Add your API keys:
```env
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=sk-your_openai_api_key_here

# Optional API Keys
VOYAGE_API_KEY=your_voyage_api_key_here

# Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# System Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
```

#### Step 2: Download Sample Data
```bash
# Download and setup Stack Overflow dataset
./scripts/setup_sosum.sh

# Verify dataset is available
ls -la datasets/sosum/data/
```

#### Step 3: Run Data Ingestion
```bash
# Start with a dry run to test configuration
python bin/ingest.py ingest \
  --config pipelines/configs/datasets/stackoverflow_hybrid.yml \
  --dry-run \
  --max-docs 100

# If successful, run actual ingestion
python bin/ingest.py ingest \
  --config pipelines/configs/datasets/stackoverflow_hybrid.yml \
  --max-docs 1000

# Check ingestion status
python bin/ingest.py status
```

#### Step 4: Test Retrieval
```bash
# Run interactive retrieval demo
# This will test different configurations with example queries
python bin/agent_retriever.py
```

#### Step 5: Try the Agent
```bash
# Interactive chat mode
python main.py

# Or single query mode
python main.py --query "Explain Python decorators with examples"
```

#### Step 6: Run Benchmarks (Optional)
```bash
# Quick benchmark (no CLI flags - edit script to configure)
# Run benchmarks (see benchmarks/README.md for available experiments)
python -m benchmarks.experiment1 --output-dir results/my_experiment

# Run experiments with custom output directory
python -m benchmarks.experiment1 --output-dir results/my_experiment
```

---

## 📖 System Overview

### Architecture

This system implements a **modular RAG architecture** with clear separation of concerns:

```
┌───────────────────────────────────────────────────────────────────────┐
│                    Advanced RAG System                                │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  📊 DATA INGESTION                🔍 RETRIEVAL             🤖 AGENTS  │
│  ┌─────────────┐                  ┌─────────────┐          ┌──────────│
│  │ Adapters    │────────────────▶│ Dense       │────────▶│LangGraph │
│  │ (Dataset    │                  │ Sparse      │          │Workflows │
│  │ Specific)   │                  │ Hybrid      │          │          │
│  └─────────────┘                  └─────────────┘          └──────────│
│        │                                 │                        │   │
│        ▼                                 ▼                        ▼   │
│  ┌─────────────┐                  ┌─────────────┐         ┌──────────┐│
│  │ Validation  │                  │ Reranking   │         │Response  ││
│  │ Chunking    │                  │ Filtering   │         │Generation││
│  │ Embedding   │                  │             │         │          ││
│  └─────────────┘                  └─────────────┘         └──────────┘│
│        │                                                              │
│        ▼                                                              │
│  ┌────────────────────────────────────────────────────────────────────┤
│  │                    QDRANT VECTOR DATABASE                          │
│  │              (Dense + Sparse + Metadata Storage)                   │
│  └────────────────────────────────────────────────────────────────────┘
│                                                                       │
│  📈 EVALUATION & BENCHMARKING                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │ Metrics     │    │ Experiments │    │ Reports     │                │
│  │ (Recall,    │    │             |    │ (Analysis)  │                │
│  │ Precision)  │    │ Optimization│    │             │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
└───────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **[Pipelines](pipelines/)** | Data ingestion & processing | Adapters, validation, chunking, embedding |
| **[Database](database/)** | Vector storage abstraction | Qdrant integration, hybrid indexing |
| **[Embedding](embedding/)** | Vector generation | Multiple providers, caching, batching |
| **[Retrievers](retrievers/)** | Search & retrieval | Dense, sparse, hybrid strategies |
| **[Agent](agent/)** | AI workflow orchestration | LangGraph, query interpretation, response generation |
| **[Benchmarks](benchmarks/)** | Evaluation framework | Metrics, experiments, analysis |

---

## 🔧 Installation & Setup

### Development Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd thesis
   ```

2. **Python Environment**
   ```bash
   # Python 3.11+ required
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   ```bash
   cp .env_example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   # Vector Database
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION=my_collection
   
   # Embedding Providers
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   
   # Agent Configuration
   EMBEDDING_STRATEGY=hybrid
   LLM_PROVIDER=openai
   ```

4. **Start Infrastructure**
   ```bash
   # Start Qdrant database
   docker-compose up -d qdrant
   
   # Verify services
   curl http://localhost:6333/health
   ```

### Production Deployment

For production deployment, set environment variables directly instead of using `.env` files:

```bash
export QDRANT_HOST=your-qdrant-instance.com
export QDRANT_API_KEY=your-qdrant-api-key
export GOOGLE_API_KEY=your-google-api-key
# ... other environment variables
```

---

## 🏃‍♂️ Usage Guide

### 1. Data Ingestion

The ingestion pipeline processes datasets into vector embeddings stored in Qdrant.

#### Basic Ingestion
```bash
# Use existing dataset configuration
python bin/ingest.py ingest --config pipelines/configs/datasets/stackoverflow_hybrid.yml

# Custom configuration
python bin/ingest.py ingest --config my_custom_config.yml --max-docs 1000

# Dry run to test configuration
python bin/ingest.py ingest --config my_config.yml --dry-run

# Canary deployment for testing
python bin/ingest.py ingest --config my_config.yml --canary --verify
```

#### Supported Datasets
- **Stack Overflow (SOSum)**: Programming Q&A with code snippets  
- **Custom**: Bring your own data with adapters

#### Configuration Example
```yaml
# my_config.yml
dataset:
  name: "my_dataset"
  adapter: "pipelines.adapters.custom.MyAdapter"
  path: "data/my_documents/"

embedding:
  strategy: "hybrid"
  dense:
    provider: "google"
    model: "text-embedding-004"
  sparse:
    provider: "splade"
    model: "naver/splade-cocondenser-ensembledistil"

qdrant:
  collection: "my_collection"
  host: "localhost"
  port: 6333
```

### 2. Retrieval Testing

Test different retrieval strategies before using them in agents.

```bash
# Run interactive retrieval demo with various configurations
python bin/agent_retriever.py
```

The script will demonstrate:
- Dense-only retrieval
- Sparse-only retrieval  
- Hybrid retrieval with different alpha values
- Multiple example queries

To customize retrieval programmatically, use the `ConfigurableRetrieverAgent` class:

```python
from bin.agent_retriever import ConfigurableRetrieverAgent

# Initialize with a specific config
agent = ConfigurableRetrieverAgent(
    config_path='pipelines/configs/retrieval/advanced_reranked.yml'
)

# Retrieve documents
results = agent.retrieve(
    query="Python asyncio best practices",
    top_k=10
)
```

### 3. Agent Workflows

The agent system uses LangGraph for sophisticated query processing.

#### Simple Query
```bash
python main.py --query "What are the environmental benefits of wind energy?"
```

#### Interactive Mode
```bash
python main.py
```

#### Custom Agent Configuration
```bash
python main.py \
  --query "Compare solar vs wind energy efficiency"
```

### 4. Benchmarking & Evaluation

Run comprehensive evaluations

#### Grid Search Optimization
```bash
# Interactive benchmark optimizer (follow prompts)
python -m benchmarks.run_benchmark_optimization

# Or use the 2D grid search with CLI
python -m benchmarks.optimize_2d_grid_alpha_rrfk \
  --scenario-yaml benchmark_scenarios/your_scenario.yml \
  --dataset-path datasets/sosum/data \
  --n-folds 5 \
  --output-dir results/optimization
```

**Note:** `report_generator.py` is a helper class used by experiment scripts, not a standalone CLI tool. Reports are generated automatically by experiment scripts.

---

## 📁 Repository Structure

```
thesis/
├── 📖 readme.md                    # This file
├── 🐳 docker-compose.yml           # Infrastructure setup
├── ⚙️ config.yml                   # Main system configuration
├── 🚀 main.py                      # Agent workflow entry point
│
├── 📊 pipelines/                   # Data ingestion & processing
│   ├── 📖 README.md
│   ├── 🔌 adapters/               # Dataset-specific adapters
│   ├── ⚙️ configs/                # Dataset configurations
│   ├── 📥 ingest/                 # Core ingestion pipeline
│   └── 📈 eval/                   # Evaluation framework
│
├── 🗄️ database/                    # Vector database abstraction
│   ├── 📖 README.md
│   ├── base.py                    # Abstract interfaces
│   └── qdrant_controller.py       # Qdrant implementation
│
├── 🧠 embedding/                   # Embedding generation
│   ├── 📖 README.md
│   ├── factory.py                 # Provider factory
│   ├── base_embedder.py          # Abstract interfaces
│   └── providers/                 # Provider implementations
│
├── 🔍 retrievers/                  # Search & retrieval
│   ├── 📖 README.md
│   ├── base.py                    # Abstract interfaces
│   └── dense_retriever.py         # Dense/hybrid retrieval
│
├── 🤖 agent/                      # LangGraph agent workflows
│   ├── 📖 README.md
│   ├── graph.py                   # Agent workflow definition
│   ├── schema.py                  # Data models
│   └── nodes/                     # Agent node implementations
│
├── 📈 benchmarks/                 # Evaluation & experiments
│   ├── 📖 README.md
│   ├── benchmark_*.py             # Core benchmarking
│   ├── experiment*.py             # Specific experiments
│   └── statistical_analyzer.py    # Advanced analytics
│
├── 🛠️ scripts/                     # Utility scripts
│   ├── 📖 README.md
│   └── setup/                     # Setup and maintenance
│
├── 🧪 tests/                      # Comprehensive test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── pipeline/                  # End-to-end tests
│
└── 📊 results/                    # Generated outputs
    ├── benchmarks/                # Benchmark results
    ├── experiments/               # Experiment data
    └── reports/                   # Analysis reports
```

---

## 🎛️ Configuration

The system uses hierarchical YAML configuration with environment variable overrides.

### Main Configuration (`config.yml`)
```yaml
# Global system settings
system:
  log_level: "INFO"
  cache_dir: "cache/"
  output_dir: "output/"

# Default database settings
database:
  provider: "qdrant"
  host: "${QDRANT_HOST:localhost}"
  port: "${QDRANT_PORT:6333}"
  collection: "${QDRANT_COLLECTION:default_collection}"

# Default embedding settings
embedding:
  provider: "${EMBEDDING_PROVIDER:google}"
  strategy: "${EMBEDDING_STRATEGY:hybrid}"
  cache_enabled: true

# Agent configuration
agent:
  llm_provider: "${LLM_PROVIDER:openai}"
  model: "${LLM_MODEL:gpt-4}"
  temperature: 0.1
  max_tokens: 2000
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant database host | `localhost` |
| `QDRANT_PORT` | Qdrant database port | `6333` |
| `QDRANT_API_KEY` | Qdrant API key (for cloud) | `None` |
| `GOOGLE_API_KEY` | Google AI API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `VOYAGE_API_KEY` | Voyage AI API key | Optional |
| `EMBEDDING_STRATEGY` | Retrieval strategy | `hybrid` |
| `LLM_PROVIDER` | LLM provider | `openai` |

---

## 🔌 Extension Points

### Adding New Datasets

1. **Create an Adapter**
   ```python
   # pipelines/adapters/my_dataset.py
   from pipelines.contracts import BaseAdapter, Document
   
   class MyDatasetAdapter(BaseAdapter):
       def load_documents(self) -> List[Document]:
           # Your loading logic here
           return documents
   ```

2. **Create Configuration**
   ```yaml
   # pipelines/configs/datasets/my_dataset.yml
   dataset:
     name: "my_dataset"
     adapter: "pipelines.adapters.my_dataset.MyDatasetAdapter"
     path: "data/my_dataset/"
   ```

### Adding New Embedding Providers

1. **Implement Provider**
   ```python
   # embedding/my_provider.py
   from embedding.base_embedder import BaseEmbedder
   
   class MyEmbedder(BaseEmbedder):
       def embed_documents(self, texts: List[str]) -> List[List[float]]:
           # Your embedding logic here
           return embeddings
   ```

2. **Register in Factory**
   ```python
   # embedding/factory.py
   from .my_provider import MyEmbedder
   
   EMBEDDER_REGISTRY["my_provider"] = MyEmbedder
   ```

### Adding New Agent Nodes

1. **Create Node**
   ```python
   # agent/nodes/my_node.py
   from agent.schema import AgentState
   
   def my_custom_node(state: AgentState) -> AgentState:
       # Your node logic here
       return state
   ```

2. **Add to Graph**
   ```python
   # agent/graph.py
   from .nodes.my_node import my_custom_node
   
   graph.add_node("my_node", my_custom_node)
   ```

---

## 🌟 Complete Workflow Example

Here's a comprehensive example showing how to build a complete RAG system from scratch:

### 1. Setup & Configuration
```bash
# Initial setup
git clone <repository-url>
cd thesis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env_example .env
# Add your API keys to .env

# Start infrastructure
docker-compose up -d qdrant
```

### 2. Data Preparation
```bash
# Download sample dataset
./scripts/setup_sosum.sh

# Verify data structure
ls -la datasets/sosum/data/
head -n 3 datasets/sosum/data/train.jsonl
```

### 3. Custom Configuration
Create a custom configuration for your use case:

```yaml
# my_custom_config.yml
dataset:
  name: "my_stackoverflow_demo"
  version: "v1.0.0"
  adapter: "stackoverflow"
  path: "datasets/sosum/data"

chunking:
  strategy: "recursive"
  chunk_size: 500
  chunk_overlap: 100
  separators: ["\n\n", "\n", " ", ""]

embedding:
  strategy: "hybrid"
  dense:
    provider: "google"
    model: "text-embedding-004"
    dimensions: 768
    batch_size: 32
  sparse:
    provider: "sparse"
    model: "Qdrant/bm25"
    batch_size: 32

qdrant:
  collection: "my_demo_collection"
  dense_vector_name: "dense"
  sparse_vector_name: "sparse"

upload:
  batch_size: 50
  wait: true
  versioning: true

validation:
  min_char_length: 50
  max_char_length: 5000
  remove_duplicates: true
  clean_html: true
  preserve_code_blocks: true
  allowed_languages: ["en"]

smoke_tests:
  min_success_rate: 0.8
  golden_queries:
    - query: "Python function definition"
      min_recall: 0.1
    - query: "JavaScript error handling"
      min_recall: 0.1
```

### 4. Data Ingestion
```bash
# Test configuration with dry run
python bin/ingest.py ingest \
  --config my_custom_config.yml \
  --dry-run \
  --max-docs 100 \
  --verbose

# Run actual ingestion
python bin/ingest.py ingest \
  --config my_custom_config.yml \
  --max-docs 1000

# Verify ingestion
python bin/qdrant_inspector.py list
python bin/qdrant_inspector.py stats my_demo_collection
```

### 5. Retrieval Testing
```bash
# Test different retrieval strategies
python bin/agent_retriever.py \
  --query "How to handle Python exceptions?" \
  --top_k 5 \
  --collection my_demo_collection

# Test hybrid search
python bin/agent_retriever.py \
  --query "JavaScript async await best practices" \
  --strategy hybrid \
  --alpha 0.7 \
  --top_k 10
```

### 6. Agent Interaction
```bash
# Interactive chat
python main.py

# Single query
python main.py --query "Explain Python decorators with examples"
```

### 7. Performance Evaluation
```bash
# Quick benchmark (no CLI flags - edit script to configure)
# Run benchmarks (see benchmarks/README.md for available experiments)
python -m benchmarks.experiment1 --output-dir results/my_experiment

# Run experiments with output directory control
python -m benchmarks.experiment1 --output-dir results/exp_$(date +%Y%m%d)

# Advanced 2D grid optimization
python -m benchmarks.optimize_2d_grid_alpha_rrfk \
  --scenario-yaml benchmark_scenarios/your_scenario.yml \
  --dataset-path datasets/sosum/data \
  --n-folds 5 \
  --output-dir results/optimization_$(date +%Y%m%d)
```

### 8. Production Deployment
```bash
# Set production environment variables
export ENVIRONMENT=production
export QDRANT_HOST=your-production-qdrant.com
export QDRANT_API_KEY=your-production-api-key

# Run with production config
python bin/ingest.py ingest \
  --config production_config.yml \
  --verify

# Start production agent
python main.py
```

---

## 🎓 Learning Path

### For Beginners
1. **Start with Quick Start**: Follow the step-by-step tutorial
2. **Understand Architecture**: Read the system overview and component descriptions
3. **Try Examples**: Run the provided examples with sample data
4. **Read Component READMEs**: Deep dive into individual components

### For Developers
1. **Code Architecture**: Study the `pipelines/contracts.py` and core interfaces
2. **Extension Points**: Learn how to add custom adapters and components
3. **Configuration System**: Master the hierarchical configuration system
4. **Testing**: Run and understand the test suite

### For Researchers
1. **Benchmarking Framework**: Explore the evaluation and metrics system
2. **Experiments**: Run optimization experiments and statistical analysis
3. **Custom Metrics**: Implement domain-specific evaluation metrics
4. **Publication**: Use the analysis tools for research publication

### For System Administrators
1. **Deployment**: Learn Docker setup and production configuration
2. **Monitoring**: Understand logging and health checks
3. **Troubleshooting**: Master the debugging and error resolution
4. **Performance**: Optimize for your specific use case

---

## 🎯 Project Status & Roadmap

### Current Features (✅ Complete)
- ✅ **Multi-provider Embedding Support**: Google, OpenAI, Voyage, HuggingFace
- ✅ **Hybrid Retrieval**: Dense + Sparse with RRF fusion
- ✅ **LangGraph Agent System**: Sophisticated query interpretation
- ✅ **Comprehensive Benchmarking**: Statistical analysis and optimization
- ✅ **Production-Ready Pipeline**: Error handling, monitoring, lineage
- ✅ **Extensive Documentation**: Component-level guides and tutorials

### In Development (🚧 In Progress)
- 🚧 **Advanced Reranking**: Cross-encoder and LLM-based reranking
- 🚧 **Multi-modal Support**: Image and document embedding support
- 🚧 **Distributed Processing**: Horizontal scaling capabilities
- 🚧 **Real-time Updates**: Live document updates and incremental indexing

### Future Roadmap (🗺️ Planned)
- 🗺️ **Graph RAG**: Knowledge graph integration
- 🗺️ **Federated Search**: Multi-source retrieval aggregation
- 🗺️ **Advanced Analytics**: User behavior and query analysis
- 🗺️ **API Services**: REST/GraphQL API endpoints
- 🗺️ **Web Interface**: Interactive web dashboard

---

## 📈 Performance & Benchmarking

### Running Performance Tests

To measure actual performance characteristics of your deployment:

```bash
# Run comprehensive benchmark (no CLI flags - edit script to configure)
# Run benchmarks (see benchmarks/README.md for available experiments)
python -m benchmarks.experiment1 --output-dir results/my_experiment

# Run specific experiments with CLI flags
python -m benchmarks.experiment1 --test --output-dir results/exp1_test
python -m benchmarks.experiment3 --test --output-dir results/exp3_test

# Run 2D grid optimization for hybrid parameters
python -m benchmarks.optimize_2d_grid_alpha_rrfk \
  --scenario-yaml benchmark_scenarios/your_scenario.yml \
  --dataset-path datasets/sosum/data \
  --n-folds 5 \
  --max-queries-dev 100 \
  --output-dir results/optimization
```

### Metrics to Measure

The benchmarking framework supports measuring:

**Retrieval Quality Metrics:**
- Recall@K (proportion of relevant documents retrieved)
- Precision@K (proportion of retrieved documents that are relevant)
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)

**Performance Metrics:**
- Query latency (time per query)
- Throughput (queries per second)
- Ingestion speed (documents per minute)
- Memory usage
- Storage requirements

**Note:** Actual performance will vary based on:
- Dataset size and complexity
- Hardware specifications
- Embedding provider and model choice
- Retrieval strategy configuration (dense/sparse/hybrid)
- Network latency (for API-based embeddings)

### System Requirements

**Minimum Requirements:**
- Python 3.11+
- 8GB RAM (for development/testing)
- 10GB storage
- 2 CPU cores

**Recommended for Production:**
- 16GB+ RAM
- SSD storage (varies by dataset size)
- 4+ CPU cores
- Dedicated GPU (optional, for local embedding models)

Run your own benchmarks to determine optimal hardware for your specific use case.

---

## 📜 Acknowledgments

### Core Technologies
- **LangChain**: Document processing and LLM integration
- **Qdrant**: High-performance vector database
- **LangGraph**: Agent workflow orchestration
- **Pydantic**: Data validation and serialization
- **FastAPI**: API framework (future)

### Research & Inspiration
- **RAG Papers**: Lewis et al. (2020), Karpukhin et al. (2020)
- **Hybrid Retrieval**: Combining dense and sparse representations
- **Evaluation Frameworks**: BEIR, MS MARCO benchmarks
- **LLM Agents**: Plan-and-Execute patterns

---

**🚀 Ready to build the next generation of RAG systems?**

Start with our [Quick Start Guide](#-quick-start) and join the community of developers building intelligent information retrieval systems!

For questions, support, or contributions, please:
- 📧 Contact: [spyrchat@ece.auth.gr]

**Happy building! 🎉**
