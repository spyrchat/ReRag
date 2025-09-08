# Project Structure Documentation

This document describes the current organization of the RAG retrieval pipeline project after the cleanup and reorganization.

## 📁 Core Project Structure

### Main Application
```
├── main.py                     # Main application entry point
├── config.yml                  # Main configuration file
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

### Agent System
```
agent/
├── __init__.py
├── graph.py                    # LangGraph agent workflow
├── schema.py                   # Agent state schema
└── nodes/
    ├── retriever.py            # Configurable retriever node
    ├── generator.py            # Response generation node
    ├── query_interpreter.py    # Query analysis node
    └── memory_updater.py       # Conversation memory node
```

### Components (Modular Pipeline System)
```
components/
├── retrieval_pipeline.py      # Core pipeline framework
├── rerankers.py               # Reranking components
├── filters.py                 # Filtering components
└── advanced_rerankers.py      # Advanced reranking implementations
```

### Configuration Management
```
config/
├── __init__.py
└── config_loader.py           # Configuration loading utilities
```

### Database Controllers
```
database/
├── __init__.py
├── base.py                    # Base database interface
├── postgres_controller.py     # PostgreSQL controller
└── qdrant_controller.py       # Qdrant vector database controller
```

### Embedding System
```
embedding/
├── __init__.py
├── factory.py                 # Embedding factory
├── bedrock_embeddings.py      # AWS Bedrock embeddings
├── embeddings.py              # Core embedding utilities
├── processor.py              # Embedding processing
├── recursive_splitter.py     # Document splitting
├── sparse_embedder.py        # Sparse embeddings
├── splitter.py               # Text splitting utilities
└── utils.py                  # Embedding utilities
```

### Pipeline Configurations
```
pipelines/
├── README.md                  # Pipeline documentation
├── __init__.py
├── contracts.py              # Core pipeline contracts
├── configs/
│   └── retrieval/             # YAML retrieval configurations
│       ├── ci_google_gemini.yml
│       ├── fast_hybrid.yml
│       ├── modern_dense.yml
│       └── modern_hybrid.yml
├── adapters/                  # Data adapters
├── eval/                     # Evaluation components
└── ingest/                   # Ingestion pipelines
```

### CLI Tools
```
bin/
├── __init__.py
├── agent_retriever.py         # CLI agent retriever
├── ingest.py                  # Data ingestion utility
├── qdrant_inspector.py        # Qdrant inspection tool
├── retrieval_pipeline.py     # Direct pipeline usage
└── switch_agent_config.py     # Configuration switching utility
```

### Examples
```
# Note: Examples directory not present in current structure
# Usage examples are provided in documentation and test files
```

### Benchmarking System
```
benchmarks/
├── __init__.py
├── benchmark_contracts.py     # Benchmark interfaces
├── benchmark_optimizer.py     # Configuration optimization
├── benchmarks_adapters.py     # Dataset adapters for evaluation
├── benchmarks_metrics.py      # Evaluation metrics (Precision, Recall, NDCG)
├── benchmarks_runner.py       # Main benchmark orchestrator
├── run_benchmark_optimization.py # Optimization scripts
└── run_real_benchmark.py      # Real data benchmarking
```

### Benchmark Scenarios
```
benchmark_scenarios/
├── dense_baseline.yml         # Simple dense retrieval
├── dense_high_precision.yml   # High precision dense config
├── dense_high_recall.yml      # High recall dense config
├── hybrid_advanced.yml        # Advanced hybrid configuration
├── hybrid_reranking.yml       # Full reranking pipeline
├── hybrid_retrieval.yml       # Basic hybrid retrieval
├── hybrid_weighted.yml        # Weighted hybrid approach
├── quick_test.yml             # Quick performance test
└── sparse_bm25.yml           # BM25 baseline
```

### Additional Components
```
datasets/                      # Dataset storage
├── sosum/                    # SOSum Stack Overflow dataset

extraction_output/             # Table extraction results
├── *.csv                     # Extracted tables from documents

logs/                         # Application logs
├── agent.log                 # Agent workflow logs
├── query_interpreter.log    # Query processing logs
└── (other log files...)

playground/                   # Development and testing scripts
processors/                   # Legacy processing components
retrievers/                   # Base retriever implementations
scripts/                      # Utility scripts
```

## 🧪 Test Organization

All tests are now organized under the `tests/` directory with clear categorization:

### Test Structure
```
tests/
├── __init__.py
├── requirements-minimal.txt   # Minimal test dependencies
└── pipeline/                  # Pipeline component tests
    ├── __init__.py
    ├── run_tests.py           # Test runner
    ├── test_components.py     # Component integration tests
    ├── test_config.py         # Configuration validation tests
    ├── test_end_to_end.py     # End-to-end pipeline tests
    ├── test_minimal.py        # Minimal functionality tests
    ├── test_minimal_pipeline.py # CI-friendly minimal tests
    ├── test_qdrant.py         # Qdrant database tests
    ├── test_qdrant_connectivity.py # Database connectivity tests
    └── test_runner.py         # Test execution utilities
```
│   └── test_retriever_node.py
├── components/                # Component unit tests
│   ├── test_retrieval_pipeline.py
│   └── test_rerankers.py
├── retrieval/                 # Retrieval system tests
│   ├── test_extensibility.py
│   ├── test_modular_pipeline.py
│   ├── test_advanced_rerankers.py
│   └── test_answer_retrieval.py
├── ingestion/                 # Data ingestion tests
│   ├── test_new_adapter.py
│   └── test_adapter_qa.py
├── embedding/                 # Embedding system tests
│   └── test_sparse_embeddings.py
├── examples/                  # Example tests
│   ├── test_sosum_minimal.py
│   └── test_sosum_adapter.py
├── pipelines/                 # Pipeline tests
│   └── smoke_tests.py
└── benchmarks/                # Performance tests
    ├── retriever_test.py
    └── test_aws_connection.py
```

### Test Categories

1. **Unit Tests** (`tests/components/`): Test individual components in isolation
2. **Integration Tests** (`tests/agent/`, `tests/retrieval/`): Test component interactions
3. **System Tests** (`tests/examples/`, `tests/pipelines/`): End-to-end testing
4. **Performance Tests** (`tests/benchmarks/`): Performance and load testing

## 🗂️ Deprecated Code

All obsolete code has been moved to the `deprecated/` directory:

```
deprecated/
├── old_debug_scripts/         # Debug and analysis scripts
├── old_playground/            # Experimental code
├── old_processors/            # Legacy processor implementations
└── old_tests/                # Superseded test files
```

## 📋 Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Test Categories
```bash
# Component tests
python -m pytest tests/components/

# Agent tests
python -m pytest tests/agent/

# Retrieval tests
python -m pytest tests/retrieval/

# Integration tests
python tests/test_agent_retrieval.py
```

### Run Individual Tests
```bash
python tests/components/test_rerankers.py
python tests/agent/test_retriever_node.py
```

## 🔧 Configuration Management

### Pipeline Configurations
- **Location**: `pipelines/configs/retrieval/`
- **Format**: YAML files defining retrieval pipelines
- **Switching**: Use `bin/switch_agent_config.py`

### Environment Configuration
- **Main Config**: `config.yml`
- **Environment Variables**: `.env`
- **Loading**: Via `config/config_loader.py`

## 📚 Documentation

### User Guides
```
docs/
├── AGENT_INTEGRATION.md       # Agent integration guide
├── EXTENSIBILITY.md           # How to extend the system
├── SYSTEM_EXTENSION_GUIDE.md  # System extension guide
└── CODE_CLEANUP_SUMMARY.md    # Cleanup summary
```

### API Documentation
- Docstrings in all major components
- Type hints throughout codebase
- Configuration examples in YAML files

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env_example .env
   # Edit .env with your settings
   ```

3. **Run Basic Tests**:
   ```bash
   python tests/run_all_tests.py
   ```

4. **Start the Agent**:
   ```bash
   python main.py
   ```

## 🎯 Key Features

- **Modular Design**: Easy to add/remove components
- **YAML Configuration**: Flexible pipeline configuration
- **Comprehensive Testing**: Full test coverage
- **Clear Documentation**: Extensive guides and examples
- **Clean Architecture**: Well-organized codebase
- **Type Safety**: Full type hints
- **Extensible**: Easy to add new components
