# Advanced RAG System with LangGraph Agent & Benchmarking

A production-ready, configurable RAG (Retrieval-Augmented Generation) system featuring LangGraph agent workflows, modular retrieval pipelines, and comprehensive benchmarking capabilities.

## Key Features

- **🤖 LangGraph Agent**: Intelligent agent workflows with configurable retrieval
- **⚙️ YAML-Configurable Pipelines**: Switch retrieval strategies without code changes  
- **🔄 Hybrid Retrieval**: Dense, sparse, and hybrid retrieval methods with RRF fusion
- **🎯 Advanced Reranking**: CrossEncoder, BGE, and multi-stage reranking
- **📊 Comprehensive Benchmarking**: Built-in evaluation framework with multiple metrics
- **🗄️ Vector Database**: Qdrant integration with optimized indexing
- **🔧 Modular Architecture**: Easily extensible components and filters
- **📈 Performance Monitoring**: Rich metadata, logging, and health checks

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LangGraph     │────│  Configurable    │────│   Retrieval     │────│   Benchmarking  │
│     Agent       │    │  Retriever Agent │    │   Pipeline      │    │    Framework    │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       │                                │                                │
                 ┌─────▼─────┐                 ┌───────▼────────┐                ┌─────▼─────┐
                 │ Retrievers│                 │   Rerankers    │                │  Filters  │
                 │           │                 │                │                │           │
                 │• Dense    │                 │• CrossEncoder  │                │• Score    │
                 │• Sparse   │                 │• BGE Reranker  │                │• Content  │
                 │• Hybrid   │                 │• Multi-stage   │                │• Custom   │
                 │• RRF      │                 │• Adaptive      │                │• Metadata │
                 └───────────┘                 └────────────────┘                └───────────┘
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_google_api_key
# OPENAI_API_KEY=your_openai_api_key
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
```

### 2. Start Vector Database

```bash
# Using Docker Compose (recommended)
docker-compose up -d qdrant

# Or run Qdrant directly
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Interactive Chat with Agent

```bash
# Start the interactive chat agent
python main.py
```

Example conversation:
```
You: How to handle exceptions in Python?
Agent: Python provides several mechanisms for exception handling...
```

### 4. Configuration Management

```bash
# List available retrieval configurations
python bin/switch_agent_config.py --list

# Switch to different retrieval strategy
python bin/switch_agent_config.py modern_hybrid

# Test the new configuration
python -c "
from agent.graph import graph
result = graph.invoke({'question': 'test query'})
print(result['answer'])
"
```

## Available Configurations

| Configuration | Description | Retrieval Method | Performance | Use Case |
|---------------|-------------|------------------|-------------|----------|
| `ci_google_gemini` | CI/CD optimized | Dense only | Fast | Testing, CI |
| `fast_hybrid` | Speed optimized | Hybrid + RRF | Very Fast | Production chat |
| `modern_dense` | Dense semantic | Dense + Reranking | Medium | Semantic search |
| `modern_hybrid` | Best quality | Hybrid + CrossEncoder | Slower | Research, Q&A |

### Benchmark Scenarios

| Scenario | Focus | Components | Metrics |
|----------|-------|------------|---------|
| `dense_baseline` | Simple dense retrieval | Google embeddings | Precision@K, Recall@K |
| `hybrid_retrieval` | Dense + sparse fusion | RRF fusion | MRR, NDCG |
| `hybrid_reranking` | Full reranking pipeline | CrossEncoder + filters | F1, MAP |
| `sparse_bm25` | Traditional IR | BM25 only | Baseline metrics |

## Configuration Examples

### Fast Hybrid Configuration
```yaml
# pipelines/configs/retrieval/fast_hybrid.yml
description: "Fast hybrid retrieval optimized for agent response speed"

retrieval_pipeline:
  retriever:
    type: hybrid
    top_k: 10
    score_threshold: 0.05
    fusion_method: rrf
    
    fusion:
      method: rrf
      rrf_k: 50
      dense_weight: 0.8
      sparse_weight: 0.2
    
    embedding:
      strategy: hybrid
      dense:
        provider: google
        model: models/embedding-001
        dimensions: 768
        api_key_env: GOOGLE_API_KEY
```

### Modern Dense with Reranking
```yaml
# pipelines/configs/retrieval/modern_dense.yml
description: "Dense semantic retrieval with Google embeddings and neural reranking"

retrieval_pipeline:
  retriever:
    type: dense
    top_k: 15
    score_threshold: 0.0
    
  stages:
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_k: 10
        
    - type: filter
      config:
        type: score
        min_score: 0.3
```

## Project Structure

```
Thesis/
├── main.py                    # Interactive chat application
├── config.yml                 # Main configuration file
├── docker-compose.yml         # Docker services (Qdrant, PostgreSQL)
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
│
├── agent/                     # LangGraph Agent System
│   ├── graph.py              # Main agent workflow graph
│   ├── schema.py             # Agent state definitions
│   └── nodes/                # Agent workflow nodes
│       ├── retriever.py      # Configurable retriever node
│       ├── generator.py      # Response generation node
│       ├── query_interpreter.py # Query analysis node
│       └── memory_updater.py # Conversation memory node
│
├── components/               # Modular Retrieval Components
│   ├── retrieval_pipeline.py # Core pipeline framework
│   ├── rerankers.py          # CrossEncoder, BGE rerankers
│   ├── advanced_rerankers.py # Multi-stage reranking
│   └── filters.py            # Score, content, metadata filters
│
├── pipelines/                # Data Pipelines & Configurations
│   ├── configs/
│   │   └── retrieval/        # YAML retrieval configurations
│   │       ├── fast_hybrid.yml
│   │       ├── modern_dense.yml
│   │       ├── modern_hybrid.yml
│   │       └── ci_google_gemini.yml
│   ├── adapters/             # Dataset adapters (BEIR, custom)
│   └── ingest/               # Data ingestion pipeline
│
├── benchmarks/               # Evaluation Framework
│   ├── benchmarks_runner.py  # Main benchmark orchestrator
│   ├── benchmarks_metrics.py # Precision, Recall, NDCG, MRR
│   ├── benchmarks_adapters.py # Dataset adapters for evaluation
│   └── run_real_benchmark.py # Real data benchmarking
│
├── benchmark_scenarios/      # Predefined Benchmark Configurations
│   ├── dense_baseline.yml    # Simple dense retrieval
│   ├── hybrid_retrieval.yml  # Hybrid dense+sparse
│   ├── hybrid_reranking.yml  # Full reranking pipeline
│   └── sparse_bm25.yml       # BM25 baseline
│
├── bin/                      # Command-line Utilities
│   ├── switch_agent_config.py # Configuration management
│   ├── agent_retriever.py    # Standalone retriever CLI
│   ├── qdrant_inspector.py   # Database inspection tool
│   └── ingest.py             # Data ingestion utility
│
├── database/                 # Database Controllers
│   ├── qdrant_controller.py  # Vector database operations
│   └── postgres_controller.py # Relational database operations
│
├── embedding/                # Embedding & Text Processing
│   ├── factory.py            # Embedding provider factory
│   ├── bedrock_embeddings.py # AWS Bedrock embeddings
│   ├── sparse_embedder.py    # BM25 sparse embeddings
│   └── processor.py          # Text processing utilities
│
├── tests/                    # Comprehensive Test Suite
│   ├── pipeline/             # Pipeline component tests
│   │   ├── test_minimal_pipeline.py
│   │   ├── test_qdrant_connectivity.py
│   │   └── test_end_to_end.py
│   └── requirements-minimal.txt
│
├── docs/                     # Documentation
│   ├── PROJECT_STRUCTURE.md  # Detailed project structure
│   ├── QUICK_START_GUIDE.md  # Getting started guide
│   └── SOSUM_INGESTION.md    # Dataset ingestion guide
│
└── logs/                     # Application Logs
    ├── agent.log             # Agent workflow logs
    └── query_interpreter.log # Query processing logs
```

## Benchmarking & Evaluation

### Running Benchmarks

```bash
# Run comprehensive benchmark with real StackOverflow data
python benchmarks/run_real_benchmark.py

# Run specific benchmark scenario
python benchmarks/run_benchmark_optimization.py --scenario hybrid_reranking

# Quick performance test
python benchmarks/run_benchmark_optimization.py --scenario quick_test
```

### Available Metrics

- **Precision@K**: Fraction of relevant documents in top-K results
- **Recall@K**: Fraction of relevant documents retrieved in top-K
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **F1 Score**: Harmonic mean of precision and recall
- **MAP (Mean Average Precision)**: Mean of precision scores at each relevant document

### Custom Benchmark Configuration

```yaml
# benchmark_scenarios/custom_scenario.yml
scenario_name: "custom_hybrid"
description: "Custom hybrid retrieval evaluation"

benchmark:
  retrieval:
    strategy: hybrid
    top_k: 20
    score_threshold: 0.0
  evaluation:
    k_values: [1, 5, 10, 20]
    metrics: ["precision", "recall", "mrr", "ndcg"]

retrieval_pipeline:
  retriever:
    type: hybrid
    fusion_method: rrf
    # ... configuration details
```

## Testing

### Run Test Suite

```bash
# Run minimal pipeline tests (CI-friendly)
python -m pytest tests/pipeline/test_minimal_pipeline.py -v

# Run all pipeline tests
python -m pytest tests/pipeline/ -v

# Run tests with coverage
python -m pytest tests/pipeline/ --cov=components --cov=agent

# Run specific test categories
python -m pytest tests/pipeline/ -m "not requires_api"  # No API required
python -m pytest tests/pipeline/ -m "requires_api"     # Requires API keys
```

### Test Qdrant Connectivity

```bash
# Test vector database connection
python -m pytest tests/pipeline/test_qdrant_connectivity.py -v

# Inspect Qdrant collections
python bin/qdrant_inspector.py --list-collections
python bin/qdrant_inspector.py --collection-info sosum_stackoverflow_hybrid_v1
```

### Integration Testing

```bash
# Test agent with different configurations
python -c "
from agent.graph import graph
configs = ['fast_hybrid', 'modern_dense', 'modern_hybrid']
for config in configs:
    print(f'Testing {config}...')
    # Switch config and test
"
```

## Documentation

- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed project organization
- **[Quick Start Guide](docs/QUICK_START_GUIDE.md)** - Getting started tutorial
- **[SOSUM Ingestion](docs/SOSUM_INGESTION.md)** - Dataset ingestion guide
- **[MLOps Architecture](docs/MLOPS_PIPELINE_ARCHITECTURE.md)** - System architecture details

## Extending the System

### Add a Custom Reranker

```python
# components/my_custom_reranker.py
from components.retrieval_pipeline import Reranker, RetrievalResult
from typing import List

class MyCustomReranker(Reranker):
    @property
    def component_name(self) -> str:
        return "my_custom_reranker"
    
    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        # Your custom reranking logic
        for result in results:
            result.score = self.calculate_custom_score(query, result.document.page_content)
            result.metadata["reranked_by"] = self.component_name
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def calculate_custom_score(self, query: str, content: str) -> float:
        # Implement your scoring logic
        return 0.5  # Placeholder
```

### Create a New Configuration

```yaml
# pipelines/configs/retrieval/my_config.yml
description: "My custom retrieval configuration"

retrieval_pipeline:
  retriever:
    type: hybrid
    top_k: 15
    
  stages:
    - type: reranker
      config:
        model_type: my_custom
        custom_param: "value"
        
    - type: filter
      config:
        type: score
        min_score: 0.4
```

### Switch and Test

```bash
# Switch to your configuration
python bin/switch_agent_config.py my_config

# Test the new configuration
python -c "
from agent.graph import graph
result = graph.invoke({'question': 'test query', 'chat_history': []})
print(f'Answer: {result[\"answer\"]}')
print(f'Retrieved docs: {len(result.get(\"retrieved_documents\", []))}')
"
```

## Production Features

### Performance Optimization
- **Lazy Initialization**: Components load only when needed
- **Connection Pooling**: Efficient database connection management
- **Batch Processing**: Optimized embedding and reranking batches
- **Caching**: LRU caching for repeated queries and embeddings
- **Async Operations**: Non-blocking I/O for better throughput

### Monitoring & Observability
- **Structured Logging**: JSON logs with correlation IDs
- **Performance Metrics**: Response times, cache hit rates, error rates
- **Health Checks**: Database connectivity, model availability
- **Rich Metadata**: Retrieval paths, scores, and method tracking

### Configuration Management
- **Environment-based Configs**: Different configs per environment
- **Hot Reloading**: Switch configurations without restart
- **Validation**: Schema validation for all configurations
- **Rollback**: Easy rollback to previous configurations

### Error Handling
- **Graceful Degradation**: Fallback to simpler methods on failures
- **Circuit Breakers**: Prevent cascade failures
- **Retry Logic**: Exponential backoff for transient failures
- **Comprehensive Logging**: Detailed error context and stack traces

## Use Cases & Applications

### Document Q&A Systems
- **Knowledge Base Search**: Corporate wikis, documentation, FAQs
- **Research Assistance**: Academic papers, technical documentation
- **Customer Support**: Automated response generation with context

### Code & Technical Search
- **Semantic Code Search**: Find code snippets by functionality
- **API Documentation**: Contextual API usage examples
- **Stack Overflow Integration**: Programming Q&A with real data

### Domain-Specific Applications
- **Legal Research**: Case law, regulations, legal precedents
- **Medical Literature**: Research papers, clinical guidelines
- **Financial Analysis**: Reports, earnings calls, market research

### Multi-Modal Retrieval
- **Table Extraction**: Structured data from documents
- **Image-Text Retrieval**: Combined visual and textual search
- **Temporal Queries**: Time-aware information retrieval


## Contributing

1. **Fork the Repository**
   ```bash
   git fork https://github.com/your-org/thesis-rag-system
   cd thesis-rag-system
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   pip install -r tests/requirements-minimal.txt
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/my-new-component
   ```

4. **Follow Extension Patterns**
   - Add new components to `components/`
   - Create configuration files in `pipelines/configs/retrieval/`
   - Add tests to `tests/pipeline/`
   - Update documentation

5. **Run Tests**
   ```bash
   python -m pytest tests/pipeline/ -v
   python -m pytest tests/pipeline/test_minimal_pipeline.py
   ```

6. **Submit Pull Request**
   - Ensure all tests pass
   - Include benchmark results if applicable
   - Update documentation for new features

### Development Guidelines

- **Code Style**: Follow PEP 8, use type hints
- **Testing**: Write tests for new components
- **Documentation**: Update README and docstrings
- **Configuration**: Provide example YAML configs
- **Backwards Compatibility**: Maintain API compatibility

## Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Scale specific services
docker-compose up -d --scale qdrant=2

# View logs
docker-compose logs -f rag-system

# Stop services
docker-compose down
```

### Custom Docker Build

```bash
# Build the image
docker build -t my-rag-system .

# Run with environment variables
docker run -d \
  -e GOOGLE_API_KEY=your_key \
  -e QDRANT_HOST=qdrant \
  -p 8000:8000 \
  my-rag-system
```

## Troubleshooting

### Common Issues

**Qdrant Connection Failed**
```bash
# Check Qdrant status
docker ps | grep qdrant
curl http://localhost:6333/health

# Restart Qdrant
docker-compose restart qdrant
```

**API Key Issues**
```bash
# Check environment variables
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY

# Test API connectivity
python -c "
import os
from embedding.factory import EmbeddingFactory
factory = EmbeddingFactory()
embedder = factory.create_embedder('google')
print('Google API working!')
"
```

**Configuration Not Found**
```bash
# List available configurations
python bin/switch_agent_config.py --list

# Validate configuration
python -c "
import yaml
with open('pipelines/configs/retrieval/modern_hybrid.yml') as f:
    config = yaml.safe_load(f)
    print('Configuration valid!')
"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangGraph**: Agent workflow orchestration
- **Qdrant**: High-performance vector database
- **Sentence Transformers**: Embedding models and rerankers
- **Google AI**: Embedding API services
- **BEIR**: Benchmark datasets for information retrieval

---

**Ready to build production RAG systems?** Start with our [Quick Start Guide](docs/QUICK_START_GUIDE.md)!
