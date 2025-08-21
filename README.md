# 🚀 Advanced RAG Retrieval System with LangGraph Agent

A production-ready, modular RAG (Retrieval-Augmented Generation) system with configurable pipelines and LangGraph agent integration.

## ✨ **Key Features**

- 🔧 **YAML-Configurable Pipelines**: Switch retrieval strategies without code changes
- 🤖 **LangGraph Agent Integration**: Seamless agent workflows with rich metadata
- 📈 **Modular Components**: Easily extensible rerankers, filters, and retrievers
- 🔄 **Multiple Retrieval Methods**: Dense, sparse, and hybrid retrieval
- ⚡ **Production Ready**: Robust error handling, logging, and monitoring
- 🧪 **A/B Testing Support**: Compare configurations easily
- 📊 **Rich Metadata**: Access scores, methods, and quality metrics

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LangGraph     │────│  Configurable    │────│   Retrieval     │
│     Agent       │    │  Retriever Agent │    │   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       │                                │                                │
                 ┌─────▼─────┐                 ┌───────▼────────┐                ┌─────▼─────┐
                 │ Retrievers │                 │   Rerankers    │                │  Filters  │
                 │           │                 │               │                │           │
                 │• Dense    │                 │• CrossEncoder │                │• Score    │
                 │• Sparse   │                 │• BGE Reranker │                │• Content  │
                 │• Hybrid   │                 │• Multi-stage  │                │• Custom   │
                 └───────────┘                 └────────────────┘                └───────────┘
```

## 🚀 **Quick Start**

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Configure Environment**

```bash
# Copy example config
cp config.yml.example config.yml

# Set up your API keys and database connections in config.yml
```

### 3. **Start Using the System**

```python
# main.py - Chat with your agent
from agent.graph import graph

state = {"question": "How to handle Python exceptions?"}
result = graph.invoke(state)
print(result["answer"])
```

### 4. **Switch Retrieval Configurations**

```bash
# List available configurations
python bin/switch_agent_config.py --list

# Switch to advanced reranked pipeline  
python bin/switch_agent_config.py advanced_reranked

# Test the configuration
python test_agent_retriever_node.py
```

## 📋 **Available Configurations**

| Configuration | Description | Components | Use Case |
|---------------|-------------|------------|----------|
| `basic_dense` | Simple dense retrieval | Dense retriever only | Development, testing |
| `advanced_reranked` | Production quality | Dense + CrossEncoder + filters | Production RAG |
| `hybrid_multistage` | Best performance | Hybrid + multi-stage reranking | High-quality results |
| `experimental` | Latest features | BGE reranker + custom filters | Experimentation |

## 🔧 **Configuration Example**

```yaml
# pipelines/configs/retrieval/advanced_reranked.yml
retrieval_pipeline:
  retriever:
    type: dense
    top_k: 10
    
  stages:
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: "ms-marco-MiniLM-L-6-v2"
        
    - type: filter
      config:
        type: score
        min_score: 0.5
        
    - type: answer_enhancer
      config:
        boost_factor: 2.0
```

## 📂 **Project Structure**

```
Thesis/
├── 📁 agent/                     # LangGraph agent implementation
│   ├── graph.py                  # Main agent graph
│   ├── schema.py                 # Agent state schemas
│   └── nodes/                    # Agent nodes (retriever, generator, etc.)
│
├── 📁 components/                # Modular retrieval components
│   ├── retrieval_pipeline.py    # Main pipeline orchestrator
│   ├── rerankers.py             # Reranking implementations
│   ├── filters.py               # Filtering implementations
│   └── advanced_rerankers.py    # Advanced reranking strategies
│
├── 📁 pipelines/                 # Data processing and configuration
│   ├── configs/retrieval/       # Retrieval pipeline configurations
│   ├── adapters/                # Dataset adapters (BEIR, etc.)
│   └── ingest/                  # Data ingestion pipeline
│
├── 📁 bin/                       # Command-line utilities
│   ├── switch_agent_config.py   # Configuration management
│   ├── agent_retriever.py       # Configurable retriever agent
│   └── retrieval_pipeline.py    # Direct pipeline usage
│
├── 📁 docs/                      # Documentation
│   ├── SYSTEM_EXTENSION_GUIDE.md # Complete extension guide
│   ├── AGENT_INTEGRATION.md     # Agent integration details
│   └── EXTENSIBILITY.md         # Quick extensibility overview
│
├── 📁 tests/                     # Test suite
│   ├── retrieval/               # Retrieval pipeline tests
│   └── agent/                   # Agent integration tests
│
├── 📁 deprecated/                # Legacy code (organized)
│   ├── old_processors/          # Superseded by new pipeline
│   ├── old_debug_scripts/       # Legacy debugging tools
│   └── old_playground/          # Legacy test scripts
│
├── 📁 database/                  # Database controllers
├── 📁 embedding/                 # Embedding utilities
├── 📁 retrievers/               # Base retrievers
├── 📁 examples/                 # Usage examples
└── 📁 config/                   # Configuration utilities
```

## 🧪 **Testing**

```bash
# Test agent integration
python test_agent_retriever_node.py

# Run all tests
python tests/run_all_tests.py

# Test specific components
python -m pytest tests/retrieval/ -v
```

## 📚 **Documentation**

- **[🚀 System Extension Guide](docs/SYSTEM_EXTENSION_GUIDE.md)** - Complete guide to extending the system
- **[🤖 Agent Integration](docs/AGENT_INTEGRATION.md)** - How the agent uses configurable pipelines  
- **[⚡ Extensibility Overview](docs/EXTENSIBILITY.md)** - Quick overview of extension capabilities
- **[🏗️ Architecture](docs/MLOPS_PIPELINE_ARCHITECTURE.md)** - System architecture details

## 🔧 **Extending the System**

### Add a Custom Reranker

```python
# components/my_reranker.py
from .rerankers import BaseReranker

class MyCustomReranker(BaseReranker):
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # Your custom reranking logic
        for doc in documents:
            doc.metadata["score"] = self.calculate_score(query, doc.page_content)
        
        return sorted(documents, key=lambda x: x.metadata["score"], reverse=True)
```

### Create a New Configuration

```yaml
# pipelines/configs/retrieval/my_config.yml
retrieval_pipeline:
  retriever:
    type: hybrid
    top_k: 15
    
  stages:
    - type: reranker
      config:
        model_type: my_custom
        custom_param: "value"
```

### Switch and Test

```bash
python bin/switch_agent_config.py my_config
python test_agent_retriever_node.py
```

## 🚀 **Production Usage**

The system is designed for production use with:

- **Robust Error Handling**: Graceful degradation when components fail
- **Comprehensive Logging**: Monitor retrieval performance and quality
- **Configuration Management**: Easy deployment of different strategies
- **Performance Optimization**: Efficient batching and caching support
- **Monitoring Ready**: Built-in metrics and health checks

## 🎯 **Use Cases**

- **Document Q&A Systems**: High-quality retrieval for knowledge bases
- **Research Assistants**: Multi-modal retrieval for academic content
- **Customer Support**: Context-aware response generation
- **Code Search**: Semantic search over codebases
- **Legal Research**: Precise retrieval from legal documents

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add your extension following the patterns in `docs/SYSTEM_EXTENSION_GUIDE.md`
4. Add tests for your components
5. Submit a pull request

## 📊 **Performance**

The system supports various performance optimization strategies:

- **Caching**: LRU caching for repeated queries
- **Batching**: Efficient batch processing for rerankers
- **Adaptive Top-K**: Dynamic result count based on query complexity
- **Multi-threading**: Parallel processing for pipeline stages

## 🔄 **Migration from Legacy**

If you have existing code using the deprecated `processors/` system:

1. Check `deprecated/old_processors/` for reference
2. Use the new pipeline configurations in `pipelines/configs/retrieval/`
3. Follow the migration patterns in `docs/AGENT_INTEGRATION.md`

## 📝 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎉 Ready to build amazing RAG systems?** Start with the [System Extension Guide](docs/SYSTEM_EXTENSION_GUIDE.md)!
