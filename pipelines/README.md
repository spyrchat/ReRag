# Ingestion Pipeline

A comprehensive, theory-backed ingestion pipeline for retrieval-augmented generation (RAG) systems. This pipeline implements deterministic IDs, idempotent loads, dataset adapters, and comprehensive evaluation.

## 🎯 Mission

This pipeline guarantees:
- **Reproducibility**: Same raw data → same chunk IDs and vectors
- **Idempotency**: Reruns don't duplicate anything; only changed content is updated
- **Portability**: New datasets plug in without touching downstream code
- **Observability**: You can prove what you loaded, when, with which config/code

## 🏗️ Architecture

```
pipelines/
├── contracts.py          # Base schemas and interfaces
├── adapters/             # Dataset-specific adapters
│   ├── __init__.py
│   ├── README.md
│   ├── loader.py                # Base adapter utilities
│   ├── my_dataset_schema.py    # Template for new adapters
│   └── stackoverflow.py         # Implemented adapter
├── ingest/               # Core ingestion components
│   ├── validator.py      # Document validation and cleaning
│   ├── chunker.py        # Advanced chunking strategies
│   ├── embedder.py       # Embedding pipeline with caching
│   ├── uploader.py       # Vector store uploader
│   ├── smoke_tests.py    # Post-ingestion verification
│   └── pipeline.py       # Main orchestrator
├── eval/                 # Evaluation framework
│   └── evaluator.py      # Unified retrieval evaluation
└── configs/              # Configuration files
    ├── README.md
    ├── __init__.py
    ├── datasets/         # Dataset-specific configs
    ├── retrieval/        # Retrieval strategy configs
    └── examples/         # Example configurations
```

## 🚀 Quick Start

### 1. Install Dependencies

The pipeline uses your existing dependencies plus these optional packages:
```bash
pip install -r requirements.txt  # For evaluation metrics
```

### 2. Configure Your Setup

Copy and modify a configuration template:
```bash
# Use the StackOverflow config as a template
cp pipelines/configs/datasets/stackoverflow_hybrid.yml my_config.yml
# Edit my_config.yml for your needs
```

### 3. Ingest Your First Dataset

```bash
# Ingest StackOverflow dataset
python bin/ingest.py ingest --config pipelines/configs/datasets/stackoverflow_hybrid.yml

# Check ingestion status
python bin/ingest.py status
```

### 4. Check Status

```bash
python bin/ingest.py status
```

## 📖 Core Concepts

### Dataset Adapters

Each dataset needs an adapter that implements:
- `read_rows()`: Read raw dataset into standardized format
- `to_documents()`: Convert to LangChain Documents
- `get_evaluation_queries()`: Provide evaluation queries

### Deterministic IDs

- **Document ID**: `{source}:{external_id}:{content_hash[:12]}`
- **Chunk ID**: `{doc_id}#c{chunk_index:04d}`

This ensures identical content always gets the same ID.

### Chunking Strategies

- **Recursive**: General-purpose character-based chunking
- **Semantic**: Sentence-boundary aware chunking
- **Code Aware**: Preserves code blocks and functions
- **Table Aware**: Preserves table structure
- **Auto**: Automatically selects strategy based on content

### Embedding Strategies

- **Dense**: Semantic embeddings (HuggingFace, Titan)
- **Sparse**: Keyword-based embeddings (BM25, FastEmbed)
- **Hybrid**: Both dense and sparse for optimal recall

## 🎛️ Advanced Usage

### Batch Processing

```bash
# Example batch configuration structure
cat > batch_config.json << EOF
{
  "datasets": [
    {"type": "stackoverflow", "path": "datasets/sosum/data", "version": "1.0.0"},
    {"type": "my_custom_dataset", "path": "/path/to/my_data", "version": "1.0.0"}
  ]
}
EOF

# The batch-ingest CLI command is not yet implemented.
# For batch ingestion, run ingest multiple times with different configs
# or use the BatchIngestionPipeline class programmatically.
```

### Custom Dataset Adapter

```python
from pipelines.contracts import DatasetAdapter, BaseRow
from pipelines.adapters.base import MyCustomRow

class MyDatasetAdapter(DatasetAdapter):
    @property
    def source_name(self) -> str:
        return "my_dataset"
    
    def read_rows(self, split) -> Iterable[MyCustomRow]:
        # Your data reading logic
        pass
    
    def to_documents(self, rows, split) -> List[Document]:
        # Convert to LangChain Documents
        pass
    
    def get_evaluation_queries(self, split) -> List[Dict]:
        # Return evaluation queries
        pass
```

### Canary → Promote Workflow

```bash
# 1. Canary deployment (test with small sample)
python bin/ingest.py ingest --config my_config.yml --canary --max-docs 100

# 2. Verify canary collection manually using Qdrant dashboard
# or by testing retrieval with bin/agent_retriever.py

# 3. If good, run full ingestion (re-run without --canary)
python bin/ingest.py ingest --config my_config.yml

# 4. Clean up canary collections
python bin/ingest.py cleanup
```

## 🔧 Configuration

### Main Configuration (config.yml)
```yaml
embedding_strategy: hybrid

embedding:
  dense:
    provider: hf
    model_name: sentence-transformers/all-MiniLM-L6-v2
  sparse:
    provider: fastembed
    model_name: Qdrant/bm25

chunking:
  strategy: semantic
  chunk_size: 500
  chunk_overlap: 50

qdrant:
  collection: my_collection
  dense_vector_name: dense
  sparse_vector_name: sparse
```

### Dataset-Specific Configs
Each dataset can have specialized settings in `pipelines/configs/`.

## 📈 Monitoring & Observability

### Lineage Tracking
Every ingestion run creates a lineage record:
```json
{
  "run_id": "uuid",
  "dataset_name": "stackoverflow",
  "config_hash": "abc123",
  "git_commit": "def456",
  "total_documents": 100,
  "successful_chunks": 850,
  "sample_doc_ids": ["doc1", "doc2"],
  "environment": {...}
}
```

### Logs
- Console output for real-time monitoring
- Structured logs in `logs/ingestion.log`
- Per-run lineage in `output/lineage/`

### Smoke Tests
Automatic post-ingestion validation:
- Collection exists and is populated
- Vector dimensions are consistent
- Golden queries return reasonable results
- Embedding quality metrics

## 🛠️ Extending the Pipeline

### Adding a New Dataset

1. **Create an adapter** in `pipelines/adapters/my_dataset.py` (follow the pattern in `stackoverflow.py`)
2. **Add configuration** in `pipelines/configs/datasets/my_dataset.yml`
3. **Reference adapter in config** under `dataset.adapter` key
4. **Test** with dry run: `python bin/ingest.py ingest --config pipelines/configs/datasets/my_dataset.yml --dry-run --max-docs 10`

### Adding a New Chunking Strategy

```python
from pipelines.ingest.chunker import ChunkingStrategy

class MyChunkingStrategy(ChunkingStrategy):
    @property
    def strategy_name(self) -> str:
        return "my_strategy"
    
    def chunk(self, documents) -> List[Document]:
        # Your chunking logic
        pass

# Register in ChunkingStrategyFactory.STRATEGIES
```

### Adding a New Smoke Test

```python
from pipelines.ingest.smoke_tests import SmokeTest, SmokeTestResult

class MyCustomTest(SmokeTest):
    @property
    def test_name(self) -> str:
        return "my_test"
    
    def run(self, config) -> SmokeTestResult:
        # Your test logic
        pass
```

## 🎯 Best Practices

### Development Workflow
1. **Start with dry runs** to validate data processing
2. **Use canary deployments** for new configurations
3. **Run evaluations** to measure retrieval quality
4. **Monitor lineage** for reproducibility

### Production Deployment
1. **Pin configuration versions** with git commits
2. **Use deterministic seeds** for dataset splits
3. **Archive lineage records** for compliance
4. **Monitor embedding quality** over time

### Performance Optimization
1. **Enable embedding caching** for repeated runs
2. **Tune batch sizes** based on your hardware
3. **Use appropriate chunking** for your content type
4. **Monitor vector store performance**

## 🔍 Troubleshooting

### Common Issues

**Embeddings are all zeros**
```bash
# Check embedding configuration with verbose logging
python bin/ingest.py ingest --config my_config.yml --dry-run --max-docs 1 --verbose
```

**Collection not found**
```bash
# Check Qdrant connection and collection settings
python bin/ingest.py status
```

**Need to verify ingestion**
```bash
# Use the verify flag during ingestion
python bin/ingest.py ingest --config my_config.yml --max-docs 10 --verify
```

**Import errors**
Make sure you're running from the project root and all dependencies are installed.

## 📝 Development Notes

This pipeline is designed to integrate seamlessly with your existing:
- Qdrant vector database setup
- Embedding factory (HF, Titan, FastEmbed)
- Configuration system (YAML-based)
- Document processing pipeline

The theory-backed design ensures that you can confidently:
- Add new datasets without touching core code
- Compare retrieval quality across configurations
- Reproduce any ingestion run exactly
- Scale to production workloads

## 🎉 Example: Complete Workflow

```bash
# 1. Setup - commit your configuration
git add pipelines/configs/datasets/my_dataset.yml
git commit -m "Add dataset configuration"

# 2. Test with dry run
python bin/ingest.py ingest \
  --config pipelines/configs/datasets/my_dataset.yml \
  --dry-run --max-docs 5 --verbose

# 3. Canary deployment (small test batch)
python bin/ingest.py ingest \
  --config pipelines/configs/datasets/my_dataset.yml \
  --canary --max-docs 50

# 4. Verify canary manually using:
#    - Qdrant dashboard at http://localhost:6333/dashboard
#    - bin/agent_retriever.py for test queries

# 5. Full deployment (if canary looks good)
python bin/ingest.py ingest \
  --config pipelines/configs/datasets/my_dataset.yml \
  --verify

# 6. Check final status
python bin/ingest.py status

# 7. Clean up canary collections
python bin/ingest.py cleanup
```

This gives you a production-ready, theory-backed ingestion pipeline that scales across datasets and maintains full lineage! 🚀
