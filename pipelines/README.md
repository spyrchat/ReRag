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
│   ├── natural_questions.py
│   ├── stackoverflow.py
│   └── energy_papers.py
├── ingest/               # Core ingestion components
│   ├── validator.py      # Document validation and cleaning
│   ├── chunker.py        # Advanced chunking strategies
│   ├── embedder.py       # Embedding pipeline with caching
│   ├── uploader.py       # Vector store uploader
│   ├── smoke_tests.py    # Post-ingestion verification
│   └── pipeline.py       # Main orchestrator
├── eval/                 # Evaluation framework
│   └── evaluator.py      # Unified retrieval evaluation
└── configs/              # Dataset-specific configurations
    ├── natural_questions.yml
    ├── stackoverflow.yml
    └── energy_papers.yml
```

## 🚀 Quick Start

### 1. Install Dependencies

The pipeline uses your existing dependencies plus these optional packages:
```bash
pip install numpy  # For evaluation metrics
```

### 2. Configure Your Setup

Copy and modify a configuration template:
```bash
cp pipelines/configs/energy_papers.yml my_config.yml
# Edit my_config.yml for your needs
```

### 3. Ingest Your First Dataset

```bash
# Ingest your energy papers (using existing papers/ directory)
python bin/ingest.py ingest energy_papers papers/ --config my_config.yml

# Dry run with limited documents for testing
python bin/ingest.py ingest energy_papers papers/ --dry-run --max-docs 10

# Canary deployment (test with separate collection)
python bin/ingest.py ingest energy_papers papers/ --canary --verify
```

### 4. Check Status

```bash
python bin/ingest.py status
```

### 5. Evaluate Retrieval

```bash
python bin/ingest.py evaluate energy_papers papers/ --output-dir results/
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
# Create batch configuration
cat > batch_config.json << EOF
{
  "datasets": [
    {"type": "energy_papers", "path": "papers/", "version": "1.0.0"},
    {"type": "stackoverflow", "path": "/path/to/stackoverflow", "version": "1.0.0"}
  ]
}
EOF

# Run batch ingestion
python bin/ingest.py batch-ingest batch_config.json --max-docs 100
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
# 1. Canary deployment
python bin/ingest.py ingest my_dataset /path/to/data --canary

# 2. Verify canary collection
python bin/ingest.py evaluate my_dataset /path/to/data --output-dir canary_results/

# 3. If good, promote (re-run without --canary)
python bin/ingest.py ingest my_dataset /path/to/data

# 4. Clean up canary
python bin/ingest.py cleanup
```


### Usage
```bash
python bin/ingest.py evaluate energy_papers papers/ \
  --split test \
  --output-dir evaluation_results/
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
  "dataset_name": "energy_papers",
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

1. **Create an adapter** in `pipelines/adapters/my_dataset.py`
2. **Add configuration** in `pipelines/configs/my_dataset.yml`
3. **Register in CLI** by adding to `get_adapter()` in `bin/ingest.py`
4. **Test** with dry run: `python bin/ingest.py ingest my_dataset /path --dry-run`

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
# Check embedding configuration
python bin/ingest.py ingest my_dataset /path --dry-run --max-docs 1 -v
```

**Collection not found**
```bash
# Check Qdrant connection and collection settings
python bin/ingest.py status
```

**Evaluation shows zero recall**
```bash
# Verify evaluation queries match ingested content
python bin/ingest.py evaluate my_dataset /path --max-docs 10
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
# 1. Setup
git add . && git commit -m "Setup ingestion pipeline"

# 2. Test with dry run
python bin/ingest.py ingest energy_papers papers/ \
  --config pipelines/configs/energy_papers.yml \
  --dry-run --max-docs 5 --verbose

# 3. Canary deployment
python bin/ingest.py ingest energy_papers papers/ \
  --canary --max-docs 50

# 4. Evaluate canary
python bin/ingest.py evaluate energy_papers papers/ \
  --output-dir canary_eval/

# 5. Full deployment (if canary looks good)
python bin/ingest.py ingest energy_papers papers/

# 6. Production evaluation
python bin/ingest.py evaluate energy_papers papers/ \
  --output-dir production_eval/

# 7. Check final status
python bin/ingest.py status
```

This gives you a production-ready, theory-backed ingestion pipeline that scales across datasets and maintains full lineage! 🚀
