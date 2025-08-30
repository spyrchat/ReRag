# Unified Configuration System

## Overview

The thesis project now uses a **unified configuration system** that consolidates all settings into a single `config.yml` file. This eliminates configuration sprawl and makes the system easier to maintain and configure.

## Configuration Structure

### Main Configuration File: `config.yml`

```yaml
# Google Gemini Embedding Configuration
embedding:
  dense:
    provider: google
    model: models/embedding-001
    dimensions: 768
    api_key_env: GOOGLE_API_KEY
    batch_size: 32
    vector_name: dense
  sparse:
    provider: sparse
    model: Qdrant/bm25
    vector_name: sparse
  strategy: hybrid

# Retriever Configurations (embedded)
retrievers:
  dense:
    type: dense
    top_k: 10
    # ... specific configurations
  
  hybrid:
    type: hybrid
    top_k: 10
    fusion_method: rrf
    # ... specific configurations

# Pipeline and Benchmark Settings
retrieval_pipeline:
  default_retriever: hybrid
  components:
    - type: retriever
    - type: score_filter
    - type: reranker

benchmark:
  evaluation:
    k_values: [1, 5, 10, 20]
    metrics: ["precision", "recall", "f1", "mrr", "ndcg"]
  retrieval:
    strategy: hybrid
    top_k: 20
```

## Usage

### Loading Configuration

```python
from config.config_loader import load_config, get_retriever_config

# Load main configuration
config = load_config("config.yml")

# Get retriever-specific config
retriever_config = get_retriever_config(config, "hybrid")
```

### Creating Pipelines

```python
from components.retrieval_pipeline import RetrievalPipelineFactory

# Create pipeline from unified config
pipeline = RetrievalPipelineFactory.create_from_unified_config(config, "hybrid")
```

### Running Benchmarks

```python
from benchmarks.benchmarks_runner import BenchmarkRunner

# Initialize with unified config
runner = BenchmarkRunner(config)
results = runner.run_benchmark(adapter)
```

### Configuration Overrides

```python
from config.config_loader import load_config_with_overrides

# Load with overrides for specific experiments
overrides = {
    "benchmark": {
        "retrieval": {"strategy": "dense", "top_k": 50}
    }
}
config = load_config_with_overrides("config.yml", overrides)
```

## Migration from Old System

### Removed Files/Directories
- `pipelines/configs/retrievers/` (consolidated into main config)
- `pipelines/configs/retrieval/` (consolidated into main config)  
- `pipelines/configs/retriever_config_loader.py` (replaced by enhanced config_loader)

### Updated Components
- `config/config_loader.py` - Enhanced with retriever/benchmark config extraction
- `components/retrieval_pipeline.py` - Added `create_from_unified_config()` method
- `benchmarks/benchmarks_runner.py` - Uses unified config structure
- `database/qdrant_controller.py` - Accepts config parameter

## Environment Variables

Still supported for backward compatibility:
- `QDRANT_COLLECTION` - Default collection name
- `GOOGLE_API_KEY` - Google Gemini API key
- Database connection settings (PostgreSQL, Qdrant)

## Benefits

1. **Single Source of Truth** - All configuration in one place
2. **No Configuration Sprawl** - Eliminated scattered config files
3. **Easy Overrides** - Simple override mechanism for experiments
4. **Type Safety** - Structured configuration with validation
5. **Backward Compatibility** - Environment variables still work
6. **Google Gemini Ready** - Optimized for Google Gemini embeddings

## Examples

See the `examples/` directory:
- `unified_config_example.py` - Configuration loading and pipeline creation
- `unified_benchmark_example.py` - Running benchmarks with different strategies
- `benchmark_config_override.yml` - Example override configuration
