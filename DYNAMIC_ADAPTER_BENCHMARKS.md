# Dynamic Adapter Loading for Benchmarks

## Overview

The benchmark system now supports **dynamic adapter loading from YAML configuration files**, eliminating the need to modify Python code when adding new dataset adapters.

## Changes Made

### 1. **Updated `benchmarks_runner.py`**

Added dynamic adapter loading support:

```python
def create_adapter_from_config(self, qdrant_client=None) -> BenchmarkAdapter:
    """Create a benchmark adapter from configuration."""
    dataset_config = self.config.get("dataset", {})
    
    adapter_spec = dataset_config["adapter"]
    dataset_path = dataset_config["path"]
    collection_name = self.config.get("retrieval", {}).get("qdrant", {}).get("collection_name")
    
    # Load adapter dynamically
    adapter = AdapterLoader.load_adapter(
        adapter_spec=adapter_spec,
        dataset_path=dataset_path,
        version="1.0.0",
        qdrant_client=qdrant_client,
        collection_name=collection_name
    )
    
    return adapter
```

**Updated method signatures:**
- `run_benchmark(adapter=None, ..., qdrant_client=None)` - adapter now optional
- `run_benchmark_with_individual_results(adapter=None, ..., qdrant_client=None)` - adapter now optional

### 2. **Updated Experiment Runners**

**`experiment1.py` and `experiment3.py`:**

```python
# OLD - Hardcoded adapter
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
adapter = StackOverflowBenchmarkAdapter(
    dataset_path=config['dataset']['path'],
    qdrant_client=qdrant_client,
    collection_name=qdrant_cfg['collection_name']
)
results = runner.run_benchmark(adapter=adapter, max_queries=...)

# NEW - Dynamic loading from config
runner = BenchmarkRunner(config)
results = runner.run_benchmark(
    qdrant_client=qdrant_client,
    max_queries=config['max_queries']
)
```

### 3. **Updated YAML Configurations**

All benchmark YAML files now include the adapter specification:

```yaml
dataset:
  adapter: "benchmarks.benchmarks_adapters.StackOverflowBenchmarkAdapter"
  path: "/home/spiros/Desktop/Thesis/datasets/sosum/data"
  use_ground_truth: true
```

**Updated files:**
- `benchmark_scenarios/experiment_1/*.yml` (5 files)
- `benchmark_scenarios/experiment_2/*.yml` (2 files)  
- `benchmark_scenarios/experiment_3/*.yml` (5 files)

### 4. **Updated `optimize_alpha_fixed_k.py`**

Now reads adapter from config with fallback to CLI args:

```python
# Load adapter dynamically from config
from pipelines.adapters.loader import AdapterLoader

adapter_spec = base_cfg["dataset"].get("adapter")
if not adapter_spec:
    # Fallback to CLI args for backwards compatibility
    adapter_spec = f"{args.adapter_module}.{args.adapter_class}"

base_adapter = AdapterLoader.load_adapter(
    adapter_spec=adapter_spec,
    dataset_path=args.dataset_path,
    version="1.0.0",
    qdrant_client=qdrant_client,
    collection_name=collection_name
)
```

### 5. **Fixed Import Issues**

Fixed relative imports in benchmark modules:
- `benchmarks/__init__.py` - `from .benchmarks_adapters import ...`
- `benchmarks/benchmarks_adapters.py` - `from .benchmark_contracts import ...`
- `benchmarks/benchmarks_runner.py` - `from .benchmarks_metrics import ...`

## Usage

### Running Experiments

No code changes needed - just run as before:

```bash
# Run experiment 1
python benchmarks/experiment1.py

# Run experiment 3  
python benchmarks/experiment3.py

# Run alpha optimization
python benchmarks/optimize_alpha_fixed_k.py \
    --scenario-yaml benchmark_scenarios/experiment_2/hybrid_bge_splade_fixed_k10.yml \
    --dataset-path datasets/sosum/data \
    --n-folds 5
```

### Adding a New Dataset Adapter

1. **Create your adapter class:**

```python
# my_adapters/custom_adapter.py
from benchmarks.benchmark_contracts import BenchmarkAdapter

class CustomBenchmarkAdapter(BenchmarkAdapter):
    def __init__(self, dataset_path, qdrant_client=None, collection_name=None, **kwargs):
        self.dataset_path = dataset_path
        # ...
    
    @property
    def name(self):
        return "custom"
    
    def load_queries(self, split="test"):
        # Load your queries
        pass
```

2. **Create a YAML config:**

```yaml
name: "Custom Dataset Benchmark"
description: "Benchmark using custom dataset"

dataset:
  adapter: "my_adapters.custom_adapter.CustomBenchmarkAdapter"  # Full class path
  path: "/path/to/custom/dataset"
  use_ground_truth: true

retrieval:
  type: "dense"
  top_k: 10
  qdrant:
    collection_name: "my_collection"
  # ... rest of config

evaluation:
  k_values: [1, 5, 10]
  metrics:
    retrieval: ["precision@k", "recall@k", "mrr"]

max_queries: 100
```

3. **Run the benchmark:**

```bash
python benchmarks/experiment1.py  # Will automatically use your adapter
```

## Benefits

✅ **No code modification** - Add new datasets via YAML only
✅ **Consistent with ingestion pipeline** - Same dynamic loading approach  
✅ **Backwards compatible** - Existing code still works
✅ **Flexible** - Supports any adapter implementing `BenchmarkAdapter` interface
✅ **Clean separation** - Configuration separate from code

## Configuration Format

### Required Fields

```yaml
dataset:
  adapter: "full.module.path.AdapterClass"  # Full Python import path
  path: "/absolute/path/to/dataset"          # Dataset location
```

### Optional Fields

```yaml
dataset:
  use_ground_truth: true                     # Whether to use ground truth
  ground_truth_type: "unordered_binary"      # Type of ground truth
```

### Adapter Class Requirements

Your adapter must:
1. Inherit from `BenchmarkAdapter`
2. Accept `dataset_path`, `qdrant_client`, and `collection_name` in `__init__`
3. Implement required abstract methods:
   - `name` property
   - `tasks` property  
   - `load_queries(split="test")`
   - `get_ground_truth(query_id)`

## Migration Guide

### For Existing Benchmarks

If you have existing benchmark code:

1. **Add adapter field to YAML:**
   ```yaml
   dataset:
     adapter: "benchmarks.benchmarks_adapters.StackOverflowBenchmarkAdapter"
     path: "..."
   ```

2. **Update runner calls:**
   ```python
   # Remove manual adapter instantiation
   results = runner.run_benchmark(
       qdrant_client=qdrant_client,  # Pass client instead of adapter
       max_queries=config['max_queries']
   )
   ```

3. **Remove unused imports:**
   ```python
   # Remove: from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
   ```

## Architecture

```
┌─────────────────────────────────────────────┐
│         Benchmark YAML Config               │
│  dataset:                                   │
│    adapter: "module.path.AdapterClass"      │
│    path: "/data/path"                       │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│       BenchmarkRunner                       │
│  create_adapter_from_config()               │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│       AdapterLoader                         │
│  (from pipelines.adapters.loader)           │
│  - Dynamically imports module               │
│  - Instantiates adapter class               │
│  - Passes qdrant_client, collection_name    │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│       Custom BenchmarkAdapter               │
│  - StackOverflowBenchmarkAdapter            │
│  - Or your custom adapter                   │
└─────────────────────────────────────────────┘
```

## See Also

- [Dynamic Adapters for Ingestion](./docs/DYNAMIC_ADAPTERS.md)
- [Quick Reference](./docs/DYNAMIC_ADAPTERS_QUICKREF.md)
- [Benchmark Contracts](./benchmarks/benchmark_contracts.py)
