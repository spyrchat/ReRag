# Pipeline Configuration Directory

This directory contains all configuration files for different pipeline components, organized by purpose and usage.

## 📁 Directory Structure

```
pipelines/configs/
├── README.md                    # This documentation
├── retriever_config_loader.py   # Configuration loading utilities
├── datasets/                    # Dataset-specific pipeline configurations
├── retrieval/                   # Agent retrieval configurations
├── examples/                    # Example configurations and templates
└── legacy/                      # Deprecated/old configurations
```

## 🗂️ Configuration Categories

### 📊 `datasets/` - Dataset Pipeline Configurations
Contains configurations for processing and ingesting different datasets.

- `stackoverflow.yml` - Main SOSum Stack Overflow dataset configuration
- `stackoverflow_hybrid.yml` - Hybrid embedding variant for Stack Overflow

**Purpose**: Data ingestion, chunking, embedding, and indexing pipelines.

### 🤖 `retrieval/` - Agent Retrieval Configurations
Contains configurations for the agent's retrieval system.

- `modern_hybrid.yml` - Advanced hybrid retrieval with RRF fusion and reranking
- `modern_dense.yml` - Dense retrieval with neural reranking
- `fast_hybrid.yml` - Speed-optimized hybrid retrieval

**Purpose**: Agent question-answering and document retrieval.

### 📚 `examples/` - Example Configurations
Contains template and example configuration files.

- `batch_example.json` - Example batch processing configuration

**Purpose**: Templates for creating new configurations.

### 🗄️ `legacy/` - Deprecated Configurations
Contains older configuration files kept for compatibility.

- `stackoverflow_bge_large.yml` - BGE large embedding configuration
- `stackoverflow_e5_large.yml` - E5 large embedding configuration  
- `stackoverflow_minilm.yml` - MiniLM embedding configuration

**Purpose**: Backward compatibility and reference.

## 🚀 Usage

### For Dataset Processing
```python
# Load dataset configuration
from pipelines.configs.retriever_config_loader import load_config
config = load_config("datasets/stackoverflow.yml")
```

### For Agent Retrieval
```bash
# Switch agent retrieval configuration
python bin/switch_agent_config.py modern_hybrid
```

### For Benchmarking
```python
# Use different retrieval configurations
configs = [
    "retrieval/modern_hybrid.yml",
    "retrieval/modern_dense.yml", 
    "retrieval/fast_hybrid.yml"
]
```

## 🔧 Configuration Types

### Dataset Configuration Schema
```yaml
dataset:
  name: "dataset_name"
  version: "1.0.0"
  description: "Dataset description"

embedding:
  strategy: "hybrid"  # dense, sparse, or hybrid
  dense:
    provider: "provider_name"
    model: "model_name"
  sparse:
    provider: "provider_name"
    model: "model_name"

chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 50
```

### Agent Retrieval Configuration Schema
```yaml
description: "Configuration description"

retrieval_pipeline:
  retriever:
    type: "hybrid"  # dense, sparse, or hybrid
    top_k: 20
    fusion_method: "rrf"
    
  stages:
    - type: "retriever"
    - type: "score_filter"
      config:
        min_score: 0.01
    - type: "reranker"
      config:
        model_type: "cross_encoder"
```

## 📝 Best Practices

1. **Naming Convention**:
   - Dataset configs: `{dataset_name}.yml`
   - Retrieval configs: `{strategy}_{variant}.yml`
   - Legacy configs: `{original_name}.yml` (in legacy/)

2. **Documentation**:
   - Include `description` field in all configs
   - Add comments for complex parameters
   - Document expected performance characteristics

3. **Version Control**:
   - Keep legacy configs for reproducibility
   - Version dataset configurations
   - Test new configs before deployment

## 🔍 Finding Configurations

- **For data processing**: Look in `datasets/`
- **For agent retrieval**: Look in `retrieval/`
- **For examples/templates**: Look in `examples/`
- **For old/deprecated**: Look in `legacy/`

## 🛠️ Maintenance

- Regularly review and clean up unused configurations
- Move outdated configs to `legacy/` instead of deleting
- Update documentation when adding new configuration types
- Test configuration loading after structural changes

---

*Last updated: August 30, 2025*
