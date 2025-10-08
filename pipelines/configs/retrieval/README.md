# Retrieval Configurations for Agent

This directory contains retrieval pipeline configurations used by the RAG agent (`main.py`).

## Available Configurations

### 1. Dense Retrieval (BGE-M3)

#### `fast_dense_bge_m3.yml` ⚡ (Default)
- **Type**: Pure dense (semantic) retrieval
- **Model**: BAAI/bge-m3 (1024 dimensions)
- **Speed**: Fast (no reranking)
- **Results**: Top 5 documents
- **Best for**: Interactive chat with quick responses
- **Stages**: 2 (retrieval + light filtering)

#### `dense_bge_m3.yml` 🎯
- **Type**: Pure dense (semantic) retrieval
- **Model**: BAAI/bge-m3 (1024 dimensions)
- **Speed**: Moderate (with reranking)
- **Results**: Top 5 documents after reranking (10 before)
- **Best for**: High-quality responses, when accuracy matters
- **Stages**: 3 (retrieval + filtering + cross-encoder reranking)

### 2. Hybrid Retrieval (Coming Soon)

#### `fast_hybrid.yml` (if exists)
- Combines dense and sparse retrieval
- Uses RRF (Reciprocal Rank Fusion)

## Usage

### Method 1: Edit `config.yml` (Root Level)

Edit the main config file at the project root:

```yaml
# config.yml
agent_retrieval:
  config_path: pipelines/configs/retrieval/fast_dense_bge_m3.yml
```

### Method 2: Use Switch Script

```bash
# List available configs
python bin/switch_agent_config.py --list

# Switch to a specific config
python bin/switch_agent_config.py fast_dense_bge_m3

# Switch to dense with reranking
python bin/switch_agent_config.py dense_bge_m3
```

### Method 3: Programmatically

```python
from config.config_loader import load_config

# Load main config
config = load_config("config.yml")

# Get retrieval config path
retrieval_config_path = config["agent_retrieval"]["config_path"]
```

## Configuration Structure

Each retrieval config has the following structure:

```yaml
description: "Human-readable description"

retrieval_pipeline:
  retriever:
    type: "dense"           # or "sparse", "hybrid"
    top_k: 10
    score_threshold: 0.0
    
    embedding:
      provider: "huggingface"
      model: "BAAI/bge-m3"
      # ... model parameters
    
    qdrant:
      collection_name: "your_collection"
      vector_name: "dense"
  
  stages:
    - type: "retriever"
      name: "primary_retriever"
    
    - type: "score_filter"
      config:
        min_score: 0.3
    
    - type: "reranker"
      config:
        model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_k: 5
```

## Performance Comparison

| Config | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `fast_dense_bge_m3` | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | Interactive chat |
| `dense_bge_m3` | ⚡⚡ Moderate | ⭐⭐⭐⭐ Excellent | Production use |

## Creating Custom Configs

1. Copy an existing config file
2. Modify the parameters:
   - `top_k`: Number of results
   - `score_threshold`: Minimum similarity score
   - `stages`: Add/remove pipeline stages
   - `embedding`: Change model or provider
3. Save with descriptive name
4. Update `config.yml` to use it

## Qdrant Collections

Make sure your Qdrant collection matches the config:

```yaml
qdrant:
  collection_name: "sosum_stackoverflow_bge_splade_recursive_v2"
  vector_name: "dense"  # Must match your collection's vector name
```

Check available collections:
```bash
python bin/qdrant_inspector.py list-collections
```

## Troubleshooting

### Config not found error
- Check that the path in `config.yml` is correct
- Verify the file exists in `pipelines/configs/retrieval/`

### Qdrant connection error
- Ensure Qdrant is running: `docker-compose up -d qdrant`
- Check host/port in the config

### Slow retrieval
- Switch to `fast_dense_bge_m3` (no reranking)
- Reduce `top_k` value
- Enable caching in performance settings

### Poor quality results
- Switch to `dense_bge_m3` (with reranking)
- Increase `top_k` for more candidates
- Adjust `score_threshold` to filter low-quality results

## Environment Variables

Required in `.env`:
```properties
# For OpenAI (LLM)
OPENAI_API_KEY="your-key"

# Qdrant (if remote)
QDRANT_HOST="localhost"
QDRANT_PORT=6333
```

## Related Files

- **Main config**: `/config.yml`
- **Switch script**: `/bin/switch_agent_config.py`
- **Agent graph**: `/agent/graph.py`
- **Retriever node**: `/agent/nodes/retriever.py`
