# Retrieval Pipeline Configuration Guide

## Table of Contents
1. [Configuration Structure](#configuration-structure)
2. [Supported Stage Types](#supported-stage-types)
3. [Complete Examples](#complete-examples)
4. [Common Patterns](#common-patterns)

---

## Configuration Structure

### ✅ **What Works** (Required at ROOT level):

```yaml
retrieval:
  type: hybrid  # or dense, sparse, semantic
  top_k: 20
  fusion:
    method: rrf
    alpha: 0.5  # REQUIRED for hybrid! (0.0=sparse, 1.0=dense)
    rrf_k: 60

embedding:  # ROOT level, NOT nested
  strategy: hybrid
  dense:
    provider: google
    model: models/embedding-001
  sparse:
    provider: sparse
    model: Qdrant/bm25

qdrant:  # ROOT level, NOT nested
  collection_name: your_collection
  dense_vector_name: dense
  sparse_vector_name: sparse

retrieval_pipeline:  # Only for stages
  stages:
    - type: score_filter
      config: {...}
```

### ❌ **What Doesn't Work**:

```yaml
retrieval_pipeline:
  retriever:
    embedding: {...}  # ❌ Won't be read!
    qdrant: {...}     # ❌ Won't be read!
    performance: {...} # ❌ Ignored!
```

---

## Supported Stage Types

### 1. **Filters**

#### `score_filter`
Filter results by minimum score threshold.

```yaml
- type: score_filter
  config:
    min_score: 0.3  # Keep only results with score >= 0.3
```

#### `duplicate_filter`
Remove duplicate results.

```yaml
- type: duplicate_filter
  config:
    dedup_by: external_id  # or "content"
```

#### `tag_filter`
Filter by tags (for Stack Overflow content).

```yaml
- type: tag_filter
  config:
    required_tags: ["python", "javascript"]  # At least one required
    excluded_tags: ["deprecated"]             # None allowed
```

#### `result_limiter`
Limit final result count.

```yaml
- type: result_limiter
  config:
    max_results: 10
```

---

### 2. **Rerankers**

#### `cross_encoder`
CrossEncoder reranking (fast, good quality).

```yaml
- type: reranker
  config:
    model_type: cross_encoder
    model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_k: 10
```

#### `bge`
BGE reranker (higher quality, slower).

```yaml
- type: reranker
  config:
    model_type: bge
    model_name: BAAI/bge-reranker-base
    top_k: 10
```

#### `multistage`
Two-stage reranking (fast → high-quality).

```yaml
- type: reranker
  config:
    model_type: multistage
    stage1:
      model_type: cross_encoder
      model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 20  # Stage 1 keeps top 20
    stage2:
      model_type: bge
      model_name: BAAI/bge-reranker-base
      top_k: 10  # Stage 2 keeps top 10
```

#### `ensemble`
Ensemble reranking with weighted voting.

```yaml
- type: reranker
  config:
    model_type: ensemble
    rerankers:
      - model_type: cross_encoder
        model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
        weight: 0.4
      - model_type: bge
        model_name: BAAI/bge-reranker-base
        weight: 0.6
    top_k: 10
```

---

### 3. **Post-processors**

#### `answer_enhancer`
Enhance answer formatting and metadata (for Stack Overflow).

```yaml
- type: answer_enhancer
  config: {}  # No config needed
```

---

## Complete Examples

### Example 1: Simple Pipeline (Single Reranker)

```yaml
retrieval:
  type: hybrid
  top_k: 20
  fusion:
    method: rrf
    alpha: 0.5
    rrf_k: 60

embedding:
  strategy: hybrid
  dense:
    provider: google
    model: models/embedding-001
  sparse:
    provider: sparse
    model: Qdrant/bm25

qdrant:
  collection_name: my_collection
  dense_vector_name: dense
  sparse_vector_name: sparse

retrieval_pipeline:
  stages:
    - type: score_filter
      config:
        min_score: 0.01
    
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
        top_k: 10
```

---

### Example 2: Multi-Stage Pipeline (Maximum Quality)

```yaml
retrieval:
  type: hybrid
  top_k: 50  # Get many candidates
  fusion:
    method: rrf
    alpha: 0.6
    rrf_k: 60

embedding:
  strategy: hybrid
  dense:
    provider: google
    model: models/embedding-001
  sparse:
    provider: sparse
    model: Qdrant/bm25

qdrant:
  collection_name: my_collection
  dense_vector_name: dense
  sparse_vector_name: sparse

retrieval_pipeline:
  stages:
    # Stage 1: Remove low-quality results
    - type: score_filter
      config:
        min_score: 0.01
    
    # Stage 2: Remove duplicates
    - type: duplicate_filter
      config:
        dedup_by: external_id
    
    # Stage 3: Fast reranking
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
        top_k: 30
    
    # Stage 4: High-quality reranking
    - type: reranker
      config:
        model_type: bge
        model_name: BAAI/bge-reranker-base
        top_k: 15
    
    # Stage 5: Enhance answers
    - type: answer_enhancer
      config: {}
    
    # Stage 6: Final limiting
    - type: result_limiter
      config:
        max_results: 10
```

---

### Example 3: Ensemble Reranking

```yaml
retrieval:
  type: hybrid
  top_k: 40
  fusion:
    method: rrf
    alpha: 0.5
    rrf_k: 60

embedding:
  strategy: hybrid
  dense:
    provider: google
    model: models/embedding-001
  sparse:
    provider: sparse
    model: Qdrant/bm25

qdrant:
  collection_name: my_collection
  dense_vector_name: dense
  sparse_vector_name: sparse

retrieval_pipeline:
  stages:
    - type: reranker
      config:
        model_type: ensemble
        rerankers:
          - model_type: cross_encoder
            model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
            weight: 0.4
          - model_type: bge
            model_name: BAAI/bge-reranker-base
            weight: 0.6
        top_k: 10
```

---

## Common Patterns

### Pattern 1: Speed-Optimized (Single Reranker)
**Use case**: Fast responses, good enough quality

```yaml
stages:
  - type: score_filter
    config:
      min_score: 0.05
  
  - type: reranker
    config:
      model_type: cross_encoder
      model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 10
```

### Pattern 2: Quality-Optimized (Multi-Stage)
**Use case**: Maximum quality, willing to trade speed

```yaml
stages:
  - type: duplicate_filter
    config:
      dedup_by: external_id
  
  - type: reranker
    config:
      model_type: multistage
      stage1:
        model_type: cross_encoder
        top_k: 30
      stage2:
        model_type: bge
        top_k: 10
  
  - type: result_limiter
    config:
      max_results: 10
```

### Pattern 3: Balanced (Ensemble)
**Use case**: Best of both worlds

```yaml
stages:
  - type: score_filter
    config:
      min_score: 0.01
  
  - type: reranker
    config:
      model_type: ensemble
      rerankers:
        - model_type: cross_encoder
          weight: 0.5
        - model_type: bge
          weight: 0.5
      top_k: 10
```

---

## Testing Your Configuration

```bash
# Test with a query
python bin/retrieval_pipeline.py \
  --config pipelines/configs/retrieval/your_config.yml \
  --query "How to handle Python exceptions?" \
  --show-content

# List all available configurations
python bin/retrieval_pipeline.py --list-configs
```

---

## Key Points to Remember

1. ✅ **`retrieval`, `embedding`, `qdrant`** must be at **ROOT level**
2. ✅ **`alpha` is REQUIRED** for hybrid retrieval
3. ✅ **Stages are processed in order** (top to bottom)
4. ✅ **Each stage reduces result count** (use `top_k` wisely)
5. ⚠️ **More stages = slower but higher quality**
6. ⚠️ **Rerankers are expensive** - use multi-stage for large candidate sets

---

## Troubleshooting

### Error: "Alpha parameter is required"
**Fix**: Add `alpha` to `fusion` config:
```yaml
retrieval:
  fusion:
    alpha: 0.5  # Add this!
```

### Error: "Embedding configuration is required"
**Fix**: Move `embedding` to root level (not nested in `retrieval_pipeline.retriever`)

### Error: "Unknown stage type: X"
**Fix**: Check supported stage types above. Only these are implemented:
- Filters: `score_filter`, `duplicate_filter`, `tag_filter`, `result_limiter`
- Rerankers: `cross_encoder`, `bge`, `multistage`, `ensemble`
- Post-processors: `answer_enhancer`

### Warning: "Could not create X reranker"
**Cause**: Missing dependencies (e.g., sentence-transformers for rerankers)
**Fix**: Install required packages:
```bash
pip install sentence-transformers
```
