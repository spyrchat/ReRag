# Core Components

**Last Verified:** 2025-10-08  
**Status:** ✅ Verified against actual codebase

Modular retrieval components providing reranking, filtering, and pipeline orchestration capabilities.

---

## 📁 Module Structure

```
components/
├── README.md                    # This file
├── retrieval_pipeline.py        # Pipeline orchestration and factory
├── rerankers.py                 # Basic rerankers (CrossEncoder, BM25, Ensemble)
├── advanced_rerankers.py        # Advanced rerankers (Cohere, BGE)
├── filters.py                   # Filters and post-processors
└── __init__.py                  # Module exports
```

---

## 🏭 RetrievalPipelineFactory

Factory class for creating retrieval pipelines with different strategies.

### Available Methods

#### 1. create_dense_pipeline(config)
Creates a dense-only retrieval pipeline using QdrantDenseRetriever.

```python
from components.retrieval_pipeline import RetrievalPipelineFactory

config = {
    'qdrant': {'collection': 'my_collection'},
    'embedding': {'model': 'text-embedding-3-small'}
}

pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
results = pipeline.search(query="machine learning", top_k=10)
```

#### 2. create_hybrid_pipeline(config)
Creates a hybrid retrieval pipeline (dense + sparse) using QdrantHybridRetriever.

```python
from components.retrieval_pipeline import RetrievalPipelineFactory

config = {
    'qdrant': {'collection': 'my_collection'},
    'embedding': {
        'model': 'text-embedding-3-small',
        'sparse': {'enabled': True}
    }
}

pipeline = RetrievalPipelineFactory.create_hybrid_pipeline(config)
results = pipeline.search(query="machine learning", top_k=10)
```

#### 3. create_reranked_pipeline(config, reranker_model=None)
Creates a pipeline with retrieval + cross-encoder reranking.

```python
from components.retrieval_pipeline import RetrievalPipelineFactory

config = {
    'qdrant': {'collection': 'my_collection'},
    'embedding': {'model': 'text-embedding-3-small'}
}

pipeline = RetrievalPipelineFactory.create_reranked_pipeline(
    config,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

results = pipeline.search(query="machine learning", top_k=10)
```

#### 4. create_from_config(config)
Creates a pipeline from a detailed YAML-style configuration.

```python
from components.retrieval_pipeline import RetrievalPipelineFactory

config = {
    'qdrant': {'collection': 'my_collection'},
    'retrieval_pipeline': {
        'retriever': {
            'type': 'hybrid',
            'top_k': 20
        },
        'stages': [
            {
                'type': 'score_filter',
                'config': {'min_score': 0.3}
            },
            {
                'type': 'reranker',
                'config': {
                    'model_type': 'cross_encoder',
                    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    'top_k': 10
                }
            }
        ]
    }
}

pipeline = RetrievalPipelineFactory.create_from_config(config)
results = pipeline.search(query="machine learning")
```

---

## 🔄 Rerankers

### Basic Rerankers (rerankers.py)

#### CrossEncoderReranker
Uses transformer cross-encoder models for passage ranking.

```python
from components.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
    top_k=10
)

reranked = reranker.rerank(
    query="machine learning algorithms",
    results=search_results
)
```

#### SemanticReranker
Semantic-aware reranking for better understanding of query intent.

```python
from components.rerankers import SemanticReranker

reranker = SemanticReranker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k=10
)
reranked = reranker.rerank(query="machine learning", results=search_results)
```

#### BM25Reranker
Statistical reranking using BM25 algorithm.

```python
from components.rerankers import BM25Reranker

reranker = BM25Reranker(k1=1.5, b=0.75, top_k=10)
reranked = reranker.rerank(query="machine learning", results=search_results)
```

#### EnsembleReranker
Combines multiple rerankers with weighted voting.

```python
from components.rerankers import EnsembleReranker, CrossEncoderReranker, BM25Reranker

ensemble = EnsembleReranker(
    rerankers=[
        CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        BM25Reranker(k1=1.5, b=0.75)
    ],
    weights=[0.7, 0.3],
    aggregation="weighted_sum",
    top_k=10
)

reranked = ensemble.rerank(query="machine learning", results=search_results)
```

### Advanced Rerankers (advanced_rerankers.py)

#### CohereBReranker
Commercial API-based reranking using Cohere models.

```python
from components.advanced_rerankers import CohereBReranker

reranker = CohereBReranker(
    api_key="your-cohere-api-key",
    model="rerank-english-v2.0",
    top_k=10
)

reranked = reranker.rerank(query="machine learning", results=search_results)
```

#### BgeReranker
BGE (BAAI General Embedding) reranker for multilingual support.

```python
from components.advanced_rerankers import BgeReranker

reranker = BgeReranker(
    model_name="BAAI/bge-reranker-base",
    device="cpu",
    top_k=10
)

reranked = reranker.rerank(query="machine learning", results=search_results)
```

#### ColBERTReranker
Late-interaction reranking using ColBERT models.

```python
from components.advanced_rerankers import ColBERTReranker

reranker = ColBERTReranker(
    model_name="colbert-ir/colbertv2.0",
    top_k=10
)

reranked = reranker.rerank(query="machine learning", results=search_results)
```

#### MultiStageReranker
Multi-stage progressive reranking for efficiency.

```python
from components.advanced_rerankers import MultiStageReranker
from components.rerankers import BM25Reranker, CrossEncoderReranker

reranker = MultiStageReranker(
    stages=[
        BM25Reranker(top_k=50),
        CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=10)
    ]
)

reranked = reranker.rerank(query="machine learning", results=search_results)
```

---

## 🔍 Filters and Post-Processors

### Filters (filters.py)

#### ScoreFilter
Filters results below a minimum score threshold.

```python
from components.filters import ScoreFilter

filter = ScoreFilter(min_score=0.5)
filtered = filter.filter(query="machine learning", results=search_results)
```

#### MetadataFilter
Filters results based on metadata criteria.

```python
from components.filters import MetadataFilter

filter = MetadataFilter(
    filter_criteria={
        'language': 'python',
        'category': ['tutorial', 'documentation']
    }
)

filtered = filter.filter(query="machine learning", results=search_results)
```

#### TagFilter
Filters results based on required or excluded tags.

```python
from components.filters import TagFilter

filter = TagFilter(
    required_tags=['python', 'machine-learning'],
    excluded_tags=['deprecated']
)

filtered = filter.filter(query="machine learning", results=search_results)
```

#### DuplicateFilter
Removes duplicate results based on external_id or content.

```python
from components.filters import DuplicateFilter

filter = DuplicateFilter(dedup_by="external_id")  # or "content" or "both"
deduplicated = filter.filter(query="machine learning", results=search_results)
```

### Post-Processors (filters.py)

#### AnswerEnhancer
Enhances answer formatting and metadata.

```python
from components.filters import AnswerEnhancer

enhancer = AnswerEnhancer()
enhanced = enhancer.post_process(query="machine learning", results=search_results)
```

#### ContextEnricher
Enriches results with additional context information.

```python
from components.filters import ContextEnricher

enricher = ContextEnricher()
enriched = enricher.post_process(query="machine learning", results=search_results)
```

#### ResultLimiter
Limits the number of results returned.

```python
from components.filters import ResultLimiter

limiter = ResultLimiter(max_results=10)
limited = limiter.post_process(query="machine learning", results=search_results)
```

---

## 📊 Available Components Summary

### Factory Methods:
- ✅ create_dense_pipeline(config) - Dense retrieval only
- ✅ create_sparse_pipeline(config) - Sparse retrieval only
- ✅ create_hybrid_pipeline(config) - Dense + sparse retrieval
- ✅ create_semantic_pipeline(config) - Semantic retrieval with intelligent routing
- ✅ create_reranked_pipeline(config, reranker_model) - With cross-encoder reranking
- ✅ create_from_config(config) - Full config-based pipeline
- ✅ create_from_retriever_config(retriever_type, global_config) - From retriever config file
- ✅ create_from_unified_config(config, retrieval_type) - From simplified config

### Rerankers (rerankers.py):
- ✅ CrossEncoderReranker - Transformer-based reranking
- ✅ SemanticReranker - Semantic-aware reranking
- ✅ BM25Reranker - Statistical reranking (BM25 algorithm)
- ✅ EnsembleReranker - Combine multiple rerankers with weighted voting

### Advanced Rerankers (advanced_rerankers.py):
- ✅ CohereBReranker - Cohere API reranking (commercial)
- ✅ BgeReranker - BGE model reranking (multilingual)
- ✅ ColBERTReranker - ColBERT late-interaction reranking
- ✅ MultiStageReranker - Multi-stage progressive reranking

### Filters:
- ✅ ScoreFilter - Minimum score threshold
- ✅ MetadataFilter - Filter by metadata criteria
- ✅ TagFilter - Filter by required/excluded tags
- ✅ DuplicateFilter - Remove duplicate results

### Post-Processors:
- ✅ AnswerEnhancer - Enhance answer formatting and metadata
- ✅ ContextEnricher - Enrich results with additional context
- ✅ ResultLimiter - Limit number of results

---

## 📝 Notes

- All classes and methods listed above have been **verified against the actual codebase**
- Examples use real configuration patterns from the project
- See components/__init__.py for full exports list
- See individual module files for detailed docstrings

---

**Related Documentation:**
- [Retrievers](../retrievers/README.md) - Base retrieval implementations
- [Database](../database/README.md) - Vector storage
- [Pipelines](../pipelines/README.md) - Data ingestion
