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

#### BM25Reranker
Statistical reranking using BM25 algorithm.

```python
from components.rerankers import BM25Reranker

reranker = BM25Reranker(k1=1.5, b=0.75)
reranked = reranker.rerank(query="machine learning", results=search_results, top_k=10)
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
    aggregation="weighted_sum"
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

#### DiversityFilter
Ensures result diversity by removing similar documents.

```python
from components.filters import DiversityFilter

filter = DiversityFilter(
    similarity_threshold=0.85,
    max_similar_docs=2
)

diverse = filter.filter(query="machine learning", results=search_results)
```

### Post-Processors (filters.py)

#### DeduplicationPostProcessor
Removes duplicate or near-duplicate results.

```python
from components.filters import DeduplicationPostProcessor

processor = DeduplicationPostProcessor(
    similarity_threshold=0.95,
    keep_first=True
)

deduplicated = processor.process(query="machine learning", results=search_results)
```

---

## 📊 Available Components Summary

### Factory Methods:
- ✅ create_dense_pipeline(config) - Dense retrieval only
- ✅ create_hybrid_pipeline(config) - Dense + sparse retrieval
- ✅ create_reranked_pipeline(config, reranker_model) - With cross-encoder reranking
- ✅ create_from_config(config) - Full config-based pipeline

### Rerankers:
- ✅ CrossEncoderReranker - Transformer-based reranking
- ✅ BM25Reranker - Statistical reranking
- ✅ EnsembleReranker - Combine multiple rerankers
- ✅ CohereBReranker - Cohere API reranking
- ✅ BgeReranker - BGE model reranking

### Filters:
- ✅ ScoreFilter - Minimum score threshold
- ✅ MetadataFilter - Filter by metadata
- ✅ DiversityFilter - Ensure result diversity

### Post-Processors:
- ✅ DeduplicationPostProcessor - Remove duplicates

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
