"""
Modular components for extensible retrieval pipelines.
"""

from .retrieval_pipeline import (
    RetrievalPipeline,
    RetrievalPipelineFactory,
    RetrievalResult,
    RetrievalComponent,
    BaseRetriever,
    Reranker,
    ResultFilter,
    PostProcessor
)

from .rerankers import (
    CrossEncoderReranker,
    SemanticReranker,
    BM25Reranker,
    EnsembleReranker
)

from .filters import (
    ScoreFilter,
    MetadataFilter,
    TagFilter,
    DuplicateFilter,
    AnswerEnhancer,
    ContextEnricher,
    ResultLimiter
)

__all__ = [
    # Core pipeline
    'RetrievalPipeline',
    'RetrievalPipelineFactory',
    'RetrievalResult',
    'RetrievalComponent',
    'BaseRetriever',
    'Reranker',
    'ResultFilter',
    'PostProcessor',

    # Rerankers
    'CrossEncoderReranker',
    'SemanticReranker',
    'BM25Reranker',
    'EnsembleReranker',

    # Filters and processors
    'ScoreFilter',
    'MetadataFilter',
    'TagFilter',
    'DuplicateFilter',
    'AnswerEnhancer',
    'ContextEnricher',
    'ResultLimiter',
]
