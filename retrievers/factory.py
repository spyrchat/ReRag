from typing import Dict, Any
from bm25_retriever import get_bm25_retriever
from dense_retriever import get_qdrant_retriever
from hybrid_retriever import HybridRetriever
from base import BaseRetriever


def get_retriever(strategy: str, **kwargs) -> BaseRetriever:
    if strategy == "dense":
        return get_qdrant_retriever(**kwargs)
    elif strategy == "sparse":
        return get_bm25_retriever(**kwargs)
    elif strategy == "hybrid":
        return HybridRetriever(
            dense_retriever=get_qdrant_retriever(**kwargs),
            sparse_retriever=get_bm25_retriever(**kwargs)
        )
    else:
        raise ValueError(f"Unknown retriever strategy: {strategy}")
