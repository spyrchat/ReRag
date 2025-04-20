import os
from typing import List

from embedding.hybrid_embedder import HybridEmbedder
from embedding.bm25_sparse_embedder import BM25SparseEmbedder
from embedding.bedrock_embeddings import TitanEmbedder, TitanLangchainWrapper

# If needed later:
# from embedding.openai_embedder import OpenAIEmbedderWrapper
# from embedding.instructor_embedder import InstructorWrapper


def get_dense_embedder(model_name: str):
    if model_name == "titan":
        return TitanLangchainWrapper(TitanEmbedder())
    # elif model_name == "openai":
    #     return OpenAIEmbedderWrapper(...)
    # elif model_name == "instructor":
    #     return InstructorWrapper(...)
    else:
        raise ValueError(f"Unsupported dense embedder: {model_name}")


def get_sparse_embedder(model_name: str, corpus: List[str]):
    if model_name == "bm25":
        return BM25SparseEmbedder(corpus=corpus)
    # elif model_name == "splade":
    #     return SPLADESparseEmbedder(corpus=corpus)
    else:
        raise ValueError(f"Unsupported sparse embedder: {model_name}")


def get_embedder(texts: List[str]):
    strategy = os.getenv("EMBEDDING_STRATEGY", "hybrid")
    dense_model = os.getenv("DENSE_EMBEDDER", "titan")
    sparse_model = os.getenv("SPARSE_EMBEDDER", "bm25")

    if strategy == "dense":
        return get_dense_embedder(dense_model)

    elif strategy == "sparse":
        return get_sparse_embedder(sparse_model, corpus=texts)

    elif strategy == "hybrid":
        dense = get_dense_embedder(dense_model)
        sparse = get_sparse_embedder(sparse_model, corpus=texts)
        return HybridEmbedder(dense_embedder=dense, sparse_embedder=sparse)

    else:
        raise ValueError(f"Unknown EMBEDDING_STRATEGY: {strategy}")
