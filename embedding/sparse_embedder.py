import logging
from typing import List, Dict
from fastembed import SparseTextEmbedding, SparseEmbedding
from langchain_core.embeddings import Embeddings
import abc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SparseEmbedder(Embeddings, abc.ABC):
    """
    Abstract base class for all sparse embedders (BM25, SPLADE, etc).
    Defines the interface for embed_documents and embed_query.
    """
    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> Dict[int, float]:
        pass


class BM25Embedder(SparseEmbedder):
    """
    Embedder that produces sparse vectors using FastEmbed SparseTextEmbedding (BM25, etc).
    """

    def __init__(self, model_name: str = "Qdrant/bm25", device: str = "cpu"):
        self.model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name
        logger.info(f"Initialized BM25Embedder with model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        logger.info(f"Embedding {len(texts)} documents (sparse, BM25).")
        embeddings = []
        for embedding in self.model.embed(texts):
            sparse_dict = {int(idx): float(val) for idx, val in zip(
                embedding.indices, embedding.values)}
            embeddings.append(sparse_dict)
        return embeddings

    def embed_query(self, text: str) -> Dict[int, float]:
        embeddings = self.embed_documents([text])
        return embeddings[0]


class SpladeEmbedder(SparseEmbedder):
    """
    Embedder that produces sparse vectors using FastEmbed SparseTextEmbedding (Qdrant's SPLADE models).
    """

    def __init__(self, model_name: str = "Qdrant/splade-cocondenser-ensembledistil", device: str = "cpu"):
        self.model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name
        logger.info(f"Initialized SpladeEmbedder with model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        logger.info(
            f"Embedding {len(texts)} documents (sparse, FastEmbed SPLADE).")
        embeddings = []
        for embedding in self.model.embed(texts):
            sparse_dict = {int(idx): float(val) for idx, val in zip(
                embedding.indices, embedding.values)}
            embeddings.append(sparse_dict)
        return embeddings

    def embed_query(self, text: str) -> Dict[int, float]:
        embeddings = self.embed_documents([text])
        sparse_vec = embeddings[0]
        logger.info(f"[DEBUG][SPLADE] Query: {text}")
        logger.info(
            f"[DEBUG][SPLADE] Sparse vector length: {len(sparse_vec)} | Nonzero keys: {list(sparse_vec.keys())[:10]}")
        return sparse_vec
