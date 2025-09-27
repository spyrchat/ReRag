import logging
from typing import List, Dict
from fastembed import SparseTextEmbedding
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
    Embedder that produces sparse vectors using HuggingFace SPLADE SparseEncoder.
    """

    def __init__(self, model_name: str = "naver/splade-v3", device: str = "cpu"):
        from sentence_transformers import SparseEncoder
        self.model = SparseEncoder(model_name, device=device)
        self.model_name = model_name
        logger.info(f"Initialized SpladeEmbedder with model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        logger.info(f"Embedding {len(texts)} documents (sparse, SPLADE).")
        embs = self.model.encode_documents(texts)
        # embs is a list of scipy sparse matrices or dicts
        out = []
        for emb in embs:
            if hasattr(emb, 'nonzero'):
                indices = emb.nonzero()[1]
                values = emb.data
                out.append({int(idx): float(val)
                           for idx, val in zip(indices, values)})
            elif hasattr(emb, 'items'):
                out.append(dict(emb))
            else:
                out.append({i: float(v)
                           for i, v in enumerate(emb) if v != 0.0})
        return out

    def embed_query(self, text: str) -> Dict[int, float]:
        emb = self.model.encode_query([text])[0]
        if hasattr(emb, 'nonzero'):
            indices = emb.nonzero()[1]
            values = emb.data
            return {int(idx): float(val) for idx, val in zip(indices, values)}
        elif hasattr(emb, 'items'):
            return dict(emb)
        else:
            return {i: float(v) for i, v in enumerate(emb) if v != 0.0}
