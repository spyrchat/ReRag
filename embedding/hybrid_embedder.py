from typing import List, Tuple
from langchain.embeddings.base import Embeddings


class HybridEmbedder:
    def __init__(self, dense_embedder: Embeddings, sparse_embedder):
        """
        Wraps both a dense and sparse embedder.

        Args:
            dense_embedder (Embeddings): LangChain-compatible dense model
            sparse_embedder (object): Must have .get_vectors(texts: List[str]) -> List[Dict]
        """
        self.dense = dense_embedder
        self.sparse = sparse_embedder

    def embed_documents(self, texts: List[str]) -> Tuple[List[List[float]], List[dict]]:
        """
        Returns dense and sparse embeddings for a list of texts.
        """
        dense_vectors = self.dense.embed_documents(texts)
        sparse_vectors = self.sparse.get_vectors(texts)
        return dense_vectors, sparse_vectors
