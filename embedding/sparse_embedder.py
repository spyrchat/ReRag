import logging
from typing import List, Dict
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SparseEmbedder(Embeddings):
    """
    Embedder that produces sparse vectors using FastEmbed.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en", device: str = "cuda"):
        """
        Args:
            model_name (str): Name of the sparse model to load (e.g., BGE sparse models).
            device (str): Device to run the model ("cpu" or "cuda").
        """
        self.model = TextEmbedding(
            model_name=model_name,
            embedding_type="sparse",
            device=device
        )
        logger.info(
            f"Initialized SparseEmbedder with model: {model_name}, device: {device}")

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Embed multiple texts into sparse format (dictionary of token_id -> weight).

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[Dict[int, float]]: List of sparse embeddings (one per text).
        """
        logger.info(f"Embedding {len(texts)} documents (sparse).")
        return list(self.model.embed(texts))

    def embed_query(self, text: str) -> Dict[int, float]:
        """
        Embed a single query text into sparse format.

        Args:
            text (str): Single text query.

        Returns:
            Dict[int, float]: Sparse vector for query.
        """
        return next(self.model.embed([text]))
