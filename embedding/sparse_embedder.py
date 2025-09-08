import logging
from typing import List, Dict
from fastembed import SparseTextEmbedding
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SparseEmbedder(Embeddings):
    """
    Embedder that produces sparse vectors using FastEmbed SparseTextEmbedding.
    """

    def __init__(self, model_name: str = "Qdrant/bm25", device: str = "cpu"):
        """
        Args:
            model_name (str): Name of the sparse model to load (e.g., "Qdrant/bm25").
            device (str): Device to run the model ("cpu" or "cuda").
        """
        self.model = SparseTextEmbedding(
            model_name=model_name
        )
        self.model_name = model_name
        logger.info(
            f"Initialized SparseEmbedder with model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Embed multiple texts into sparse format (dictionary of token_id -> weight).

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[Dict[int, float]]: List of sparse embeddings (one per text).
        """
        logger.info(f"Embedding {len(texts)} documents (sparse).")
        embeddings = []
        for embedding in self.model.embed(texts):
            # Convert SparseEmbedding to dict
            sparse_dict = {}
            for idx, val in zip(embedding.indices, embedding.values):
                sparse_dict[int(idx)] = float(val)
            embeddings.append(sparse_dict)
        return embeddings

    def embed_query(self, text: str) -> Dict[int, float]:
        """
        Embed a single query text into sparse format.

        Args:
            text (str): Single text query.

        Returns:
            Dict[int, float]: Sparse vector for query.
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]
