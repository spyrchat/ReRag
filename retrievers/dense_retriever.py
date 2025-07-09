from qdrant_client import QdrantClient
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from typing import List, Tuple

from .base import BaseRetriever
from logs.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantDenseRetriever(BaseRetriever):
    """
    Retriever for dense vector search using Qdrant and LangChain.
    Returns top-k most similar documents and scores for a given query.
    """

    def __init__(
        self,
        embedding: Embeddings,
        vectorstore: QdrantVectorStore,
        top_k: int = 5,
    ):
        """
        Initialize the retriever with embedding model and Qdrant vectorstore.
        Args:
            embedding (Embeddings): The dense embedding model.
            vectorstore (QdrantVectorStore): LangChain-compatible Qdrant vector store.
            top_k (int): Default number of hits to return.
        """
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.top_k = top_k
        logger.info(
            f"QdrantDenseRetriever initialized (top_k={self.top_k})"
        )

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Perform a dense similarity search in Qdrant for the given query string.
        Args:
            query (str): The query string to embed and search.
            k (int, optional): Number of results to return. Defaults to self.top_k.
        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples.
        """
        logger.info(
            f"Retrieving dense results for query: '{query}' (top_k={k or self.top_k})")
        results = self.vectorstore.similarity_search_with_score(
            query, k=k or self.top_k)
        logger.info(f"Retrieved {len(results)} documents for query.")
        return results

    def get_relevant_documents(self, query: str) -> List[Tuple[Document, float]]:
        """
        Compatibility method for LangChain retriever interface.
        Args:
            query (str): The query string.
        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples.
        """
        return self.retrieve(query)
