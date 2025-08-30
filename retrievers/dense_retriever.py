"""
Modern dense retriever that integrates with the retrieval pipeline architecture.
"""

from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from components.retrieval_pipeline import RetrievalResult
from .base_retriever import ModernBaseRetriever
import logging

logger = logging.getLogger(__name__)


class QdrantDenseRetriever(ModernBaseRetriever):
    """
    Dense vector retriever using Qdrant and LangChain.
    Performs semantic similarity search using dense embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dense retriever.

        Args:
            config: Configuration dictionary containing:
                - embedding: Dense embedding configuration
                - qdrant: Qdrant database configuration
                - top_k: Number of results to retrieve (default: 5)
                - score_threshold: Minimum score threshold (default: 0.0)
        """
        super().__init__(config)

        # Validate required configuration
        if 'embedding' not in config:
            logger.warning("No embedding config found, using default")
        if 'qdrant' not in config:
            logger.warning("No qdrant config found, using default")

        # Initialize components lazily
        self.embedding = None
        self.vectorstore = None
        self._initialized = False

    def _initialize_components(self):
        """Initialize embedding and vector store components."""
        if self._initialized:
            return

        try:
            # Initialize embedding
            from embedding.factory import get_embedder
            embedding_config = self.config.get('embedding', {
                'type': 'sentence_transformers',
                'model': 'sentence-transformers/all-MiniLM-L6-v2'
            })
            self.embedding = get_embedder(embedding_config)

            # Initialize Qdrant vector store
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(config=self.config)
            self.vectorstore = qdrant_db.as_langchain_vectorstore(
                dense_embedding=self.embedding
            )

            self._initialized = True
            logger.info(
                f"Dense retriever initialized with embedding: {type(self.embedding).__name__}")

        except Exception as e:
            logger.error(
                f"Failed to initialize dense retriever components: {e}")
            # Don't raise, just mark as failed to initialize
            self._initialized = False

    @property
    def component_name(self) -> str:
        return "dense_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform dense similarity search.

        Args:
            query: Search query
            k: Number of results to retrieve

        Returns:
            List of RetrievalResult objects
        """
        if not self._initialized:
            self._initialize_components()

        if not self._initialized:
            logger.warning(
                "Dense retriever not properly initialized, returning empty results")
            return []

        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Convert to RetrievalResult objects
            retrieval_results = []
            for document, score in results:
                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=score,
                    additional_metadata={
                        'search_type': 'dense_similarity',
                        'embedding_model': type(self.embedding).__name__
                    }
                )
                retrieval_results.append(retrieval_result)

            # Normalize scores for consistency
            retrieval_results = self._normalize_scores(retrieval_results)

            return retrieval_results

        except Exception as e:
            logger.error(f"Error during dense search: {e}")
            return []


# Backward compatibility alias
DenseRetriever = QdrantDenseRetriever
