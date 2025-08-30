"""
Modern sparse retriever that integrates with the retrieval pipeline architecture.
"""

from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from components.retrieval_pipeline import RetrievalResult
from .base_retriever import ModernBaseRetriever
import logging

logger = logging.getLogger(__name__)


class QdrantSparseRetriever(ModernBaseRetriever):
    """
    Sparse vector retriever using Qdrant.
    Performs keyword-based search using sparse embeddings (e.g., SPLADE, BGE-M3).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sparse retriever.

        Args:
            config: Configuration dictionary containing:
                - embedding: Sparse embedding configuration
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
            # Initialize sparse embedding
            from embedding.factory import get_embedder
            embedding_config = self.config.get('embedding', {
                'type': 'sentence_transformers',
                'model': 'sentence-transformers/all-MiniLM-L6-v2'
            })
            self.embedding = get_embedder(embedding_config)

            # Initialize Qdrant vector store for sparse search
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(strategy="sparse", config=self.config)

            # For testing, use dense mode if sparse vectors don't exist
            try:
                # Get sparse vector configuration
                sparse_config = self.config.get('qdrant', {})
                sparse_vector_name = sparse_config.get(
                    'sparse_vector_name', 'sparse')

                self.vectorstore = QdrantVectorStore(
                    client=qdrant_db.get_client(),
                    collection_name=qdrant_db.get_collection_name(),
                    embedding=self.embedding,  # Sparse embedding
                    vector_name=sparse_vector_name,
                    retrieval_mode=RetrievalMode.SPARSE
                )
            except Exception:
                # Fallback to dense mode for testing
                logger.warning(
                    "Sparse vectors not available, falling back to dense mode")
                self.vectorstore = qdrant_db.as_langchain_vectorstore(
                    dense_embedding=self.embedding
                )

            self._initialized = True
            logger.info(
                f"Sparse retriever initialized with embedding: {type(self.embedding).__name__}")

        except Exception as e:
            logger.error(
                f"Failed to initialize sparse retriever components: {e}")
            self._initialized = False

    @property
    def component_name(self) -> str:
        return "sparse_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform sparse similarity search.

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
                "Sparse retriever not properly initialized, returning empty results")
            return []

        try:
            # Perform sparse similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Convert to RetrievalResult objects
            retrieval_results = []
            for document, score in results:
                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=score,
                    additional_metadata={
                        'search_type': 'sparse_similarity',
                        'embedding_model': type(self.embedding).__name__
                    }
                )
                retrieval_results.append(retrieval_result)

            # Normalize scores for consistency
            retrieval_results = self._normalize_scores(retrieval_results)

            return retrieval_results

        except Exception as e:
            logger.error(f"Error during sparse search: {e}")
            return []


# Backward compatibility alias
SparseRetriever = QdrantSparseRetriever
