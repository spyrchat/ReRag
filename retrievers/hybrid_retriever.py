"""
Modern hybrid retriever that integrates with the retrieval pipeline architecture.
"""

from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from components.retrieval_pipeline import RetrievalResult
from .base_retriever import ModernBaseRetriever
import logging

logger = logging.getLogger(__name__)


class QdrantHybridRetriever(ModernBaseRetriever):
    """
    Hybrid retriever combining dense and sparse vector search using Qdrant.
    Leverages both semantic similarity (dense) and keyword matching (sparse).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hybrid retriever.

        Args:
            config: Configuration dictionary containing:
                - embedding: Configuration with both dense and sparse embeddings
                - qdrant: Qdrant database configuration
                - top_k: Number of results to retrieve (default: 5)
                - score_threshold: Minimum score threshold (default: 0.0)
                - fusion_method: How to combine dense/sparse scores (default: 'rrf')
        """
        super().__init__(config)

        # Validate required configuration
        if 'embedding' not in config:
            logger.warning("No embedding config found, using defaults")
        if 'qdrant' not in config:
            logger.warning("No qdrant config found, using defaults")

        # Initialize components
        self.dense_embedding = None
        self.sparse_embedding = None
        self.vectorstore = None
        self.fusion_method = config.get(
            'fusion_method', 'rrf')  # Reciprocal Rank Fusion
        self._initialized = False

    def _initialize_components(self):
        """Initialize embeddings and vector store components."""
        if self._initialized:
            return

        try:
            # Initialize embeddings
            from embedding.factory import get_embedder

            embedding_config = self.config.get('embedding', {})

            # For testing, use same embedding for both dense and sparse if not specified
            if 'dense' in embedding_config and 'sparse' in embedding_config:
                self.dense_embedding = get_embedder(embedding_config['dense'])
                self.sparse_embedding = get_embedder(
                    embedding_config['sparse'])
            else:
                # Use same embedding for both - fallback for testing
                default_config = embedding_config if embedding_config else {
                    'type': 'sentence_transformers',
                    'model': 'sentence-transformers/all-MiniLM-L6-v2'
                }
                self.dense_embedding = get_embedder(default_config)
                self.sparse_embedding = get_embedder(default_config)

            # Initialize Qdrant vector store for hybrid search
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(strategy="hybrid", config=self.config)

            # Get vector configuration
            qdrant_config = self.config.get('qdrant', {})
            dense_vector_name = qdrant_config.get('dense_vector_name', 'dense')
            sparse_vector_name = qdrant_config.get(
                'sparse_vector_name', 'sparse')

            try:
                self.vectorstore = QdrantVectorStore(
                    client=qdrant_db.get_client(),
                    collection_name=qdrant_db.get_collection_name(),
                    embedding=self.dense_embedding,
                    vector_name=dense_vector_name,
                    sparse_embedding=self.sparse_embedding,
                    sparse_vector_name=sparse_vector_name,
                    retrieval_mode=RetrievalMode.HYBRID
                )
            except Exception:
                # Fallback to dense mode if hybrid is not available
                logger.warning(
                    "Hybrid mode not available, falling back to dense mode")
                self.vectorstore = qdrant_db.as_langchain_vectorstore(
                    dense_embedding=self.dense_embedding
                )

            self._initialized = True
            logger.info(f"Hybrid retriever initialized with dense: {type(self.dense_embedding).__name__}, "
                        f"sparse: {type(self.sparse_embedding).__name__}")

        except Exception as e:
            logger.error(
                f"Failed to initialize hybrid retriever components: {e}")
            self._initialized = False

    @property
    def component_name(self) -> str:
        return "hybrid_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.

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
                "Hybrid retriever not properly initialized, returning empty results")
            return []

        try:
            # Perform hybrid similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Convert to RetrievalResult objects
            retrieval_results = []
            for document, score in results:
                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=score,
                    additional_metadata={
                        'search_type': 'hybrid_similarity',
                        'dense_embedding_model': type(self.dense_embedding).__name__,
                        'sparse_embedding_model': type(self.sparse_embedding).__name__,
                        'fusion_method': self.fusion_method
                    }
                )
                retrieval_results.append(retrieval_result)

            # Normalize scores for consistency
            retrieval_results = self._normalize_scores(retrieval_results)

            return retrieval_results

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return []

    def _perform_separate_searches(self, query: str, k: int) -> Dict[str, List[RetrievalResult]]:
        """
        Perform separate dense and sparse searches for advanced fusion.

        Args:
            query: Search query
            k: Number of results per search type

        Returns:
            Dictionary with 'dense' and 'sparse' search results
        """
        results = {'dense': [], 'sparse': []}

        try:
            # Dense search
            dense_config = self.config.copy()
            dense_config['embedding'] = self.config['embedding']['dense'] if 'dense' in self.config.get(
                'embedding', {}) else self.config.get('embedding', {})

            from .dense_retriever import QdrantDenseRetriever
            dense_retriever = QdrantDenseRetriever(dense_config)
            results['dense'] = dense_retriever._perform_search(query, k)

            # Sparse search
            sparse_config = self.config.copy()
            sparse_config['embedding'] = self.config['embedding']['sparse'] if 'sparse' in self.config.get(
                'embedding', {}) else self.config.get('embedding', {})

            from .sparse_retriever import QdrantSparseRetriever
            sparse_retriever = QdrantSparseRetriever(sparse_config)
            results['sparse'] = sparse_retriever._perform_search(query, k)

        except Exception as e:
            logger.warning(
                f"Separate searches failed, falling back to Qdrant hybrid: {e}")

        return results


# Backward compatibility alias
HybridRetriever = QdrantHybridRetriever
