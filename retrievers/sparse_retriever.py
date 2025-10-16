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
        """Initialize sparse retriever with ONLY explicit configuration."""
        super().__init__(config)

        if 'embedding' not in config:
            raise ValueError(
                "Embedding configuration is required for sparse retriever")
        if 'qdrant' not in config:
            raise ValueError(
                "Qdrant configuration is required for sparse retriever")

        # Always use the provided embedding config as a flat dict
        self.embedding_config = config['embedding']
        self.embedding = None
        self._initialized = False

    def _initialize_components(self):
        """Initialize components using ONLY explicit configuration."""
        if self._initialized:
            return

        try:
            from embedding.factory import get_embedder

            # Use exact embedding configuration provided
            self.embedding = get_embedder(self.embedding_config)

            # Initialize Qdrant with exact config
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(config=self.config)
            self.qdrant_db = qdrant_db

            self._initialized = True
            logger.info(
                "Sparse retriever initialized with explicit configuration only")

        except Exception as e:
            logger.error(f"Failed to initialize sparse retriever: {e}")
            raise

    @property
    def component_name(self) -> str:
        return "sparse_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform sparse similarity search using direct Qdrant API to preserve external_id.

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
            # Get sparse query vector
            if hasattr(self.embedding, 'embed_query'):
                query_vector = self.embedding.embed_query(query)
            else:
                query_vector = self.embedding.embed_documents([query])[0]

            # CRITICAL: Validate that we received a sparse vector (dictionary)
            if not isinstance(query_vector, dict):
                raise TypeError(
                    f"Sparse retriever requires dictionary embedding, but received {type(query_vector)}. "
                    f"This indicates a dense embedding model is configured instead of sparse. "
                    f"Check your embedding configuration."
                )

            # # Log sparse vector statistics for debugging
            # logger.info(
            #     f"[SPARSE_RETRIEVER] Query vector has {len(query_vector)} non-zero elements"
            # )
            # logger.debug(
            #     f"[SPARSE_RETRIEVER] Sample indices: {list(query_vector.keys())[:10]}"
            # )

            # Perform Qdrant sparse search using Query API
            from qdrant_client.models import SparseVector

            query_response = self.qdrant_db.client.query_points(
                collection_name=self.qdrant_db.collection_name,
                query=SparseVector(
                    indices=list(query_vector.keys()),
                    values=list(query_vector.values())
                ),
                using=self.qdrant_db.sparse_vector_name,
                limit=k,
                with_payload=True
            )

            # Convert query_points response to RetrievalResult objects
            # IMPORTANT: query_points returns QueryResponse with .points attribute
            results = []
            for point in query_response.points:
                payload = point.payload or {}

                document = Document(
                    page_content=payload.get('text', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        'external_id': payload.get('external_id'),
                        'qdrant_id': str(point.id),
                        'chunk_id': payload.get('chunk_id')
                    }
                )

                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=point.score,
                    additional_metadata={
                        'search_type': 'sparse_vector',
                        'embedding_model': type(self.embedding).__name__,
                        'external_id': payload.get('external_id'),
                        'non_zero_elements': len(query_vector)
                    }
                )
                results.append(retrieval_result)

            # logger.info(
            #     f"[SPARSE_RETRIEVER] Retrieved {len(results)} results for query"
            # )

            # Normalize scores for consistency (optional, depends on your pipeline)
            # results = self._normalize_scores(results)

            return results

        except TypeError as e:
            logger.error(f"Configuration error in sparse retriever: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during sparse search: {e}")
            import traceback
            traceback.print_exc()
            return []


# Backward compatibility alias
SparseRetriever = QdrantSparseRetriever
