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
            # Initialize sparse embedding - get the sparse embedding config
            from embedding.factory import get_embedder

            # Extract sparse embedding config from the main config
            embedding_section = self.config.get('embedding', {})
            if 'sparse' in embedding_section:
                embedding_config = embedding_section['sparse']
            else:
                # Default sparse embedding config
                embedding_config = {
                    'provider': 'sparse',
                    'model': 'Qdrant/bm25',
                    'vector_name': 'sparse'
                }

            self.embedding = get_embedder(embedding_config)

            # Initialize Qdrant database
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(config=self.config)

            # Store qdrant_db for direct API access
            self.qdrant_db = qdrant_db

            self._initialized = True
            logger.info(
                f"Sparse retriever initialized with embedding: {type(self.embedding).__name__}")

        except Exception as e:
            logger.error(
                f"Failed to initialize sparse retriever components: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False

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
                # For BM25/sparse embeddings that might not have embed_query
                query_vector = self.embedding.embed_documents([query])[0]

            # Convert sparse dict to Qdrant sparse vector format for named sparse vectors
            if isinstance(query_vector, dict):
                from qdrant_client.models import NamedSparseVector

                search_result = self.qdrant_db.client.search(
                    collection_name=self.qdrant_db.collection_name,
                    query_vector=NamedSparseVector(
                        name=self.qdrant_db.sparse_vector_name,
                        vector={"indices": list(query_vector.keys()), "values": list(
                            query_vector.values())}
                    ),
                    limit=k,
                    with_payload=True
                )
            else:
                # Dense vector format (list) - fallback
                from qdrant_client.models import NamedVector

                search_result = self.qdrant_db.client.search(
                    collection_name=self.qdrant_db.collection_name,
                    query_vector=NamedVector(
                        name=self.qdrant_db.sparse_vector_name,
                        vector=query_vector
                    ),
                    limit=k,
                    with_payload=True
                )

            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_result:
                payload = result.payload or {}

                # Create document with preserved external_id
                document = Document(
                    page_content=payload.get('page_content', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        # Ensure external_id is in metadata
                        'external_id': payload.get('external_id'),
                        # Also store the Qdrant UUID for reference
                        'qdrant_id': str(result.id)
                    }
                )

                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=result.score,
                    additional_metadata={
                        'search_type': 'sparse_similarity',
                        'embedding_model': type(self.embedding).__name__,
                        # Also add to retrieval metadata
                        'external_id': payload.get('external_id')
                    }
                )
                retrieval_results.append(retrieval_result)

            # Normalize scores for consistency
            retrieval_results = self._normalize_scores(retrieval_results)

            return retrieval_results

        except Exception as e:
            logger.error(f"Error during sparse search: {e}")
            import traceback
            traceback.print_exc()
            return []


# Backward compatibility alias
SparseRetriever = QdrantSparseRetriever
