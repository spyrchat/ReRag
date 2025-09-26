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

        # Require explicit configuration - NO DEFAULTS
        if 'embedding' not in config:
            raise ValueError(
                "Embedding configuration is required for sparse retriever")
        if 'qdrant' not in config:
            raise ValueError(
                "Qdrant configuration is required for sparse retriever")

        # Use ONLY the provided embedding config
        embedding_config = config['embedding']
        if config.get('embedding', {}).get('strategy') == 'sparse':
            # For hybrid scenarios, extract sparse config
            if 'sparse' not in embedding_config:
                raise ValueError("Sparse embedding configuration is required")
            self.embedding_config = embedding_config['sparse']
        else:
            # For pure sparse scenarios, use the entire embedding config
            self.embedding_config = embedding_config

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

            # Perform Qdrant search
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

            # Convert to RetrievalResult objects (for both branches)
            results = []
            for result in search_result:
                payload = result.payload or {}
                document = Document(
                    page_content=payload.get('page_content', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        'external_id': payload.get('external_id'),
                        'qdrant_id': str(result.id),
                        'chunk_id': payload.get('chunk_id')
                    }
                )
                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=result.score,
                    additional_metadata={
                        'search_type': 'sparse_component',
                        'external_id': payload.get('external_id')
                    }
                )
                results.append(retrieval_result)

            # Normalize scores for consistency
            results = self._normalize_scores(results)
            return results

        except Exception as e:
            logger.error(f"Error during sparse search: {e}")
            import traceback
            traceback.print_exc()
            return []


# Backward compatibility alias
SparseRetriever = QdrantSparseRetriever
