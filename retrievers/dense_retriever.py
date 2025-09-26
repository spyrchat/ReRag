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
from retrievers.base_retriever import RetrievalResult
logger = logging.getLogger(__name__)


class QdrantDenseRetriever(ModernBaseRetriever):
    """
    Dense vector retriever using Qdrant and LangChain.
    Performs semantic similarity search using dense embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize dense retriever with ONLY explicit configuration."""
        super().__init__(config)

        # Require explicit embedding configuration - NO DEFAULTS
        if 'embedding' not in config:
            raise ValueError(
                "Embedding configuration is required for dense retriever")
        if 'qdrant' not in config:
            raise ValueError(
                "Qdrant configuration is required for dense retriever")

        # Use ONLY the provided embedding config
        embedding_config = config['embedding']
        if config.get('embedding', {}).get('strategy') == 'dense':
            # For hybrid scenarios, extract dense config
            if 'dense' not in embedding_config:
                raise ValueError("Dense embedding configuration is required")
            self.embedding_config = embedding_config['dense']
        else:
            # For pure dense scenarios, use the entire embedding config
            self.embedding_config = embedding_config

        self.embedding = None
        self.vectorstore = None
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
                "Dense retriever initialized with explicit configuration only")

        except Exception as e:
            logger.error(f"Failed to initialize dense retriever: {e}")
            raise

    @property
    def component_name(self) -> str:
        return "dense_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform dense similarity search using direct Qdrant API to preserve external_id.

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
            # Get query embedding
            query_vector = self.embedding.embed_query(query)

            # Get Qdrant database instance
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(config=self.config)

            # Direct Qdrant search to preserve external_id
            from qdrant_client.models import NamedVector

            search_result = qdrant_db.client.search(
                collection_name=qdrant_db.collection_name,
                query_vector=NamedVector(
                    name=qdrant_db.dense_vector_name,
                    vector=query_vector
                ),
                limit=k,
                with_payload=True  # Include all payload data including external_id
            )

            # Convert to RetrievalResult objects
            retrieval_results = []
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
                        'search_type': 'dense_similarity',
                        'embedding_model': type(self.embedding).__name__,
                        'external_id': payload.get('external_id')
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
