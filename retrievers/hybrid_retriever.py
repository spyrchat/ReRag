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

        # Load fusion parameters from config
        fusion_config = config.get('fusion', {})
        self.rrf_k = fusion_config.get('rrf_k', 60)  # Standard RRF constant
        self.dense_weight = fusion_config.get('dense_weight', 0.5)
        self.sparse_weight = fusion_config.get('sparse_weight', 0.5)
        self._initialized = False

    def _initialize_components(self):
        """Initialize embeddings and vector store components."""
        if self._initialized:
            return

        try:
            # Initialize embeddings
            from embedding.factory import get_embedder

            embedding_section = self.config.get('embedding', {})

            # Extract dense and sparse embedding configs
            if 'dense' in embedding_section:
                dense_config = embedding_section['dense']
            else:
                # Default dense embedding config
                dense_config = {
                    'provider': 'google',
                    'model': 'models/embedding-001',
                    'dimensions': 768,
                    'api_key_env': 'GOOGLE_API_KEY'
                }

            if 'sparse' in embedding_section:
                sparse_config = embedding_section['sparse']
            else:
                # Default sparse embedding config
                sparse_config = {
                    'provider': 'sparse',
                    'model': 'Qdrant/bm25',
                    'vector_name': 'sparse'
                }

            self.dense_embedding = get_embedder(dense_config)
            self.sparse_embedding = get_embedder(sparse_config)

            # Initialize Qdrant database
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(config=self.config)

            # Store qdrant_db for direct API access
            self.qdrant_db = qdrant_db

            self._initialized = True
            logger.info(f"Hybrid retriever initialized with dense: {type(self.dense_embedding).__name__}, "
                        f"sparse: {type(self.sparse_embedding).__name__}")

        except Exception as e:
            logger.error(
                f"Failed to initialize hybrid retriever components: {e}")
            import traceback
            traceback.print_exc()
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
            # Perform separate dense and sparse searches then combine
            dense_results = self._perform_dense_search(query, k)
            sparse_results = self._perform_sparse_search(query, k)

            # Combine results using Reciprocal Rank Fusion (RRF)
            combined_results = self._fuse_results(
                dense_results, sparse_results, k)

            return combined_results

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _perform_dense_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Perform dense search using direct Qdrant API."""
        try:
            # Get dense query vector
            query_vector = self.dense_embedding.embed_query(query)

            # Direct Qdrant search for dense vectors
            from qdrant_client.models import NamedVector

            search_result = self.qdrant_db.client.search(
                collection_name=self.qdrant_db.collection_name,
                query_vector=NamedVector(
                    name=self.qdrant_db.dense_vector_name,
                    vector=query_vector
                ),
                limit=k,
                with_payload=True
            )

            # Convert to RetrievalResult objects
            results = []
            for result in search_result:
                payload = result.payload or {}

                document = Document(
                    page_content=payload.get('page_content', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        'external_id': payload.get('external_id'),
                        'qdrant_id': str(result.id)
                    }
                )

                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=result.score,
                    additional_metadata={
                        'search_type': 'dense_component',
                        'embedding_model': type(self.dense_embedding).__name__,
                        'external_id': payload.get('external_id')
                    }
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            logger.error(f"Dense search component failed: {e}")
            return []

    def _perform_sparse_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Perform sparse search using direct Qdrant API."""
        try:
            # Get sparse query vector
            if hasattr(self.sparse_embedding, 'embed_query'):
                query_vector = self.sparse_embedding.embed_query(query)
            else:
                query_vector = self.sparse_embedding.embed_documents([query])[
                    0]

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
            results = []
            for result in search_result:
                payload = result.payload or {}

                document = Document(
                    page_content=payload.get('page_content', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        'external_id': payload.get('external_id'),
                        'qdrant_id': str(result.id)
                    }
                )

                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=result.score,
                    additional_metadata={
                        'search_type': 'sparse_component',
                        'embedding_model': type(self.sparse_embedding).__name__,
                        'external_id': payload.get('external_id')
                    }
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            logger.error(f"Sparse search component failed: {e}")
            return []

    def _fuse_results(self, dense_results: List[RetrievalResult], sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Combine dense and sparse results using standard fusion methods."""
        try:
            if self.fusion_method == 'rrf':
                return self._fuse_with_rrf(dense_results, sparse_results, k)
            elif self.fusion_method == 'weighted_sum':
                return self._fuse_with_weighted_sum(dense_results, sparse_results, k)
            else:
                logger.warning(
                    f"Unknown fusion method: {self.fusion_method}, falling back to RRF")
                return self._fuse_with_rrf(dense_results, sparse_results, k)
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            # Fallback to dense results
            return dense_results[:k]

    def _fuse_with_rrf(self, dense_results: List[RetrievalResult], sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Standard Reciprocal Rank Fusion (Cormack et al. 2009)."""
        doc_scores = {}
        rrf_k = self.rrf_k  # Use configurable RRF constant

        # Add dense results with standard RRF scoring
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.document.metadata.get('external_id')
            if doc_id:
                rrf_score = 1.0 / (rrf_k + rank)  # Standard RRF formula
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'result': result, 'dense_score': rrf_score, 'sparse_score': 0}
                else:
                    doc_scores[doc_id]['dense_score'] = rrf_score

        # Add sparse results with standard RRF scoring
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.document.metadata.get('external_id')
            if doc_id:
                rrf_score = 1.0 / (rrf_k + rank)  # Standard RRF formula
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'result': result, 'dense_score': 0, 'sparse_score': rrf_score}
                else:
                    doc_scores[doc_id]['sparse_score'] = rrf_score

        # Combine scores and sort
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = scores['dense_score'] + scores['sparse_score']
            result = scores['result']
            result.score = combined_score
            result.metadata.update({
                'search_type': 'hybrid_rrf',
                'dense_rrf_score': scores['dense_score'],
                'sparse_rrf_score': scores['sparse_score'],
                'fusion_method': 'rrf',
                'rrf_k': rrf_k
            })
            combined_results.append(result)

        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]

    def _fuse_with_weighted_sum(self, dense_results: List[RetrievalResult], sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Weighted sum fusion with score normalization."""
        # Get weights from config
        dense_weight = self.dense_weight
        sparse_weight = self.sparse_weight

        # Normalize weights to sum to 1
        total_weight = dense_weight + sparse_weight
        if total_weight > 0:
            dense_weight /= total_weight
            sparse_weight /= total_weight
        else:
            dense_weight = sparse_weight = 0.5

        # Normalize scores using min-max normalization
        def normalize_scores(results):
            if not results:
                return {}
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score

            normalized = {}
            for result in results:
                doc_id = result.document.metadata.get('external_id')
                if doc_id and score_range > 0:
                    normalized[doc_id] = {
                        'result': result,
                        'score': (result.score - min_score) / score_range
                    }
                elif doc_id:
                    normalized[doc_id] = {'result': result, 'score': 1.0}
            return normalized

        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)

        # Combine normalized scores
        doc_scores = {}
        all_doc_ids = set(dense_normalized.keys()) | set(
            sparse_normalized.keys())

        for doc_id in all_doc_ids:
            dense_score = dense_normalized.get(doc_id, {}).get('score', 0.0)
            sparse_score = sparse_normalized.get(doc_id, {}).get('score', 0.0)

            combined_score = dense_weight * dense_score + sparse_weight * sparse_score

            # Use the result from whichever retriever found this document
            result = (dense_normalized.get(doc_id)
                      or sparse_normalized.get(doc_id))['result']
            result.score = combined_score
            result.metadata.update({
                'search_type': 'hybrid_weighted',
                'dense_weight': dense_weight,
                'sparse_weight': sparse_weight,
                'dense_norm_score': dense_score,
                'sparse_norm_score': sparse_score,
                'fusion_method': 'weighted_sum'
            })

            doc_scores[doc_id] = result

        # Sort by combined score and return top k
        combined_results = list(doc_scores.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]


# Backward compatibility alias
HybridRetriever = QdrantHybridRetriever
