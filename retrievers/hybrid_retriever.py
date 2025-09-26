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
                - top_k: Number of results to retrieve
                - score_threshold: Minimum score threshold
                - fusion: Fusion configuration with alpha parameter
        """
        super().__init__(config)

        # NO DEFAULT CONFIGURATION - Use only explicit config
        if 'embedding' not in config:
            raise ValueError(
                "Embedding configuration is required for hybrid retriever")
        if 'qdrant' not in config:
            raise ValueError(
                "Qdrant configuration is required for hybrid retriever")

        # Initialize components
        self.dense_embedding = None
        self.sparse_embedding = None
        self.vectorstore = None

        # ONLY use alpha parameter from fusion config
        fusion_config = config.get('fusion', {})
        if 'alpha' not in fusion_config:
            raise ValueError(
                "Alpha parameter is required in fusion configuration")

        # 0.0 = pure sparse, 1.0 = pure dense
        self.alpha = fusion_config['alpha']
        self.fusion_method = fusion_config.get(
            'method', 'rrf')  # Default to RRF

        # RRF constant (only used if method is RRF)
        self.rrf_k = fusion_config.get('rrf_k', 60)

        self._initialized = False

    def _initialize_components(self):
        """Initialize embeddings and vector store components using ONLY explicit config."""
        if self._initialized:
            return

        try:
            from embedding.factory import get_embedder

            # Must exist, no defaults
            embedding_section = self.config['embedding']

            # Use ONLY the provided dense and sparse configs
            if 'dense' not in embedding_section:
                raise ValueError("Dense embedding configuration is required")
            if 'sparse' not in embedding_section:
                raise ValueError("Sparse embedding configuration is required")

            # Use exact configurations provided
            dense_config = embedding_section['dense']
            sparse_config = embedding_section['sparse']

            self.dense_embedding = get_embedder(dense_config)
            self.sparse_embedding = get_embedder(sparse_config)

            # Initialize Qdrant database with exact config
            from database.qdrant_controller import QdrantVectorDB
            qdrant_db = QdrantVectorDB(config=self.config)
            self.qdrant_db = qdrant_db

            self._initialized = True
            logger.info(
                f"Hybrid retriever initialized with alpha={self.alpha}")

        except Exception as e:
            logger.error(
                f"Failed to initialize hybrid retriever components: {e}")
            raise

    @property
    def component_name(self) -> str:
        return "hybrid_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        """
        if not self._initialized:
            self._initialize_components()

        try:
            # Perform separate dense and sparse searches
            dense_results = self._perform_dense_search(query, k)
            sparse_results = self._perform_sparse_search(query, k)

            # Combine results using alpha-weighted fusion
            combined_results = self._fuse_results_with_alpha(
                dense_results, sparse_results, k)

            return combined_results

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            raise

    def _perform_dense_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Perform dense search using direct Qdrant API."""
        try:
            query_vector = self.dense_embedding.embed_query(query)
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

            results = []
            for result in search_result:
                payload = result.payload or {}
                document = Document(
                    page_content=payload.get('page_content', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        'external_id': payload.get('external_id'),
                        'qdrant_id': str(result.id),
                        # <-- Always include chunk_id
                        'chunk_id': payload.get('chunk_id')
                    }
                )

                retrieval_result = self._create_retrieval_result(
                    document=document,
                    score=result.score,
                    additional_metadata={
                        'search_type': 'dense_component',
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
            if hasattr(self.sparse_embedding, 'embed_query'):
                query_vector = self.sparse_embedding.embed_query(query)
            else:
                query_vector = self.sparse_embedding.embed_documents([query])[
                    0]

            if isinstance(query_vector, dict):
                from qdrant_client.models import NamedSparseVector
                search_result = self.qdrant_db.client.search(
                    collection_name=self.qdrant_db.collection_name,
                    query_vector=NamedSparseVector(
                        name=self.qdrant_db.sparse_vector_name,
                        vector={"indices": list(query_vector.keys()),
                                "values": list(query_vector.values())}
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

            results = []
            for result in search_result:
                payload = result.payload or {}
                document = Document(
                    page_content=payload.get('page_content', ''),
                    metadata={
                        **payload.get('metadata', {}),
                        'external_id': payload.get('external_id'),
                        'qdrant_id': str(result.id),
                        # <-- Always include chunk_id
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

            return results

        except Exception as e:
            logger.error(f"Sparse search component failed: {e}")
            return []

    def _fuse_results_with_alpha(self, dense_results: List[RetrievalResult],
                                 sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """
        Fuse results using only alpha parameter.
        alpha = 0.0: pure sparse
        alpha = 1.0: pure dense  
        alpha = 0.5: equal weighting
        """
        if self.fusion_method == 'rrf':
            return self._alpha_weighted_rrf(dense_results, sparse_results, k)
        elif self.fusion_method == 'weighted_sum':
            return self._alpha_weighted_sum(dense_results, sparse_results, k)
        else:
            raise ValueError(
                f"Unsupported fusion method: {self.fusion_method}")

    def _alpha_weighted_rrf(self, dense_results: List[RetrievalResult],
                            sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """RRF fusion with alpha weighting."""
        doc_scores = {}

        # Process dense results with alpha weighting
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.document.metadata.get('external_id')
            if doc_id:
                rrf_score = self.alpha * (1.0 / (self.rrf_k + rank))
                doc_scores[doc_id] = {
                    'result': result,
                    'dense_score': rrf_score,
                    'sparse_score': 0.0
                }

        # Process sparse results with (1-alpha) weighting
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.document.metadata.get('external_id')
            if doc_id:
                rrf_score = (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank))
                if doc_id in doc_scores:
                    doc_scores[doc_id]['sparse_score'] = rrf_score
                else:
                    doc_scores[doc_id] = {
                        'result': result,
                        'dense_score': 0.0,
                        'sparse_score': rrf_score
                    }

        # Combine and sort
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = scores['dense_score'] + scores['sparse_score']
            result = scores['result']
            result.score = combined_score
            result.metadata.update({
                'search_type': 'hybrid_alpha_rrf',
                'alpha': self.alpha,
                'dense_rrf_score': scores['dense_score'],
                'sparse_rrf_score': scores['sparse_score'],
                'fusion_method': 'rrf'
            })
            combined_results.append(result)

        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]

    def _alpha_weighted_sum(self, dense_results: List[RetrievalResult],
                            sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Weighted sum fusion with alpha parameter."""
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return {}
            scores = [r.score for r in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0

            normalized = {}
            for result in results:
                doc_id = result.document.metadata.get('external_id')
                if doc_id:
                    normalized[doc_id] = {
                        'result': result,
                        'score': (result.score - min_score) / score_range
                    }
            return normalized

        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)

        # Combine with alpha weighting
        doc_scores = {}
        all_doc_ids = set(dense_normalized.keys()) | set(
            sparse_normalized.keys())

        for doc_id in all_doc_ids:
            dense_score = dense_normalized.get(doc_id, {}).get('score', 0.0)
            sparse_score = sparse_normalized.get(doc_id, {}).get('score', 0.0)

            # Alpha weighting: alpha * dense + (1-alpha) * sparse
            combined_score = self.alpha * dense_score + \
                (1.0 - self.alpha) * sparse_score

            result = (dense_normalized.get(doc_id)
                      or sparse_normalized.get(doc_id))['result']
            result.score = combined_score
            result.metadata.update({
                'search_type': 'hybrid_alpha_weighted',
                'alpha': self.alpha,
                'dense_norm_score': dense_score,
                'sparse_norm_score': sparse_score,
                'fusion_method': 'weighted_sum'
            })
            doc_scores[doc_id] = result

        combined_results = list(doc_scores.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:k]


# Backward compatibility alias
HybridRetriever = QdrantHybridRetriever
