"""
Modern hybrid retriever that integrates with the retrieval pipeline architecture.
"""

from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from components.retrieval_pipeline import RetrievalResult
from .base_retriever import ModernBaseRetriever
from .sparse_retriever import QdrantSparseRetriever
from .dense_retriever import QdrantDenseRetriever
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
        print('rrf_K: ', self.rrf_k)

        self._initialized = False

    def _initialize_components(self):
        """Initialize embeddings and vector store components using ONLY explicit config."""
        if self._initialized:
            return

        try:
            from embedding.factory import get_embedder

            embedding_section = self.config['embedding']
            if 'dense' not in embedding_section:
                raise ValueError("Dense embedding configuration is required")
            if 'sparse' not in embedding_section:
                raise ValueError("Sparse embedding configuration is required")

            dense_config = embedding_section['dense']
            sparse_config = embedding_section['sparse']

            # Repackage configs for sub-retrievers to match standalone usage
            dense_retriever_config = {**self.config, 'embedding': dense_config}
            sparse_retriever_config = {
                **self.config, 'embedding': sparse_config}

            self.dense_retriever = QdrantDenseRetriever(dense_retriever_config)
            self.sparse_retriever = QdrantSparseRetriever(
                sparse_retriever_config)

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
            if dense_results is None:
                logger.error(
                    "Dense retriever returned None! Replacing with [].")
                dense_results = []
            logger.debug(f"Dense results: {len(dense_results)}")

            sparse_results = self._perform_sparse_search(query, k)
            if sparse_results is None:
                logger.error(
                    "Sparse retriever returned None! Replacing with [].")
                sparse_results = []
            logger.debug(f"Sparse results: {len(sparse_results)}")

            # Combine results using alpha-weighted fusion
            combined_results = self._fuse_results_with_alpha(
                dense_results, sparse_results, k)

            return combined_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _perform_dense_search(self, query: str, k: int) -> List[RetrievalResult]:
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

        return self.dense_retriever._perform_search(query, k)

    def _perform_sparse_search(self, query: str, k: int) -> List[RetrievalResult]:
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
        results = self.sparse_retriever._perform_search(query, k)
        return results

    def _fuse_results_with_alpha(self, dense_results: List[RetrievalResult], sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Enhanced fusion with pure modes and fixed RRF."""
        if self.alpha >= 0.99:
            logger.info("Using pure dense mode (alpha >= 0.99)")
            return dense_results[:k]
        elif self.alpha <= 0.01:
            logger.info("Using pure sparse mode (alpha <= 0.01)")
            return sparse_results[:k]
        if self.fusion_method == 'rrf':
            return self._rrf_fusion(dense_results, sparse_results, k)
        elif self.fusion_method == 'weighted_sum':
            return self._alpha_weighted_sum(dense_results, sparse_results, k)
        else:
            raise ValueError(
                f"Unsupported fusion method: {self.fusion_method}")

    def _rrf_fusion(self, dense_results: List[RetrievalResult], sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Standard (optionally weighted) RRF fusion at chunk level, robust to missing keys and stable in ordering."""
        # Use chunk_id as the stable key for all fusion
        def get_key(r):
            return r.document.metadata.get('chunk_id')

        # Filter out results with missing chunk_id (None)
        dense_results = [r for r in dense_results if get_key(r) is not None]
        sparse_results = [r for r in sparse_results if get_key(r) is not None]

        # Build rank mappings (1-based); missing = None
        dense_ranks = {get_key(r): rank for rank,
                       r in enumerate(dense_results, 1)}
        sparse_ranks = {get_key(r): rank for rank,
                        r in enumerate(sparse_results, 1)}
        all_keys = set(dense_ranks) | set(sparse_ranks)

        # Build a single representative map: prefer dense, then sparse
        representative = {}
        for r in dense_results:
            key = get_key(r)
            if key not in representative:
                representative[key] = r
        for r in sparse_results:
            key = get_key(r)
            if key not in representative:
                representative[key] = r

        # Compute RRF scores and min-rank for tie-breaking
        scored = []
        for key in all_keys:
            dense_rank = dense_ranks.get(key)
            sparse_rank = sparse_ranks.get(key)
            dense_rrf = 1.0 / \
                (self.rrf_k + dense_rank) if dense_rank is not None else 0.0
            sparse_rrf = 1.0 / \
                (self.rrf_k + sparse_rank) if sparse_rank is not None else 0.0
            final_score = self.alpha * dense_rrf + \
                (1.0 - self.alpha) * sparse_rrf
            min_rank = min([r for r in [dense_rank, sparse_rank] if r is not None]) if (
                dense_rank is not None or sparse_rank is not None) else float('inf')
            result = representative[key]
            result.score = final_score
            result.metadata.update({
                'search_type': 'hybrid_rrf',
                'alpha': self.alpha,
                'dense_rrf': dense_rrf,
                'sparse_rrf': sparse_rrf,
                'fusion_method': 'rrf',
                'min_rank': min_rank
            })
            # -min_rank: lower is better
            scored.append((final_score, -min_rank, key, result))

        # Sort by score desc, then by better (lower) min-rank, then by key for stability
        scored.sort(reverse=True)
        results = [t[-1] for t in scored[:k]]
        return results

    def _alpha_weighted_sum(self, dense_results: List[RetrievalResult],
                            sparse_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Weighted sum fusion with alpha parameter."""
        # Normalize scores

        def normalize_scores(results: List[RetrievalResult]) -> Dict:
            """Robust score normalization preserving relative rankings."""
            if not results:
                return {}

            scores = [r.score for r in results]

            # Use min-max normalization with small epsilon to prevent division by zero
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range < 1e-10:  # Prevent division by zero
                # If all scores are the same, use rank-based normalization
                normalized = {}
                for rank, result in enumerate(results, 1):
                    doc_id = result.document.metadata.get('chunk_id')
                    if doc_id:
                        normalized[doc_id] = {
                            'result': result,
                            'score': 1.0 / rank  # Rank-based fallback
                        }
                return normalized

            normalized = {}
            for result in results:
                doc_id = result.document.metadata.get('chunk_id')
                if doc_id:
                    norm_score = (result.score - min_score) / score_range
                    normalized[doc_id] = {
                        'result': result,
                        'score': norm_score
                    }

            return normalized

        dense_normalized = dense_results
        sparse_normalized = sparse_results

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
