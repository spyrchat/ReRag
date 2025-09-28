"""
Modern base retriever that integrates with the retrieval pipeline architecture.
"""

from typing import Any, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from components.retrieval_pipeline import BaseRetriever, RetrievalResult
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    document: Any
    score: float
    retrieval_method: str
    metadata: Dict[str, Any]


class ModernBaseRetriever(BaseRetriever):
    """
    Modern base retriever implementing the retrieval pipeline interface.
    All modern retrievers should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with ONLY the provided configuration - no defaults merged."""
        # Store the exact configuration provided - NO MERGING WITH DEFAULTS
        self.config = config.copy()  # Use exact copy of provided config

        # Only extract essential parameters that must exist
        self.top_k = config.get('top_k')
        self.score_threshold = config.get('score_threshold', 0.0)

        if self.top_k is None:
            raise ValueError(
                "top_k parameter is required in retriever configuration")

        # Performance settings - only if explicitly provided
        self.performance_config = config.get('performance', {})
        self.enable_caching = self.performance_config.get(
            'enable_caching', False)
        self.batch_size = self.performance_config.get('batch_size', 1)

        self._initialized = False  # Track initialization state

    def _initialize_components(self):
        """Initialize common components. Override in subclasses."""
        # Enable lazy initialization by default
        self._lazy_init = self.config.get('lazy_initialization', True)

        # If lazy initialization is disabled, subclasses should override this
        if not self._lazy_init:
            # Actual initialization happens in subclasses
            pass

    @abstractmethod
    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform the actual search operation.

        Args:
            query: Search query
            k: Number of results to retrieve

        Returns:
            List of RetrievalResult objects
        """
        pass

    def retrieve(self, query: str, k: int = None) -> List[RetrievalResult]:
        """
        Retrieve documents for the given query.

        Args:
            query: Search query
            k: Number of results to retrieve (defaults to configured top_k)

        Returns:
            List of RetrievalResult objects
        """
        if k is None:
            k = self.top_k

        try:
            # Perform the search
            results = self._perform_search(query, k)

            # Apply score threshold filtering
            if self.score_threshold > 0:
                results = [r for r in results if r.score >=
                           self.score_threshold]

            # Ensure we don't return more than requested
            results = results[:k]

        except Exception as e:
            return []

    def _create_retrieval_result(
        self,
        document: Document,
        score: float,
        additional_metadata: Dict[str, Any] = None
    ) -> RetrievalResult:
        """
        Construct a RetrievalResult dataclass instance for a retrieved document.

        This method extracts the chunk_id from the document's metadata (if present)
        and adds it to the result metadata. It also merges any additional metadata provided.

        Args:
            document (Document): The retrieved document object, typically with metadata.
            score (float): The relevance score for the retrieved document.
            additional_metadata (Dict[str, Any], optional): Any extra metadata to include in the result.

        Returns:
            RetrievalResult: A dataclass instance containing the document, score, retrieval method, and metadata.
        """
        metadata = {
            "retriever_config": self.config,
            "retrieval_timestamp": None,  # Can add timestamp if needed
        }

        # Extract chunk_id from document metadata if present
        chunk_id = None
        if hasattr(document, "metadata") and document.metadata:
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id:
                metadata["chunk_id"] = chunk_id

        if additional_metadata:
            metadata.update(additional_metadata)

        return RetrievalResult(
            document=document,
            score=score,
            retrieval_method=self.component_name,
            metadata=metadata
        )

    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Normalize scores to [0, 1] range.

        Args:
            results: List of retrieval results

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            # Normalize to [0, 1]
            for result in results:
                result.score = (result.score - min_score) / \
                    (max_score - min_score)

        return results

    def _validate_config(self, required_keys: List[str]):
        """
        Validate that required configuration keys are present.

        Args:
            required_keys: List of required configuration keys

        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {missing_keys}")
