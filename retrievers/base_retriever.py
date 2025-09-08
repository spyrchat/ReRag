"""
Modern base retriever that integrates with the retrieval pipeline architecture.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from components.retrieval_pipeline import BaseRetriever, RetrievalResult
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class ModernBaseRetriever(BaseRetriever):
    """
    Modern base retriever implementing the retrieval pipeline interface.
    All modern retrievers should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retriever with configuration.

        Args:
            config: Configuration dictionary containing retrieval parameters
        """
        self.config = config
        self.top_k = config.get("top_k", 5)
        self.score_threshold = config.get("score_threshold", 0.0)
        self._initialized = False  # Track initialization state

        # Initialize any common components here
        self._initialize_components()

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

        logger.debug(f"Retrieving {k} results for query: {query[:50]}...")

        try:
            # Perform the search
            results = self._perform_search(query, k)

            # Apply score threshold filtering
            if self.score_threshold > 0:
                results = [r for r in results if r.score >=
                           self.score_threshold]
                logger.debug(
                    f"Filtered to {len(results)} results above threshold {self.score_threshold}")

            # Ensure we don't return more than requested
            results = results[:k]

            logger.debug(f"Retrieved {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def _create_retrieval_result(
        self,
        document: Document,
        score: float,
        additional_metadata: Dict[str, Any] = None
    ) -> RetrievalResult:
        """
        Create a RetrievalResult object with proper metadata.

        Args:
            document: The retrieved document
            score: Relevance score
            additional_metadata: Additional metadata to include

        Returns:
            RetrievalResult object
        """
        metadata = {
            "retriever_config": self.config,
            "retrieval_timestamp": None,  # Can add timestamp if needed
        }

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
