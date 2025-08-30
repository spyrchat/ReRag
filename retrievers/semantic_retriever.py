"""
Modern semantic retriever that integrates with the retrieval pipeline architecture.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from components.retrieval_pipeline import RetrievalResult
from .base_retriever import ModernBaseRetriever
import logging

logger = logging.getLogger(__name__)


class SemanticRetriever(ModernBaseRetriever):
    """
    Advanced semantic retriever that can use multiple retrieval strategies
    and combine them intelligently based on query analysis.

    This retriever can:
    - Analyze query intent and complexity
    - Route to appropriate retrieval strategies
    - Combine multiple retrieval methods
    - Apply semantic post-processing
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the semantic retriever.

        Args:
            config: Configuration dictionary containing:
                - strategies: List of retrieval strategies to use
                - query_analyzer: Configuration for query analysis
                - routing_rules: Rules for strategy selection
                - top_k: Number of results to retrieve (default: 5)
                - score_threshold: Minimum score threshold (default: 0.0)
        """
        super().__init__(config)

        # Initialize retrieval strategies
        self.strategies = {}
        self.strategy_weights = {}
        self.query_analyzer = None

        # Configuration
        self.routing_rules = config.get('routing_rules', {})
        self.default_strategy = config.get('default_strategy', 'hybrid')

    def _initialize_components(self):
        """Initialize retrieval strategies and query analyzer."""
        try:
            strategies_config = self.config.get('strategies', {})

            # Initialize available strategies
            for strategy_name, strategy_config in strategies_config.items():
                if strategy_config.get('enabled', True):
                    strategy = self._create_strategy(
                        strategy_name, strategy_config)
                    if strategy:
                        self.strategies[strategy_name] = strategy
                        self.strategy_weights[strategy_name] = strategy_config.get(
                            'weight', 1.0)

            # Initialize query analyzer if configured
            analyzer_config = self.config.get('query_analyzer', {})
            if analyzer_config.get('enabled', False):
                self.query_analyzer = self._create_query_analyzer(
                    analyzer_config)

            logger.info(
                f"Semantic retriever initialized with strategies: {list(self.strategies.keys())}")

        except Exception as e:
            logger.error(
                f"Failed to initialize semantic retriever components: {e}")
            # Initialize empty strategies to prevent further errors
            if not hasattr(self, 'strategies'):
                self.strategies = {}
            if not hasattr(self, 'strategy_weights'):
                self.strategy_weights = {}

    def _create_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]) -> Optional[ModernBaseRetriever]:
        """Create a retrieval strategy instance."""
        try:
            # Merge global config with strategy-specific config
            merged_config = self.config.copy()
            merged_config.update(strategy_config)

            if strategy_name == 'dense':
                from .dense_retriever import QdrantDenseRetriever
                return QdrantDenseRetriever(merged_config)
            elif strategy_name == 'sparse':
                from .sparse_retriever import QdrantSparseRetriever
                return QdrantSparseRetriever(merged_config)
            elif strategy_name == 'hybrid':
                from .hybrid_retriever import QdrantHybridRetriever
                return QdrantHybridRetriever(merged_config)
            else:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return None

        except Exception as e:
            logger.warning(f"Failed to create strategy {strategy_name}: {e}")
            return None

    def _create_query_analyzer(self, analyzer_config: Dict[str, Any]):
        """Create query analyzer for intelligent routing."""
        # Placeholder for advanced query analysis
        # Could integrate with LLMs, query classification models, etc.
        return None

    @property
    def component_name(self) -> str:
        return "semantic_retriever"

    def _perform_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform semantic search using intelligent strategy selection.

        Args:
            query: Search query
            k: Number of results to retrieve

        Returns:
            List of RetrievalResult objects
        """
        if not self.strategies:
            self._initialize_components()

        try:
            # Analyze query to determine best strategies
            selected_strategies = self._select_strategies(query)

            if not selected_strategies:
                logger.warning("No strategies selected, using default")
                selected_strategies = [self.default_strategy] if self.default_strategy in self.strategies else list(
                    self.strategies.keys())[:1]

            # If still no strategies, return empty
            if not selected_strategies:
                logger.warning("No strategies available")
                return []

            # Perform retrieval with selected strategies
            if len(selected_strategies) == 1:
                # Single strategy
                strategy_name = selected_strategies[0]
                if strategy_name not in self.strategies:
                    logger.warning(f"Strategy {strategy_name} not available")
                    return []

                strategy = self.strategies[strategy_name]
                results = strategy._perform_search(query, k)

                # Update retrieval method in metadata
                for result in results:
                    result.retrieval_method = f"semantic_{strategy_name}"

                return results
            else:
                # Multiple strategies - combine results
                return self._combine_strategy_results(query, selected_strategies, k)

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    def _select_strategies(self, query: str) -> List[str]:
        """
        Select appropriate retrieval strategies based on query analysis.

        Args:
            query: Search query

        Returns:
            List of strategy names to use
        """
        # Simple rule-based selection for now
        # Could be enhanced with ML-based query classification

        query_lower = query.lower()
        query_length = len(query.split())

        # Default to hybrid for most queries
        selected = []

        # Rule-based strategy selection
        if query_length <= 3:
            # Short queries - prefer sparse (keyword matching)
            if 'sparse' in self.strategies:
                selected.append('sparse')
        elif any(keyword in query_lower for keyword in ['how to', 'what is', 'explain', 'describe']):
            # Conceptual queries - prefer dense (semantic similarity)
            if 'dense' in self.strategies:
                selected.append('dense')
        else:
            # Default to hybrid for balanced approach
            if 'hybrid' in self.strategies:
                selected.append('hybrid')
            elif 'dense' in self.strategies and 'sparse' in self.strategies:
                # Fallback to combining dense and sparse
                selected.extend(['dense', 'sparse'])

        # Apply routing rules from config
        for rule_query, rule_strategies in self.routing_rules.items():
            if rule_query.lower() in query_lower:
                selected = rule_strategies
                break

        # Ensure selected strategies exist
        selected = [s for s in selected if s in self.strategies]

        logger.debug(
            f"Selected strategies for query '{query[:50]}...': {selected}")
        return selected

    def _combine_strategy_results(self, query: str, strategies: List[str], k: int) -> List[RetrievalResult]:
        """
        Combine results from multiple strategies using fusion techniques.

        Args:
            query: Search query
            strategies: List of strategy names
            k: Number of final results

        Returns:
            Combined and ranked results
        """
        all_results = {}  # document_id -> RetrievalResult
        strategy_results = {}

        # Collect results from each strategy
        for strategy_name in strategies:
            if strategy_name not in self.strategies:
                continue

            strategy = self.strategies[strategy_name]
            results = strategy._perform_search(
                query, k * 2)  # Get more results for fusion
            strategy_results[strategy_name] = results

            for result in results:
                doc_id = self._get_document_id(result.document)
                if doc_id not in all_results:
                    all_results[doc_id] = result
                    # Update metadata to reflect semantic fusion
                    result.retrieval_method = f"semantic_fusion"
                    if 'fusion_strategies' not in result.metadata:
                        result.metadata['fusion_strategies'] = []
                    result.metadata['fusion_strategies'].append(strategy_name)
                else:
                    # Combine scores using weighted average
                    existing = all_results[doc_id]
                    weight1 = self.strategy_weights.get(
                        existing.metadata['fusion_strategies'][-1], 1.0)
                    weight2 = self.strategy_weights.get(strategy_name, 1.0)

                    combined_score = (
                        existing.score * weight1 + result.score * weight2) / (weight1 + weight2)
                    existing.score = combined_score
                    existing.metadata['fusion_strategies'].append(
                        strategy_name)

        # Rank and return top k results
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:k]

    def _get_document_id(self, document: Document) -> str:
        """Get a unique identifier for a document."""
        # Use external_id if available, otherwise use content hash
        if hasattr(document, 'metadata') and 'external_id' in document.metadata:
            return document.metadata['external_id']
        else:
            import hashlib
            return hashlib.md5(document.page_content.encode()).hexdigest()
