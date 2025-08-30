"""
Modular and extensible retrieval pipeline for RAG systems.
Supports easy addition of components like rerankers, filters, and post-processors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Enhanced result structure for retrieval pipeline.

    Attributes:
        document (Document): The retrieved document
        score (float): Relevance score
        retrieval_method (str): Method used for retrieval
        metadata (Dict[str, Any]): Additional metadata
    """
    document: Document
    score: float
    retrieval_method: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetrievalComponent(ABC):
    """
    Base class for all retrieval pipeline components.
    All pipeline components (retrievers, rerankers, filters) inherit from this.
    """

    @property
    @abstractmethod
    def component_name(self) -> str:
        """
        Return the name of this component.

        Returns:
            str: Component name for identification and logging
        """
        pass

    @abstractmethod
    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """
        Process the query and/or results.

        Args:
            query (str): The search query
            results (List[RetrievalResult]): Current results to process
            **kwargs: Additional parameters

        Returns:
            List[RetrievalResult]: Processed results
        """
        pass


class BaseRetriever(RetrievalComponent):
    """
    Base retriever that generates initial results.
    All specific retrievers (dense, sparse, hybrid) inherit from this.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve initial results.

        Args:
            query (str): Search query
            k (int): Number of results to retrieve

        Returns:
            List[RetrievalResult]: Initial retrieval results
        """
        pass

    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """
        For retrievers, generate new results (ignore input results).

        Args:
            query (str): Search query
            results (List[RetrievalResult]): Ignored for initial retrieval
            **kwargs: Additional parameters including 'k' for result count

        Returns:
            List[RetrievalResult]: Fresh retrieval results
        """
        k = kwargs.get('k', 5)
        return self.retrieve(query, k)


class Reranker(RetrievalComponent):
    """
    Base class for reranking components.
    Rerankers take existing results and reorder them based on improved relevance scoring.
    """

    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """
        Rerank the results.

        Args:
            query (str): The search query
            results (List[RetrievalResult]): Results to rerank
            **kwargs: Additional reranking parameters

        Returns:
            List[RetrievalResult]: Reranked results
        """
        pass

    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Process by reranking."""
        return self.rerank(query, results, **kwargs)


class ResultFilter(RetrievalComponent):
    """Base class for filtering components."""

    @abstractmethod
    def filter(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Filter the results."""
        pass

    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Process by filtering."""
        return self.filter(query, results, **kwargs)


class PostProcessor(RetrievalComponent):
    """Base class for post-processing components."""

    @abstractmethod
    def post_process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Post-process the results."""
        pass

    def process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Process by post-processing."""
        return self.post_process(query, results, **kwargs)


class RetrievalPipeline:
    """
    Modular retrieval pipeline that chains components together.

    Example usage:
        pipeline = RetrievalPipeline([
            QdrantHybridRetriever(config),
            CrossEncoderReranker(model="ms-marco-MiniLM-L-12-v2"),
            MetadataFilter(min_score=0.5),
            AnswerContextEnhancer()
        ])

        results = pipeline.run(query="How to count bits?", k=10)
    """

    def __init__(self, components: List[RetrievalComponent], config: Dict[str, Any] = None):
        self.components = components
        self.config = config or {}
        self._validate_pipeline()

        logger.info(f"Initialized retrieval pipeline with {len(components)} components: "
                    f"{[comp.component_name for comp in components]}")

    def _validate_pipeline(self):
        """Validate that the pipeline has at least one retriever."""
        has_retriever = any(isinstance(comp, BaseRetriever)
                            for comp in self.components)
        if not has_retriever:
            raise ValueError(
                "Pipeline must contain at least one BaseRetriever component")

    def run(self, query: str, **kwargs) -> List[RetrievalResult]:
        """
        Run the full retrieval pipeline.

        Args:
            query: The search query
            **kwargs: Additional parameters passed to components

        Returns:
            List of RetrievalResult objects
        """
        logger.info(f"Running retrieval pipeline for query: '{query[:50]}...'")

        results = []

        for i, component in enumerate(self.components):
            component_name = component.component_name
            logger.debug(f"Step {i+1}: Running {component_name}")

            try:
                # Merge component-specific config with runtime kwargs
                component_kwargs = kwargs.copy()
                component_config = self.config.get(component_name, {})
                component_kwargs.update(component_config)

                # Process with component
                results = component.process(query, results, **component_kwargs)

                logger.debug(
                    f"{component_name} returned {len(results)} results")

            except Exception as e:
                logger.error(f"Error in {component_name}: {e}")
                # Decide whether to continue or fail
                if self.config.get('fail_on_component_error', False):
                    raise
                # Continue with previous results

        logger.info(f"Pipeline completed with {len(results)} final results")
        return results

    def add_component(self, component: RetrievalComponent, position: int = -1):
        """Add a component to the pipeline at the specified position."""
        if position == -1:
            self.components.append(component)
        else:
            self.components.insert(position, component)

        self._validate_pipeline()
        logger.info(
            f"Added {component.component_name} to pipeline at position {position}")

    def remove_component(self, component_name: str) -> bool:
        """Remove a component by name."""
        for i, comp in enumerate(self.components):
            if comp.component_name == component_name:
                removed = self.components.pop(i)
                logger.info(f"Removed {removed.component_name} from pipeline")
                self._validate_pipeline()
                return True
        return False

    def get_component(self, component_name: str) -> Optional[RetrievalComponent]:
        """Get a component by name."""
        for comp in self.components:
            if comp.component_name == component_name:
                return comp
        return None

    def to_langchain_retriever(self):
        """Create a LangChain-compatible retriever interface."""
        class LangChainWrapper:
            def __init__(self, pipeline: RetrievalPipeline):
                self.pipeline = pipeline

            def get_relevant_documents(self, query: str) -> List[Document]:
                results = self.pipeline.run(query)
                return [r.document for r in results]

            def retrieve(self, query: str, k: int = 5) -> List[Document]:
                results = self.pipeline.run(query, k=k)
                return [r.document for r in results]

        return LangChainWrapper(self)


class RetrievalPipelineFactory:
    """Factory for creating common retrieval pipeline configurations."""

    @staticmethod
    def create_dense_pipeline(config: Dict[str, Any]) -> RetrievalPipeline:
        """Create a dense-only retrieval pipeline."""
        from retrievers.dense_retriever import QdrantDenseRetriever

        # Create modern dense retriever
        retriever = QdrantDenseRetriever(config)

        return RetrievalPipeline([retriever], config)

    @staticmethod
    def create_hybrid_pipeline(config: Dict[str, Any]) -> RetrievalPipeline:
        """Create a hybrid retrieval pipeline."""
        from retrievers.hybrid_retriever import QdrantHybridRetriever

        # Create modern hybrid retriever
        retriever = QdrantHybridRetriever(config)

        return RetrievalPipeline([retriever], config)

    @staticmethod
    def create_reranked_pipeline(config: Dict[str, Any], reranker_model: str = None) -> RetrievalPipeline:
        """Create a pipeline with retrieval + reranking."""
        # Start with hybrid if available, otherwise dense
        if config.get("embedding", {}).get("sparse"):
            pipeline = RetrievalPipelineFactory.create_hybrid_pipeline(config)
        else:
            pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)

        # Add reranker if specified
        if reranker_model:
            from components.rerankers import CrossEncoderReranker
            reranker = CrossEncoderReranker(model_name=reranker_model)
            pipeline.add_component(reranker)

        return pipeline

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> 'RetrievalPipeline':
        """
        Create a retrieval pipeline from configuration.

        Args:
            config: Configuration dictionary with 'retrieval_pipeline' section

        Returns:
            Configured RetrievalPipeline

        Example config:
            retrieval_pipeline:
              retriever:
                type: dense  # or hybrid
                top_k: 10
              stages:
                - type: score_filter
                  config:
                    min_score: 0.3
                - type: reranker
                  config:
                    model_type: cross_encoder
                    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    top_k: 5
        """
        pipeline_config = config.get("retrieval_pipeline", {})

        if not pipeline_config:
            raise ValueError("No 'retrieval_pipeline' section found in config")

        # Create retriever
        retriever_config = pipeline_config.get("retriever", {})
        retriever = RetrievalPipelineFactory._create_retriever(
            retriever_config, config)

        # Initialize pipeline with retriever
        pipeline = RetrievalPipeline([retriever], config)

        # Add stages
        stages = pipeline_config.get("stages", [])
        for stage_config in stages:
            component = RetrievalPipelineFactory._create_stage_component(
                stage_config, config)
            if component:
                pipeline.add_component(component)

        logger.info(
            f"Created pipeline from config with {len(pipeline.components)} components")
        return pipeline

    @staticmethod
    def _create_retriever(retriever_config: Dict[str, Any], global_config: Dict[str, Any]) -> BaseRetriever:
        """Create retriever from configuration."""
        retriever_type = retriever_config.get("type", "dense")

        if retriever_type == "dense":
            from retrievers.dense_retriever import QdrantDenseRetriever
            return QdrantDenseRetriever(global_config)
        elif retriever_type == "sparse":
            from retrievers.sparse_retriever import QdrantSparseRetriever
            return QdrantSparseRetriever(global_config)
        elif retriever_type == "hybrid":
            from retrievers.hybrid_retriever import QdrantHybridRetriever
            return QdrantHybridRetriever(global_config)
        elif retriever_type == "semantic":
            from retrievers.semantic_retriever import SemanticRetriever
            return SemanticRetriever(global_config)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    @staticmethod
    def _create_stage_component(stage_config: Dict[str, Any], global_config: Dict[str, Any]) -> Optional[RetrievalComponent]:
        """Create a pipeline stage component from configuration."""
        stage_type = stage_config.get("type")
        config = stage_config.get("config", {})

        try:
            if stage_type == "score_filter":
                from components.filters import ScoreFilter
                return ScoreFilter(min_score=config.get("min_score", 0.3))

            elif stage_type == "duplicate_filter":
                from components.filters import DuplicateFilter
                return DuplicateFilter(dedup_by=config.get("dedup_by", "external_id"))

            elif stage_type == "tag_filter":
                from components.filters import TagFilter
                return TagFilter(
                    required_tags=config.get("required_tags"),
                    excluded_tags=config.get("excluded_tags")
                )

            elif stage_type == "answer_enhancer":
                from components.filters import AnswerEnhancer
                return AnswerEnhancer()

            elif stage_type == "result_limiter":
                from components.filters import ResultLimiter
                return ResultLimiter(max_results=config.get("max_results", 5))

            elif stage_type == "reranker":
                return RetrievalPipelineFactory._create_reranker(config)

            else:
                logger.warning(f"Unknown stage type: {stage_type}")
                return None

        except ImportError as e:
            logger.warning(f"Could not create {stage_type}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating {stage_type}: {e}")
            return None

    @staticmethod
    def _create_reranker(config: Dict[str, Any]) -> Optional[Reranker]:
        """Create reranker from configuration."""
        model_type = config.get("model_type")

        try:
            if model_type == "cross_encoder":
                from components.rerankers import CrossEncoderReranker
                return CrossEncoderReranker(
                    model_name=config.get(
                        "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                    top_k=config.get("top_k")
                )

            elif model_type == "bge":
                from components.advanced_rerankers import BgeReranker
                return BgeReranker(
                    model_name=config.get(
                        "model_name", "BAAI/bge-reranker-base"),
                    top_k=config.get("top_k")
                )

            elif model_type == "multistage":
                stage1_config = config.get("stage1", {})
                stage2_config = config.get("stage2", {})

                stage1_reranker = RetrievalPipelineFactory._create_reranker(
                    stage1_config)
                stage2_reranker = RetrievalPipelineFactory._create_reranker(
                    stage2_config)

                if stage1_reranker and stage2_reranker:
                    from components.advanced_rerankers import MultiStageReranker
                    return MultiStageReranker(
                        stage1_reranker=stage1_reranker,
                        stage2_reranker=stage2_reranker,
                        stage1_k=stage1_config.get("top_k", 10),
                        stage2_k=stage2_config.get("top_k", 5)
                    )

            elif model_type == "ensemble":
                rerankers = []
                weights = []

                for reranker_config in config.get("rerankers", []):
                    reranker = RetrievalPipelineFactory._create_reranker(
                        reranker_config)
                    if reranker:
                        rerankers.append(reranker)
                        weights.append(reranker_config.get("weight", 1.0))

                if rerankers:
                    from components.rerankers import EnsembleReranker
                    return EnsembleReranker(
                        rerankers=rerankers,
                        weights=weights
                    )

            else:
                logger.warning(f"Unknown reranker type: {model_type}")
                return None

        except ImportError as e:
            logger.warning(f"Could not create {model_type} reranker: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating {model_type} reranker: {e}")
            return None

    @staticmethod
    def create_sparse_pipeline(config: Dict[str, Any]) -> RetrievalPipeline:
        """Create a sparse-only retrieval pipeline."""
        from retrievers.sparse_retriever import QdrantSparseRetriever

        # Create modern sparse retriever
        retriever = QdrantSparseRetriever(config)

        return RetrievalPipeline([retriever], config)

    @staticmethod
    def create_semantic_pipeline(config: Dict[str, Any]) -> RetrievalPipeline:
        """Create a semantic retrieval pipeline with intelligent routing."""
        from retrievers.semantic_retriever import SemanticRetriever

        # Create semantic retriever
        retriever = SemanticRetriever(config)

        return RetrievalPipeline([retriever], config)

    @staticmethod
    def create_from_retriever_config(retriever_type: str, global_config: Dict[str, Any] = None) -> 'RetrievalPipeline':
        """
        Create a retrieval pipeline from a retriever configuration file.

        Args:
            retriever_type: Type of retriever (dense, sparse, hybrid, semantic)
            global_config: Optional global configuration to merge with

        Returns:
            Configured RetrievalPipeline
        """
        try:
            from pipelines.configs.retriever_config_loader import load_retriever_config

            # Load retriever-specific configuration
            retriever_config = load_retriever_config(retriever_type)

            # Merge with global config if provided
            if global_config:
                from pipelines.configs.retriever_config_loader import RetrieverConfigLoader
                loader = RetrieverConfigLoader()
                merged_config = loader.merge_with_global_config(
                    retriever_config, global_config)
            else:
                merged_config = retriever_config

            # Create retriever from the merged config
            retriever = RetrievalPipelineFactory._create_retriever(
                merged_config['retriever'], merged_config
            )

            # Create pipeline
            pipeline = RetrievalPipeline([retriever], merged_config)

            logger.info(
                f"Created {retriever_type} pipeline from configuration file")
            return pipeline

        except Exception as e:
            logger.error(
                f"Failed to create pipeline from {retriever_type} config: {e}")
            raise

    @staticmethod
    def list_available_retrievers() -> List[str]:
        """
        List all available retriever types from configuration files.

        Returns:
            List of available retriever types
        """
        try:
            from pipelines.configs.retriever_config_loader import RetrieverConfigLoader
            loader = RetrieverConfigLoader()
            return loader.get_available_configs()
        except Exception as e:
            logger.warning(f"Could not load retriever configs: {e}")
            return []

    @staticmethod
    def create_from_unified_config(config: Dict[str, Any], retriever_type: str = None) -> 'RetrievalPipeline':
        """
        Create a retrieval pipeline from unified configuration structure.

        Args:
            config: Complete configuration dictionary with retriever configs embedded
            retriever_type: Type of retriever to use (if not specified, uses pipeline default)

        Returns:
            Configured RetrievalPipeline

        Example usage:
            config = load_config("config.yml")
            pipeline = RetrievalPipelineFactory.create_from_unified_config(config, "hybrid")
        """
        from config.config_loader import get_retriever_config, get_pipeline_config

        # Get pipeline configuration
        pipeline_config = get_pipeline_config(config)

        # Determine retriever type
        if retriever_type is None:
            retriever_type = pipeline_config.get("default_retriever", "hybrid")

        # Get retriever-specific configuration
        retriever_config = get_retriever_config(config, retriever_type)

        # Create retriever using unified config
        retriever = RetrievalPipelineFactory._create_retriever_from_unified_config(
            retriever_config, config)

        # Initialize pipeline with retriever
        pipeline = RetrievalPipeline([retriever], config)

        # Add components from pipeline config
        components = pipeline_config.get("components", [])
        for component_config in components:
            if component_config.get("type") == "retriever":
                # Skip retriever component as it's already added
                continue

            component = RetrievalPipelineFactory._create_stage_component(
                component_config, config)
            if component:
                pipeline.add_component(component)

        logger.info(
            f"Created {retriever_type} pipeline from unified config with {len(pipeline.components)} components")
        return pipeline

    @staticmethod
    def _create_retriever_from_unified_config(retriever_config: Dict[str, Any],
                                              global_config: Dict[str, Any]) -> BaseRetriever:
        """Create retriever from unified configuration structure."""
        retriever_type = retriever_config.get("type")

        if retriever_type == "dense":
            from retrievers.dense_retriever import QdrantDenseRetriever
            return QdrantDenseRetriever(retriever_config)
        elif retriever_type == "sparse":
            from retrievers.sparse_retriever import QdrantSparseRetriever
            return QdrantSparseRetriever(retriever_config)
        elif retriever_type == "hybrid":
            from retrievers.hybrid_retriever import QdrantHybridRetriever
            return QdrantHybridRetriever(retriever_config)
        elif retriever_type == "semantic":
            from retrievers.semantic_retriever import SemanticRetriever
            return SemanticRetriever(retriever_config)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
