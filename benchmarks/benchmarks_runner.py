"""Configuration-driven benchmark execution engine."""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from benchmarks.benchmark_contracts import BenchmarkAdapter, BenchmarkQuery, BenchmarkResult
from benchmarks.benchmarks_metrics import BenchmarkMetrics
from components.retrieval_pipeline import RetrievalPipelineFactory
from config.config_loader import get_benchmark_config, get_retriever_config


class BenchmarkRunner:
    """Execute benchmarks against configurable RAG systems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use config directly instead of get_benchmark_config
        self.benchmark_config = config
        self.metrics = BenchmarkMetrics()

        # Initialize retrieval engine based on unified config
        self.retrieval_pipeline = self._init_retrieval_pipeline()

        # Initialize generation engine (optional)
        self.generation_engine = self._init_generation_engine()

    def _init_retrieval_pipeline(self):
        """Initialize retrieval pipeline from unified configuration."""
        # Try to get retriever type from multiple config locations
        retrieval_type = None
        
        # Check if explicitly set in the config (for benchmark optimizer)
        if 'default_retriever' in self.config:
            retrieval_type = self.config['default_retriever']
        elif 'retrieval' in self.config:
            retrieval_config = self.config.get("retrieval", {})
            retrieval_type = retrieval_config.get("type")
        elif 'benchmark' in self.config and 'retrieval' in self.config['benchmark']:
            benchmark_retrieval = self.config['benchmark']['retrieval']
            retrieval_type = benchmark_retrieval.get("strategy")
        
        # Use unified config factory (will use pipeline default if retrieval_type is None)
        return RetrievalPipelineFactory.create_from_unified_config(self.config, retrieval_type)

    def _init_generation_engine(self):
        """Initialize generation engine from configuration."""
        generation_config = self.config.get("generation", {})

        if not generation_config.get("enabled", False):
            return None

        # For now, return None - generation engine can be implemented later
        return None

    def run_benchmark(
        self,
        adapter: BenchmarkAdapter,
        tasks: List[str] = None,
        max_queries: int = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark with configurable components."""

        print(f"🚀 Running benchmark: {adapter.name}")

        retrieval_type = self.config.get(
            "retrieval", {}).get("type", "unknown")
        print(f"🔍 Retrieval strategy: {retrieval_type}")

        if self.generation_engine:
            print(
                f"🤖 Generation provider: {self.generation_engine.provider_name}")

        # Load queries
        queries = adapter.load_queries()
        if max_queries:
            queries = queries[:max_queries]

        print(f"📊 Evaluating {len(queries)} queries")

        results = []

        # Process each query with progress bar
        for query in tqdm(queries, desc="Processing queries"):
            result = self._evaluate_query(query, adapter)
            results.append(result)

        # Aggregate results
        return self._aggregate_results(results, adapter.name)

    def _evaluate_query(self, query: BenchmarkQuery, adapter: BenchmarkAdapter) -> BenchmarkResult:
        """Evaluate a single query with configurable components."""

        # Retrieval evaluation using the pipeline
        start_time = time.time()

        # Use the pipeline's run method
        search_results = self.retrieval_pipeline.run(
            query.query_text,
            k=self.config.get("retrieval", {}).get("top_k", 20)
        )

        retrieval_time = (time.time() - start_time) * 1000

        # Extract document IDs from results
        retrieved_doc_ids = []
        for result in search_results:
            doc_id = self._extract_document_id_from_result(result)
            retrieved_doc_ids.append(str(doc_id))

        # Compute retrieval metrics
        retrieval_scores = {}
        if query.relevant_doc_ids:
            retrieval_scores = self.metrics.retrieval_metrics(
                retrieved_doc_ids,
                query.relevant_doc_ids,
                k_values=self.config.get("evaluation", {}).get(
                    "k_values", [1, 5, 10, 20])
            )
        else:
            # If no ground truth, return NaN metrics to indicate unavailable evaluation
            k_values = self.config.get("evaluation", {}).get(
                "k_values", [1, 5, 10, 20])
            for k in k_values:
                retrieval_scores[f"precision@{k}"] = float('nan')
                retrieval_scores[f"recall@{k}"] = float('nan')
                retrieval_scores[f"ndcg@{k}"] = float('nan')
            retrieval_scores["mrr"] = float('nan')

        # Generation evaluation (if enabled)
        generation_scores = {}
        generated_answer = None
        generation_time = 0.0

        if query.expected_answer and self.generation_engine:
            start_time = time.time()
            generated_answer = self.generation_engine.generate(
                query=query.query_text,
                context_docs=search_results[:self.config.get(
                    "generation", {}).get("context_limit", 5)]
            )
            generation_time = (time.time() - start_time) * 1000

            generation_scores = self.metrics.generation_metrics(
                generated_answer,
                query.expected_answer
            )

        # Combine all scores
        all_scores = {**retrieval_scores, **generation_scores}

        return BenchmarkResult(
            query_id=query.query_id,
            retrieved_docs=retrieved_doc_ids,
            generated_answer=generated_answer,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            scores=all_scores
        )

    def _aggregate_results(self, results: List[BenchmarkResult], dataset_name: str) -> Dict[str, Any]:
        """Aggregate individual results into final metrics."""

        # Collect all scores
        all_scores = {}
        for result in results:
            if result.scores:
                for metric, score in result.scores.items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)

        # Get pipeline component names
        component_names = []
        if hasattr(self.retrieval_pipeline, 'components'):
            component_names = [
                comp.component_name for comp in self.retrieval_pipeline.components]

        # Compute averages and stats, handling NaN values
        aggregated = {
            "dataset": dataset_name,
            "config": {
                "retrieval_strategy": self.config.get("retrieval", {}).get("type", "unknown"),
                "generation_enabled": self.generation_engine is not None,
                "total_queries": len(results),
                "components": component_names
            },
            "performance": {
                "avg_retrieval_time_ms": np.mean([r.retrieval_time_ms for r in results]),
                "avg_generation_time_ms": np.mean([r.generation_time_ms for r in results]),
                "total_time_ms": sum(r.retrieval_time_ms + r.generation_time_ms for r in results)
            },
            "metrics": {}
        }

        # Handle metrics with proper NaN handling
        for metric, scores in all_scores.items():
            # Filter out NaN values for computation
            valid_scores = [s for s in scores if not np.isnan(s)]

            if valid_scores:
                aggregated["metrics"][metric] = {
                    "mean": np.mean(valid_scores),
                    "std": np.std(valid_scores),
                    "min": np.min(valid_scores),
                    "max": np.max(valid_scores),
                    "median": np.median(valid_scores),
                    "count": len(valid_scores),
                    "total_queries": len(scores)
                }
            else:
                aggregated["metrics"][metric] = {
                    "mean": float('nan'),
                    "std": float('nan'),
                    "min": float('nan'),
                    "max": float('nan'),
                    "median": float('nan'),
                    "count": 0,
                    "total_queries": len(scores),
                    "note": "No ground truth available for evaluation"
                }

        return aggregated

    def _extract_document_id_from_result(self, result) -> str:
        """
        Extract document ID from retrieval result.

        For Qdrant, we need to get the external_id from the payload since
        LangChain doesn't expose it in the document metadata.
        """
        # First try: check if external_id is in document metadata
        if hasattr(result, 'metadata') and result.metadata:
            doc_id = result.metadata.get("external_id")
            if doc_id:
                return str(doc_id)

        # Second try: check document's metadata directly
        if hasattr(result, 'page_content'):
            # This is a Document object, check its metadata
            if hasattr(result, 'metadata') and result.metadata:
                doc_id = result.metadata.get("external_id")
                if doc_id:
                    return str(doc_id)

        # Third try: if result has document attribute
        if hasattr(result, 'document'):
            if hasattr(result.document, 'metadata') and result.document.metadata:
                doc_id = result.document.metadata.get("external_id")
                if doc_id:
                    return str(doc_id)

        # Fourth try: For complex document IDs, try to extract the external_id part
        # Look for patterns like "stackoverflow_sosum:a_123456:hash" -> "a_123456"
        try:
            # Check all possible metadata locations for any ID-like fields
            metadata_sources = []

            if hasattr(result, 'metadata') and result.metadata:
                metadata_sources.append(result.metadata)
            if hasattr(result, 'document') and hasattr(result.document, 'metadata'):
                metadata_sources.append(result.document.metadata)

            for metadata in metadata_sources:
                for key, value in metadata.items():
                    if isinstance(value, str):
                        # Try to extract answer ID from complex document IDs
                        if ':a_' in value:
                            # Pattern: "stackoverflow_sosum:a_123456:hash"
                            parts = value.split(':')
                            for part in parts:
                                if part.startswith('a_'):
                                    return part

                        # Direct match for answer IDs
                        if value.startswith('a_') and value.replace('a_', '').replace('_', '').isdigit():
                            return value

        except Exception as e:
            pass

        # Fallback to unknown if no ID found
        return "unknown"
