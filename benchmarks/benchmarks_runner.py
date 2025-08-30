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
        retrieval_config = self.config.get("retrieval", {})
        retrieval_type = retrieval_config.get("type", "dense")

        # Use unified config factory
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
            doc_id = result.metadata.get(
                "external_id") or result.metadata.get("id") or "unknown"
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
            # If no ground truth, use dummy metrics for testing
            retrieval_scores = self.metrics.retrieval_metrics(
                retrieved_doc_ids,
                retrieved_doc_ids[:1] if retrieved_doc_ids else [],
                k_values=self.config.get("evaluation", {}).get(
                    "k_values", [1, 5, 10, 20])
            )

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

        # Compute averages and stats
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
            "metrics": {
                metric: {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores)
                }
                for metric, scores in all_scores.items()
            }
        }

        return aggregated
