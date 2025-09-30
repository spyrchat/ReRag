import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.retrieval_pipeline import RetrievalPipelineFactory
from benchmarks_metrics import BenchmarkMetrics
from benchmark_contracts import BenchmarkAdapter, BenchmarkQuery, BenchmarkResult
from tqdm import tqdm
from typing import List, Dict, Any
import logging
import numpy as np
import time


logger = logging.getLogger("benchmark_runner")


class BenchmarkRunner:
    """Execute benchmarks against configurable RAG systems."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with complete, self-contained configuration."""
        self.config = config
        self.benchmark_config = config  # No separate benchmark config
        self.metrics = BenchmarkMetrics()

        print(f"🔧 Initializing BenchmarkRunner with isolated config")

        # Validate config completeness
        self._validate_config_completeness()

        # Initialize retrieval engine based on provided config only
        self.retrieval_pipeline = self._init_retrieval_pipeline()

        # Initialize generation engine (optional)
        self.generation_engine = self._init_generation_engine()

    def _validate_config_completeness(self):
        """Validate config completeness based on simplified scenario structure."""
        print(f"🔍 Validating scenario configuration...")

        # Core required sections for benchmark scenarios
        required_sections = ['retrieval', 'evaluation']
        missing = []

        # Check core sections exist
        for section in required_sections:
            if section not in self.config:
                missing.append(section)

        # Validate retrieval section
        retrieval_config = self.config.get('retrieval', {})
        if retrieval_config:
            if 'type' not in retrieval_config:
                missing.append('retrieval.type')

            # Check embedding config is present (flexible location)
            has_embedding = (
                'embedding' in retrieval_config or  # Simplified: embedding in retrieval section
                'embedding' in self.config           # Legacy: embedding at root
            )
            if not has_embedding:
                missing.append('embedding configuration')

            # Check Qdrant config is present (flexible location)
            has_qdrant = (
                'qdrant' in retrieval_config or     # Simplified: qdrant in retrieval section
                'qdrant' in self.config             # Legacy: qdrant at root
            )
            if not has_qdrant:
                missing.append('qdrant configuration')
        else:
            missing.append('retrieval section')

        # Validate evaluation section
        evaluation_config = self.config.get('evaluation', {})
        if evaluation_config:
            if 'k_values' not in evaluation_config:
                missing.append('evaluation.k_values')
            if 'metrics' not in evaluation_config:
                missing.append('evaluation.metrics')
        else:
            missing.append('evaluation section')

        # Validate dataset section (required for benchmarking)
        if 'dataset' not in self.config:
            missing.append('dataset configuration')
        else:
            dataset_config = self.config['dataset']
            if 'path' not in dataset_config:
                missing.append('dataset.path')

        # Check for experiment metadata (helpful but not critical)
        optional_missing = []
        if 'name' not in self.config:
            optional_missing.append('name')
        if 'description' not in self.config:
            optional_missing.append('description')
        if 'experiment_name' not in self.config:
            optional_missing.append('experiment_name')

        # Report results
        if missing:
            raise ValueError(f"Scenario configuration incomplete. Missing required: {missing}. "
                             f"Each benchmark scenario must have: retrieval (with type, embedding, qdrant), "
                             f"evaluation (with k_values, metrics), and dataset (with path) sections.")

        if optional_missing:
            print(f"⚠️  Optional metadata missing: {optional_missing}")

        print(f"✅ Scenario configuration validation passed")

        # Show what we found for debugging
        retrieval_type = retrieval_config.get('type', 'unknown')
        embedding_location = 'retrieval section' if 'embedding' in retrieval_config else 'root level'
        qdrant_location = 'retrieval section' if 'qdrant' in retrieval_config else 'root level'

        print(f"📋 Scenario summary:")
        print(f"   Name: {self.config.get('name', 'Unnamed')}")
        print(f"   Retrieval type: {retrieval_type}")
        print(f"   Embedding config: {embedding_location}")
        print(f"   Qdrant config: {qdrant_location}")
        print(f"   K-values: {evaluation_config.get('k_values', [])}")
        print(f"   Max queries: {self.config.get('max_queries', 'all')}")

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
        print(f"🚀 Running benchmark: {adapter.name}")
        retrieval_type = self.config.get(
            "retrieval", {}).get("type", "unknown")
        print(f"🔍 Retrieval strategy: {retrieval_type}")

        queries = adapter.load_queries()
        if max_queries:
            queries = queries[:max_queries]

        print(f"📊 Evaluating {len(queries)} queries")

        results = []
        start_total = time.time()
        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            start_query = time.time()
            result = self._evaluate_query(query, adapter)
            end_query = time.time()
            logger.info(
                f"Processed query {i + 1}/{len(queries)} ({query.query_id}) in {end_query - start_query:.2f}s")
            results.append(result)
        end_total = time.time()
        logger.info(f"Processed all queries in {end_total - start_total:.2f}s")

        return self._aggregate_results(results, adapter.name)

    def run_benchmark_with_individual_results(
        self,
        adapter: BenchmarkAdapter,
        tasks: List[str] = None,
        max_queries: int = None
    ) -> Dict[str, Any]:
        """Run benchmark and return both aggregated and individual results."""

        print(f"🚀 Running benchmark: {adapter.name}")

        # Load queries
        queries = adapter.load_queries()
        if max_queries:
            queries = queries[:max_queries]

        print(f"📊 Evaluating {len(queries)} queries")

        results = []
        individual_scores = {}  # Store individual scores per metric

        # Process each query
        for query in tqdm(queries, desc="Processing queries"):
            result = self._evaluate_query(query, adapter)
            results.append(result)

            # Collect individual scores
            for metric, score in result.scores.items():
                if metric not in individual_scores:
                    individual_scores[metric] = []
                individual_scores[metric].append(score)

        # Aggregate results with individual scores
        aggregated = self._aggregate_results(results, adapter.name)

        # Add individual scores to metrics for CI calculation
        for metric, scores in individual_scores.items():
            if metric in aggregated['metrics']:
                aggregated['metrics'][metric]['scores'] = scores

        return aggregated

    def _evaluate_query(self, query: BenchmarkQuery, adapter: BenchmarkAdapter) -> BenchmarkResult:
        start_retrieval = time.time()
        search_results = self.retrieval_pipeline.run(
            query.query_text,
            k=self.config.get("retrieval", {}).get("top_k", 20)
        )
        end_retrieval = time.time()
        logger.info(
            f"Retrieval for query {query.query_id} took {end_retrieval - start_retrieval:.2f}s")

        retrieval_time = (end_retrieval - start_retrieval) * 1000

        # Extract document IDs from results
        retrieved_chunk_ids = []
        for result in search_results:
            doc_id = self._extract_document_id_from_result(result)
            retrieved_chunk_ids.append(str(doc_id))

        print(f"   Retrieved chunk IDs: {retrieved_chunk_ids[:5]}")
        # Compute retrieval metrics
        retrieval_scores = {}
        if query.relevant_doc_ids:
            retrieval_scores = self.metrics.retrieval_metrics(
                retrieved_chunk_ids,
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
            retrieved_docs=retrieved_chunk_ids,
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
            "performance": self.metrics.retrieval_time_stats([r.retrieval_time_ms for r in results]),
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
        per_query_scores = []
        for result in results:
            query_scores = {
                'query_id': result.query_id,
                **result.scores
            }
            per_query_scores.append(query_scores)

        aggregated["per_query_scores"] = per_query_scores

        return aggregated

    def _extract_document_id_from_result(self, result) -> str:
        # Print for debugging
        print("DEBUG result:", result)
        print("DEBUG payload:", getattr(result, "payload", None))

        # Try payload
        if hasattr(result, "payload") and result.payload:
            if "chunk_id" in result.payload:
                return result.payload["chunk_id"]
            # Try doc_id if chunk_id missing
            if "doc_id" in result.payload:
                return result.payload["doc_id"]

        # Try metadata
        if hasattr(result, 'metadata') and result.metadata:
            if "chunk_id" in result.metadata:
                return result.metadata["chunk_id"]
            if "doc_id" in result.metadata:
                return result.metadata["doc_id"]

        # Try document
        if hasattr(result, 'document') and hasattr(result.document, 'metadata'):
            if "chunk_id" in result.document.metadata:
                return result.document.metadata["chunk_id"]
            if "doc_id" in result.document.metadata:
                return result.document.metadata["doc_id"]

        return "unknown"
