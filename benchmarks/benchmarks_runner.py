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
            k_values = self.config.get("evaluation", {}).get("k_values", [1, 5, 10, 20])
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
        if hasattr(result, 'document') and result.document.metadata:
            doc_id = result.document.metadata.get("external_id")
            if doc_id:
                return str(doc_id)
        
        # Third try: For Qdrant, try to get the external_id from the point payload
        # We need to access the Qdrant client directly
        try:
            if hasattr(self.retrieval_pipeline, 'components'):
                for component in self.retrieval_pipeline.components:
                    if hasattr(component, 'vector_db') and hasattr(component.vector_db, 'client'):
                        qdrant_client = component.vector_db.client
                        collection_name = component.vector_db.collection_name
                        
                        # Try to find the point ID from the result
                        # This is a bit hacky but necessary due to LangChain limitations
                        content = result.document.page_content
                        
                        # Search for points with matching content (not ideal but works)
                        # Since we can't easily get the point ID from LangChain result
                        # We'll use a content-based lookup as a fallback
                        
                        # For now, we'll skip this complex lookup and rely on re-ingestion
                        break
        except Exception as e:
            pass
        
        # Fallback to unknown if no ID found
        return "unknown"
