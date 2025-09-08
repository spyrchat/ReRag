"""
Unified retrieval evaluation runner for consistent metrics across datasets.
Implements standard IR metrics with flexible gold standard handling.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import numpy as np

from pipelines.contracts import DatasetAdapter, RetrievalMetrics, EvaluationRun


logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Unified evaluation runner for retrieval systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k_values = config.get("evaluation", {}).get("k_values", [1, 3, 5, 10])
        self.similarity_threshold = config.get("evaluation", {}).get("similarity_threshold", 0.8)
        self.enable_semantic_matching = config.get("evaluation", {}).get("semantic_matching", False)
    
    def evaluate_dataset(
        self, 
        adapter: DatasetAdapter, 
        retriever: Any,
        split: str = "test"
    ) -> EvaluationRun:
        """Evaluate retriever on a dataset using its adapter."""
        logger.info(f"Evaluating {adapter.source_name} dataset (split: {split})")
        
        # Get evaluation queries from adapter
        eval_queries = adapter.get_evaluation_queries(split)
        if not eval_queries:
            raise ValueError(f"No evaluation queries found for {adapter.source_name}")
        
        logger.info(f"Running evaluation on {len(eval_queries)} queries")
        
        # Run retrieval for all queries
        query_results = []
        all_metrics = defaultdict(list)
        
        for i, query_info in enumerate(eval_queries):
            if i % 100 == 0:
                logger.info(f"Processing query {i+1}/{len(eval_queries)}")
            
            query_result = self._evaluate_single_query(query_info, retriever)
            query_results.append(query_result)
            
            # Aggregate metrics
            for metric_name, value in query_result.get("metrics", {}).items():
                all_metrics[metric_name].append(value)
        
        # Compute aggregate metrics
        aggregate_metrics = RetrievalMetrics()
        
        # Calculate averages for each k value
        for k in self.k_values:
            recall_values = all_metrics[f"recall_at_{k}"]
            precision_values = all_metrics[f"precision_at_{k}"]
            ndcg_values = all_metrics[f"ndcg_at_{k}"]
            
            if recall_values:
                aggregate_metrics.recall_at_k[k] = np.mean(recall_values)
                aggregate_metrics.precision_at_k[k] = np.mean(precision_values)
                aggregate_metrics.ndcg_at_k[k] = np.mean(ndcg_values)
        
        # Calculate MRR and MAP
        mrr_values = all_metrics["mrr"]
        map_values = all_metrics["map"]
        
        aggregate_metrics.mrr = np.mean(mrr_values) if mrr_values else 0.0
        aggregate_metrics.map_score = np.mean(map_values) if map_values else 0.0
        aggregate_metrics.total_queries = len(eval_queries)
        aggregate_metrics.total_relevant = sum(
            len(q.get("relevant_doc_ids", [])) for q in eval_queries
        )
        
        # Create evaluation run
        evaluation_run = EvaluationRun(
            dataset_name=adapter.source_name,
            dataset_version=adapter.version,
            collection_name=self.config.get("qdrant", {}).get("collection", "unknown"),
            retriever_config=self._extract_retriever_config(),
            embedding_config=self.config.get("embedding", {}),
            metrics=aggregate_metrics,
            per_query_results=query_results
        )
        
        logger.info(f"Evaluation completed. Average recall@5: {aggregate_metrics.recall_at_k.get(5, 0):.3f}")
        return evaluation_run
    
    def _evaluate_single_query(self, query_info: Dict[str, Any], retriever: Any) -> Dict[str, Any]:
        """Evaluate a single query."""
        query = query_info["query"]
        query_id = query_info.get("query_id", "unknown")
        relevant_doc_ids = set(query_info.get("relevant_doc_ids", []))
        relevance_scores = query_info.get("relevance_scores", {})
        
        try:
            # Retrieve documents
            retrieved_docs = retriever.retrieve(query)
            
            # Extract document IDs and scores
            if isinstance(retrieved_docs, list) and retrieved_docs:
                if isinstance(retrieved_docs[0], tuple):
                    # Handle (doc, score) format
                    doc_ids = [doc.metadata.get("external_id", doc.metadata.get("doc_id", "unknown")) 
                              for doc, _ in retrieved_docs]
                    scores = [score for _, score in retrieved_docs]
                else:
                    # Handle doc list format
                    doc_ids = [doc.metadata.get("external_id", doc.metadata.get("doc_id", "unknown")) 
                              for doc in retrieved_docs]
                    scores = [1.0] * len(doc_ids)  # Default scores
            else:
                doc_ids = []
                scores = []
            
            # Calculate metrics for all k values
            metrics = {}
            
            for k in self.k_values:
                # Limit to top-k
                top_k_ids = doc_ids[:k]
                top_k_scores = scores[:k]
                
                # Calculate recall@k
                if relevant_doc_ids:
                    relevant_retrieved = len(set(top_k_ids).intersection(relevant_doc_ids))
                    recall_k = relevant_retrieved / len(relevant_doc_ids)
                else:
                    recall_k = 0.0
                
                # Calculate precision@k
                precision_k = relevant_retrieved / k if k > 0 and top_k_ids else 0.0
                
                # Calculate NDCG@k
                ndcg_k = self._calculate_ndcg(top_k_ids, relevance_scores, k)
                
                metrics[f"recall_at_{k}"] = recall_k
                metrics[f"precision_at_{k}"] = precision_k
                metrics[f"ndcg_at_{k}"] = ndcg_k
            
            # Calculate MRR
            mrr = self._calculate_mrr(doc_ids, relevant_doc_ids)
            metrics["mrr"] = mrr
            
            # Calculate MAP
            map_score = self._calculate_map(doc_ids, relevant_doc_ids)
            metrics["map"] = map_score
            
            return {
                "query_id": query_id,
                "query": query,
                "retrieved_count": len(doc_ids),
                "relevant_count": len(relevant_doc_ids),
                "metrics": metrics,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            return {
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "success": False,
                "metrics": {f"{metric}_at_{k}": 0.0 for metric in ["recall", "precision", "ndcg"] for k in self.k_values}
            }
    
    def _calculate_ndcg(self, retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@k."""
        if not retrieved_ids or not relevance_scores:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_map(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculate Mean Average Precision."""
        if not relevant_ids:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_ids) if relevant_ids else 0.0
    
    def _extract_retriever_config(self) -> Dict[str, Any]:
        """Extract retriever configuration for lineage."""
        return {
            "strategy": self.config.get("embedding_strategy", "unknown"),
            "retriever_type": self.config.get("retriever", {}).get("type", "unknown"),
            "top_k": self.config.get("retriever", {}).get("top_k", 10),
            "dense_model": self.config.get("embedding", {}).get("dense", {}).get("model_name", "unknown"),
            "sparse_model": self.config.get("embedding", {}).get("sparse", {}).get("model_name", "unknown")
        }
    
    def save_results(self, evaluation_run: EvaluationRun, output_dir: Path):
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_dir / f"{evaluation_run.dataset_name}_evaluation.json"
        evaluation_run.save_to_file(results_file)
        
        # Save metrics summary
        metrics_file = output_dir / f"{evaluation_run.dataset_name}_metrics.csv"
        self._save_metrics_csv(evaluation_run.metrics, metrics_file)
        
        # Save per-query results
        query_results_file = output_dir / f"{evaluation_run.dataset_name}_per_query.csv"
        self._save_query_results_csv(evaluation_run.per_query_results, query_results_file)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def _save_metrics_csv(self, metrics: RetrievalMetrics, file_path: Path):
        """Save metrics summary as CSV."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["k", "nDCG", "Recall", "Precision", "MRR"])
            
            for k in sorted(metrics.recall_at_k.keys()):
                writer.writerow([
                    k,
                    f"{metrics.ndcg_at_k.get(k, 0):.4f}",
                    f"{metrics.recall_at_k.get(k, 0):.4f}",
                    f"{metrics.precision_at_k.get(k, 0):.4f}",
                    f"{metrics.mrr:.4f}" if k == min(metrics.recall_at_k.keys()) else ""
                ])
    
    def _save_query_results_csv(self, query_results: List[Dict[str, Any]], file_path: Path):
        """Save per-query results as CSV."""
        import csv
        
        if not query_results:
            return
        
        with open(file_path, 'w', newline='') as f:
            # Get all metric names from first successful query
            sample_metrics = {}
            for result in query_results:
                if result.get("success") and result.get("metrics"):
                    sample_metrics = result["metrics"]
                    break
            
            fieldnames = ["query_id", "query", "success"] + list(sample_metrics.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in query_results:
                row = {
                    "query_id": result.get("query_id", ""),
                    "query": result.get("query", ""),
                    "success": result.get("success", False)
                }
                
                # Add metrics
                metrics = result.get("metrics", {})
                for metric_name in sample_metrics.keys():
                    row[metric_name] = f"{metrics.get(metric_name, 0):.4f}"
                
                writer.writerow(row)


class MetricsComparator:
    """Compare evaluation results across different configurations."""
    
    @staticmethod
    def compare_runs(runs: List[EvaluationRun]) -> Dict[str, Any]:
        """Compare multiple evaluation runs."""
        if not runs:
            return {}
        
        comparison = {
            "run_count": len(runs),
            "datasets": [run.dataset_name for run in runs],
            "configurations": [
                {
                    "dataset": run.dataset_name,
                    "collection": run.collection_name,
                    "embedding_strategy": run.embedding_config.get("strategy", "unknown")
                }
                for run in runs
            ],
            "metrics_comparison": {}
        }
        
        # Compare metrics across runs
        k_values = set()
        for run in runs:
            k_values.update(run.metrics.recall_at_k.keys())
        
        for k in sorted(k_values):
            comparison["metrics_comparison"][f"recall_at_{k}"] = [
                run.metrics.recall_at_k.get(k, 0) for run in runs
            ]
            comparison["metrics_comparison"][f"precision_at_{k}"] = [
                run.metrics.precision_at_k.get(k, 0) for run in runs
            ]
            comparison["metrics_comparison"][f"ndcg_at_{k}"] = [
                run.metrics.ndcg_at_k.get(k, 0) for run in runs
            ]
        
        comparison["metrics_comparison"]["mrr"] = [run.metrics.mrr for run in runs]
        comparison["metrics_comparison"]["map"] = [run.metrics.map_score for run in runs]
        
        return comparison
    
    @staticmethod
    def save_comparison(comparison: Dict[str, Any], output_file: Path):
        """Save comparison results to JSON."""
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
