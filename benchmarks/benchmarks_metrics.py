"""Comprehensive evaluation metrics for RAG systems."""

from typing import List, Dict, Any
import numpy as np


class BenchmarkMetrics:
    """Collection of evaluation metrics for RAG systems."""

    @staticmethod
    def retrieval_metrics(
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """Compute retrieval metrics."""
        metrics = {}

        # If no ground truth is available, return NaN metrics to indicate unavailable evaluation
        if not relevant_docs:
            for k in k_values:
                metrics[f"precision@{k}"] = float('nan')
                metrics[f"recall@{k}"] = float('nan')
                metrics[f"ndcg@{k}"] = float('nan')
            metrics["mrr"] = float('nan')
            return metrics

        # Precision@K
        for k in k_values:
            retrieved_k = retrieved_docs[:k]
            if retrieved_k:
                relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
                metrics[f"precision@{k}"] = relevant_retrieved / \
                    len(retrieved_k)
            else:
                metrics[f"precision@{k}"] = 0.0

        # Recall@K
        for k in k_values:
            retrieved_k = retrieved_docs[:k]
            if relevant_docs:
                relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
                metrics[f"recall@{k}"] = relevant_retrieved / \
                    len(relevant_docs)
            else:
                metrics[f"recall@{k}"] = 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr

        # NDCG@K (simplified binary relevance)
        for k in k_values:
            retrieved_k = retrieved_docs[:k]
            if retrieved_k and relevant_docs:
                # Binary relevance: 1 if relevant, 0 if not
                relevance_scores = [
                    1.0 if doc in relevant_docs else 0.0 for doc in retrieved_k]
                dcg = sum(rel / np.log2(i + 2)
                          for i, rel in enumerate(relevance_scores))

                # Ideal DCG (best possible ordering)
                ideal_relevance = sorted(relevance_scores, reverse=True)
                idcg = sum(rel / np.log2(i + 2)
                           for i, rel in enumerate(ideal_relevance))

                metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
            else:
                metrics[f"ndcg@{k}"] = 0.0

        # F1@K
        for k in k_values:
            precision_k = metrics[f"precision@{k}"]
            recall_k = metrics[f"recall@{k}"]

            # Handle division by zero in F1 calculation
            if precision_k + recall_k > 0:
                metrics[f"f1@{k}"] = 2 * \
                    (precision_k * recall_k) / (precision_k + recall_k)
            else:
                metrics[f"f1@{k}"] = 0.0

        def calculate_map(retrieved_docs, relevant_docs):
            if not relevant_docs:
                return 0.0

            average_precision = 0.0
            relevant_found = 0

            for i, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    relevant_found += 1
                    precision_at_i = relevant_found / (i + 1)
                    average_precision += precision_at_i

            return average_precision / len(relevant_docs)

        metrics["map"] = calculate_map(retrieved_docs, relevant_docs)

        # R-Precision (precision at R, where R = number of relevant docs)
        r = len(relevant_docs)
        if r > 0 and len(retrieved_docs) >= r:
            retrieved_r = retrieved_docs[:r]
            relevant_retrieved_r = len(set(retrieved_r) & set(relevant_docs))
            metrics["r_precision"] = relevant_retrieved_r / r
        else:
            metrics["r_precision"] = 0.0

        # Success@K (binary: found at least one relevant doc in top-k)
        for k in k_values:
            retrieved_k = retrieved_docs[:k]
            has_relevant = any(doc in relevant_docs for doc in retrieved_k)
            metrics[f"success@{k}"] = 1.0 if has_relevant else 0.0

        return metrics

    @staticmethod
    def generation_metrics(
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, float]:
        """Compute simple text generation metrics."""
        metrics = {}

        if not reference_answer:
            return {"length_ratio": 0.0, "character_overlap": 0.0}

        # Simple metrics without external dependencies
        metrics["length_ratio"] = len(generated_answer) / len(reference_answer)

        # Character overlap ratio
        gen_chars = set(generated_answer.lower())
        ref_chars = set(reference_answer.lower())
        overlap = len(gen_chars & ref_chars)
        metrics["character_overlap"] = overlap / \
            len(ref_chars) if ref_chars else 0.0

        # Word overlap ratio
        gen_words = set(generated_answer.lower().split())
        ref_words = set(reference_answer.lower().split())
        word_overlap = len(gen_words & ref_words)
        metrics["word_overlap"] = word_overlap / \
            len(ref_words) if ref_words else 0.0

        return metrics
