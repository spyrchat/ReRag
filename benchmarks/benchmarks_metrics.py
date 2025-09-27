"""Comprehensive evaluation metrics for RAG systems."""

from typing import List, Dict, Any, Iterable
import numpy as np
import math


class BenchmarkMetrics:
    """Collection of evaluation metrics for RAG systems."""

    @staticmethod
    def _dedup_preserve_order(items: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @staticmethod
    def retrieval_metrics(
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = None
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics under binary relevance with order-invariant golds.

        Notes:
        - relevant_docs is treated as a SET of relevant ids (no internal order).
        - retrieved_docs are de-duplicated preserving the first occurrence.
        - NDCG@k uses ideal DCG with min(k, |relevant|) ones (binary relevance).
        - R-precision = hits in top-R divided by R, even if fewer than R retrieved.
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        metrics: Dict[str, float] = {}

        # Normalize inputs
        rel_set = set(relevant_docs or [])
        ranked = BenchmarkMetrics._dedup_preserve_order(retrieved_docs or [])

        # If no ground truth, return NaNs (unavailable evaluation)
        if not rel_set:
            for k in k_values:
                metrics[f"precision@{k}"] = float("nan")
                metrics[f"recall@{k}"] = float("nan")
                metrics[f"ndcg@{k}"] = float("nan")
                metrics[f"f1@{k}"] = float("nan")
                metrics[f"success@{k}"] = float("nan")
            metrics["mrr"] = float("nan")
            metrics["map"] = float("nan")
            metrics["r_precision"] = float("nan")
            return metrics

        def precision_at_k(k: int) -> float:
            topk = ranked[:k]
            if not topk:
                return 0.0
            hits = sum(1 for d in topk if d in rel_set)
            return hits / len(topk)

        def recall_at_k(k: int) -> float:
            topk = ranked[:k]
            hits = sum(1 for d in topk if d in rel_set)
            return hits / len(rel_set)

        # Precision@K / Recall@K
        for k in k_values:
            p = precision_at_k(k)
            r = recall_at_k(k)
            metrics[f"precision@{k}"] = p
            metrics[f"recall@{k}"] = r

        # F1@K
        for k in k_values:
            p = metrics[f"precision@{k}"]
            r = metrics[f"recall@{k}"]
            metrics[f"f1@{k}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # MRR (first relevant hit)
        mrr = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in rel_set:
                mrr = 1.0 / i
                break
        metrics["mrr"] = mrr

        # MAP (Average Precision averaged over queries; here single-query AP)
        def average_precision(ranked_list: List[str], rel: set) -> float:
            if not rel:
                return 0.0
            hits = 0
            ap_sum = 0.0
            for i, doc_id in enumerate(ranked_list, start=1):
                if doc_id in rel:
                    hits += 1
                    ap_sum += hits / i
            # Denominator is |rel| (standard definition), even if not all are retrieved
            return ap_sum / len(rel)

        metrics["map"] = average_precision(ranked, rel_set)

        # NDCG@K (binary relevance, ideal assumes top ones)
        def dcg_at_k(gains: List[float]) -> float:
            return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

        num_rel = len(rel_set)
        ideal_cache: Dict[int, float] = {}
        for k in k_values:
            topk = ranked[:k]
            gains = [1.0 if d in rel_set else 0.0 for d in topk]
            dcg = dcg_at_k(gains)
            # Ideal: min(k, |rel|) ones, then zeros
            ideal_k = min(k, num_rel)
            if ideal_k not in ideal_cache:
                ideal_cache[ideal_k] = dcg_at_k([1.0] * ideal_k)
            idcg = ideal_cache[ideal_k]
            metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

        # R-Precision: precision at R (R = number of relevant docs)
        R = num_rel
        topR = ranked[:R]
        hits_R = sum(1 for d in topR if d in rel_set)
        metrics["r_precision"] = hits_R / R if R > 0 else 0.0

        # Success@K: at least one relevant in top-k
        for k in k_values:
            topk = ranked[:k]
            metrics[f"success@{k}"] = 1.0 if any(
                d in rel_set for d in topk) else 0.0

        return metrics

    @staticmethod
    def generation_metrics(
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, float]:
        """Compute simple text generation metrics (overlap-based, dependency-free)."""
        if not reference_answer:
            return {"length_ratio": 0.0, "character_overlap": 0.0, "word_overlap": 0.0}

        metrics: Dict[str, float] = {}
        metrics["length_ratio"] = len(generated_answer) / len(reference_answer)

        # Character overlap ratio (set-based; insensitive to multiplicity)
        gen_chars = set(generated_answer.lower())
        ref_chars = set(reference_answer.lower())
        metrics["character_overlap"] = (
            len(gen_chars & ref_chars) / len(ref_chars)) if ref_chars else 0.0

        # Word overlap ratio (set-based; insensitive to multiplicity)
        gen_words = set(generated_answer.lower().split())
        ref_words = set(reference_answer.lower().split())
        metrics["word_overlap"] = (
            len(gen_words & ref_words) / len(ref_words)) if ref_words else 0.0

        return metrics
