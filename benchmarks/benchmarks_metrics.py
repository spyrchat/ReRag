"""Comprehensive evaluation metrics for RAG systems."""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import ndcg_score
import bert_score


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

        # Precision@K
        for k in k_values:
            retrieved_k = retrieved_docs[:k]
            relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
            metrics[f"precision@{k}"] = relevant_retrieved / \
                min(k, len(retrieved_docs))

        # Recall@K
        for k in k_values:
            retrieved_k = retrieved_docs[:k]
            relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
            metrics[f"recall@{k}"] = relevant_retrieved / \
                len(relevant_docs) if relevant_docs else 0

        # Mean Reciprocal Rank (MRR)
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        else:
            metrics["mrr"] = 0.0

        # NDCG@K (requires relevance scores)
        for k in k_values:
            # Simplified binary relevance
            y_true = [
                1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
            # Position-based scores
            y_score = [1.0 / (i + 1) for i in range(len(y_true))]
            if sum(y_true) > 0:
                metrics[f"ndcg@{k}"] = ndcg_score([y_true], [y_score])
            else:
                metrics[f"ndcg@{k}"] = 0.0

        return metrics

    @staticmethod
    def generation_metrics(
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, float]:
        """Compute text generation metrics."""
        metrics = {}

        # BLEU Score
        from nltk.translate.bleu_score import sentence_bleu
        reference_tokens = reference_answer.lower().split()
        generated_tokens = generated_answer.lower().split()
        metrics["bleu"] = sentence_bleu([reference_tokens], generated_tokens)

        # ROUGE Scores
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_answer, generated_answer)
        metrics["rouge1"] = rouge_scores['rouge1'].fmeasure
        metrics["rouge2"] = rouge_scores['rouge2'].fmeasure
        metrics["rougeL"] = rouge_scores['rougeL'].fmeasure

        # BERTScore
        P, R, F1 = bert_score.score(
            [generated_answer], [reference_answer], lang="en")
        metrics["bert_score_f1"] = F1.item()

        # Semantic Similarity (using your embeddings)
        # metrics["semantic_similarity"] = self._compute_semantic_similarity(generated_answer, reference_answer)

        return metrics
