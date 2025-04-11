import os
import csv
import logging
from collections import defaultdict
from typing import Dict, List

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from mongodb_utils import connect_to_mongodb, MongoAtlasRetriever
from embeddings import TitanEmbeddingWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def build_results(queries: Dict[str, str], retriever: MongoAtlasRetriever) -> Dict[str, Dict[str, float]]:
    """
    Retrieve top documents for all queries, build a BEIR-compatible results dict:
    {
      query_id: { doc_id: float_score, doc_id: float_score, ... },
      ...
    }
    """
    results = {}
    for qid, query_text in queries.items():
        logging.info(f"\nRunning query {qid}: {query_text}")
        top_docs = retriever.retrieve(query_text)

        logging.info(f"Top {len(top_docs)} docs retrieved for query {qid}:")
        for doc in top_docs:
            logging.info(
                f"   doc_id={doc.get('doc_id')} score={doc.get('score')}")

        # Convert each retrieved doc into BEIR format
        # If doc["score"] is None, default to 1.0 (or 0.0) to avoid float cast error
        results[qid] = {
            doc["doc_id"]: float(
                doc["score"]) if doc["score"] is not None else 1.0
            for doc in top_docs
        }
    return results


def save_results_csv(
    metrics_path: str,
    k_values: List[int],
    ndcg: Dict[int, float],
    recall: Dict[int, float],
    precision: Dict[int, float],
    mrr: Dict[int, float],
):
    """
    Save evaluation metrics to a CSV file.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["k", "nDCG", "Recall", "Precision", "MRR"])
        for k in k_values:
            writer.writerow([
                k,
                f"{ndcg.get(k, 0):.4f}",
                f"{recall.get(k, 0):.4f}",
                f"{precision.get(k, 0):.4f}",
                f"{mrr.get(k, 0):.4f}",
            ])


def check_overlap(results: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
    """
    For each query in 'results', find the doc_ids that appear in the ground-truth qrels.
    Return a mapping: { query_id: [matched_doc_id, ...], ... }
    """
    overlaps = defaultdict(list)
    for qid, retrieved_docs in results.items():
        if qid not in qrels:
            # If there's no ground truth for that query, skip
            continue
        # doc_ids that are relevant in qrels
        gt_doc_ids = set(qrels[qid].keys())
        retrieved_doc_ids = set(retrieved_docs.keys())
        common = gt_doc_ids.intersection(retrieved_doc_ids)
        if common:
            overlaps[qid] = list(common)
    return overlaps


def main():
    # 1) Load TREC-COVID data from local folder "trec-covid"
    dataset_path = "trec-covid"
    corpus, queries, qrels = GenericDataLoader(dataset_path).load("test")

    # 2) Connect to your MongoDB Atlas collection
    client = connect_to_mongodb()
    collection = client["aws_gen_ai"]["TrecCovid"]

    # 3) Setup Amazon Titan embeddings + MongoAtlasRetriever
    embedding_wrapper = TitanEmbeddingWrapper(
        model="amazon.titan-embed-text-v2:0")
    retriever = MongoAtlasRetriever(
        collection=collection,
        embedding_wrapper=embedding_wrapper,
        index_name="vector_search",
        top_k=100
    )

    logging.info(
        "\nRunning retrieval on ALL queries in the TREC-COVID test set...")
    results = build_results(queries, retriever)

    logging.info("\nChecking overlap with ground truth Qrels...")
    overlaps = check_overlap(results, qrels)
    logging.info(f"Queries with at least one relevant doc: {len(overlaps)}")
    for qid, matched_ids in list(overlaps.items())[:5]:
        logging.info(f"Query={qid} matched doc_ids: {matched_ids}")
        # # optionally: show relevance
        # for doc_id in matched_ids:
        #     rel = qrels[qid].get(doc_id, 0)
        #     logging.info(f"   doc_id={doc_id}, qrel_relevance={rel}")

    # 4) Evaluate with standard BEIR metrics
    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, k_values)
    mrr = evaluator.evaluate_custom(qrels, results, k_values, metric="mrr")


if __name__ == "__main__":
    main()
