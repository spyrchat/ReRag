import os
import csv
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from mongodb_utils import connect_to_mongodb, MongoAtlasRetriever
from embeddings import TitanEmbeddingWrapper
from collections import defaultdict


def build_results(queries, retriever, max_queries=5) -> dict:
    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        if i >= max_queries:
            break
        print(f"\nRunning query {qid}: {query_text}")
        top_docs = retriever.retrieve(query_text)
        print(f"Top {len(top_docs)} docs retrieved:")
        for doc in top_docs:
            print(f"- doc_id: {doc.get('doc_id')}, score: {doc.get('score')}")
        results[qid] = {
            doc["doc_id"]: float(doc.get("score", 1.0))
            for doc in top_docs
        }
    return results


def save_results_csv(metrics_path: str, k_values: list, ndcg, recall, precision, mrr):
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


def check_overlap(results, qrels):
    overlaps = defaultdict(list)
    for qid, retrieved in results.items():
        gt = set(qrels[qid].keys())
        retrieved_ids = set(retrieved.keys())
        common = gt.intersection(retrieved_ids)
        if common:
            overlaps[qid] = list(common)
    return overlaps


def main():
    dataset_path = "trec-covid"
    corpus, queries, qrels = GenericDataLoader(dataset_path).load("test")
    client = connect_to_mongodb()
    collection = client["aws_gen_ai"]["TrecCovid"]

    embedding_wrapper = TitanEmbeddingWrapper(
        model="amazon.titan-embed-text-v2:0")
    retriever = MongoAtlasRetriever(
        collection, embedding_wrapper, index_name="vector_search", top_k=100
    )

    print("\nRunning retrieval on a small sample...")
    results = build_results(queries, retriever, max_queries=5)

    print("\nChecking overlap with ground truth...")
    overlaps = check_overlap(results, qrels)
    print(f"Queries with at least one relevant retrieved doc: {len(overlaps)}")
    for qid, matched in list(overlaps.items())[:3]:
        print(f"Query {qid} matched relevant doc_ids: {matched}")

    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5]
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, k_values)
    mrr = evaluator.evaluate_custom(qrels, results, k_values, metric="mrr")

    print("\nEvaluation Metrics:")
    for k in k_values:
        print(
            f"nDCG@{k}: {ndcg.get(k, 0):.4f} | Recall@{k}: {recall.get(k, 0):.4f} | Precision@{k}: {precision.get(k, 0):.4f} | MRR@{k}: {mrr.get(k, 0):.4f}"
        )

    output_file = "results/trec-covid_metrics.csv"
    save_results_csv(output_file, k_values, ndcg, recall, precision, mrr)
    print(f"Saved evaluation metrics to: {output_file}")


if __name__ == "__main__":
    main()
