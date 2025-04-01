import os
import csv
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from mongodb_utils import connect_to_mongodb, MongoAtlasRetriever
from embeddings import TitanEmbeddingWrapper


def build_results(queries, retriever, max_queries=5) -> dict:
    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        if i >= max_queries:
            break
        top_docs = retriever.retrieve(query_text)
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


def main():
    dataset_path = "trec-covid"
    corpus, queries, qrels = GenericDataLoader(dataset_path).load("test")
    client = connect_to_mongodb()
    collection = client["aws_gen_ai"]["TrecCovid"]

    # Initialize the embedding wrapper
    embedding_wrapper = TitanEmbeddingWrapper(
        model="amazon.titan-embed-text-v2:0")

    # Initialize the MongoAtlasRetriever with the embedding wrapper
    retriever = MongoAtlasRetriever(
        collection, embedding_wrapper, index_name="vector_search", top_k=5)

    print("Running retrieval on a small sample...")
    results = build_results(queries, retriever, max_queries=5)

    # Debugging: Check results and qrels
    print(f"Results: {results}")
    print(f"Qrels keys: {list(qrels.keys())[:5]}")
    print(f"Results keys: {list(results.keys())[:5]}")

    # Evaluate metrics
    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5]  # Adjusted to match top_k
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, k_values)
    mrr = evaluator.evaluate_custom(qrels, results, k_values, metric="mrr")

    print("\nEvaluation Metrics:")
    for k in k_values:
        try:
            print(
                f"nDCG@{k}: {ndcg.get(k, 0):.4f} | Recall@{k}: {recall.get(k, 0):.4f} | Precision@{k}: {precision.get(k, 0):.4f} | MRR@{k}: {mrr.get(k, 0):.4f}")
        except KeyError:
            print(f"Metrics for k={k} could not be computed.")

    # Save CSV results
    output_file = "results/trec-covid_sample_results.csv"
    save_results_csv(results, queries, corpus, output_file)
    print(f"Saved top-k results to: {output_file}")


if __name__ == "__main__":
    main()
