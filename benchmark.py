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
    """Retrieve top documents for all queries and build a BEIR-compatible results dict."""
    results = {}
    for qid, query_text in queries.items():
        logging.info(f"\nRunning query {qid}: {query_text}")
        top_docs = retriever.retrieve(query_text)

        logging.info(f"Top {len(top_docs)} docs retrieved for query {qid}:")
        for doc in top_docs:
            logging.info(
                f"   doc_id={doc.get('doc_id')} score={doc.get('score')}")

        results[qid] = {
            doc["doc_id"]: float(
                doc["score"]) if doc["score"] is not None else 1.0
            for doc in top_docs
        }
    return results


def check_overlap(results: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
    """Check which retrieved documents overlap with the ground-truth qrels."""
    overlaps = defaultdict(list)
    for qid, retrieved_docs in results.items():
        if qid not in qrels:
            continue
        gt_doc_ids = set(qrels[qid].keys())
        retrieved_doc_ids = set(retrieved_docs.keys())
        common = gt_doc_ids.intersection(retrieved_doc_ids)
        if common:
            overlaps[qid] = list(common)
    return overlaps


def main():
    """Main pipeline: Load data, connect to Mongo, retrieve results, evaluate, and save metrics."""
    dataset_path = "trec-covid"
    corpus, queries, qrels = GenericDataLoader(dataset_path).load("test")

    client = connect_to_mongodb()
    collection = client["aws_gen_ai"]["TrecCovid"]

    embedding_wrapper = TitanEmbeddingWrapper(
        model="amazon.titan-embed-text-v2:0")
    retriever = MongoAtlasRetriever(
        collection=collection,
        embedding_wrapper=embedding_wrapper,
        index_name="vector_search",
        top_k=500
    )

    logging.info(
        "\nRunning retrieval on ALL queries in the TREC-COVID test set...")
    results = build_results(queries, retriever)
    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5, 10]

    # Evaluate all metrics
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, k_values)


if __name__ == "__main__":
    main()
