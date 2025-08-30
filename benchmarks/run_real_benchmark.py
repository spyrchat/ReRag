"""Benchmark runner using real StackOverflow data."""

import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from benchmarks.benchmarks_runner import BenchmarkRunner
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
from benchmarks.benchmark_contracts import BenchmarkQuery
from config.config_loader import load_config


def run_real_stackoverflow_benchmark():
    """Run benchmark with real StackOverflow questions."""

    print("🚀 Starting StackOverflow Benchmark with REAL Data")

    # Load your main configuration
    config = load_config("config.yml")

    # Override for benchmarking
    config["retrieval"] = {
        "type": "dense",  # Start with dense for simplicity
        "top_k": 10,
        "score_threshold": 0.1
    }

    config["evaluation"] = {
        "k_values": [1, 5, 10],
        "metrics": {
            "retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"]
        }
    }

    # Create custom adapter for real data
    class RealStackOverflowAdapter(StackOverflowBenchmarkAdapter):
        def load_queries(self, split: str = "test"):
            """Load from question.csv specifically."""
            import pandas as pd

            question_file = self.dataset_path / "question.csv"
            print(f"📂 Loading from {question_file}")

            try:
                df = pd.read_csv(question_file)
                print(f"📊 Found {len(df)} questions in dataset")

                queries = []
                for idx, row in df.iterrows():
                    if idx >= 20:  # Limit to 20 real questions
                        break

                    if pd.isna(row['question_title']) or not row['question_title']:
                        continue

                    from benchmarks.benchmark_contracts import BenchmarkQuery
                    query = BenchmarkQuery(
                        query_id=f"real_so_{row['question_id']}",
                        query_text=str(row['question_title']),
                        expected_answer=str(row['question_body'])[:500] if not pd.isna(
                            row['question_body']) else None,
                        relevant_doc_ids=None,  # No ground truth available
                        difficulty="medium",
                        category=str(row['tags']) if not pd.isna(
                            row['tags']) else "programming",
                        metadata={
                            "original_question_id": row['question_id'],
                            "question_type": row['question_type'],
                            "tags": row['tags'],
                            "source": "real_stackoverflow"
                        }
                    )
                    queries.append(query)

                print(f"✅ Loaded {len(queries)} real StackOverflow queries")
                return queries

            except Exception as e:
                print(f"❌ Error loading real data: {e}")
                return self._create_dummy_queries()

    # Initialize components
    runner = BenchmarkRunner(config)
    adapter = RealStackOverflowAdapter(
        dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data"
    )

    # Run benchmark
    print("📊 Running benchmark with real data...")
    results = runner.run_benchmark(
        adapter=adapter,
        max_queries=10  # Test with 10 real questions
    )

    # Print results
    print("\n📊 REAL STACKOVERFLOW BENCHMARK RESULTS:")
    print(f"Dataset: {results['dataset']}")
    print(f"Total Queries: {results['config']['total_queries']}")
    print(f"Avg Time: {results['performance']['avg_retrieval_time_ms']:.2f}ms")
    print(f"Components: {', '.join(results['config']['components'])}")

    print("\n🎯 Metrics:")
    for metric_name in ['precision@5', 'precision@10', 'recall@5', 'recall@10', 'mrr']:
        if metric_name in results['metrics']:
            stats = results['metrics'][metric_name]
            print(
                f"  {metric_name:12}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    print("\n📋 Sample Query Results:")
    # The results don't include individual queries, but we can see the overall performance

    return results


if __name__ == "__main__":
    run_real_stackoverflow_benchmark()
