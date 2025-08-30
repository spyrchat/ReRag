"""Simple benchmark runner using unified configuration."""

from config.config_loader import load_config
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
from benchmarks.benchmarks_runner import BenchmarkRunner
import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')


def run_simple_benchmark():
    """Run a simple benchmark with Google Gemini embeddings."""

    print("🚀 Starting StackOverflow Benchmark with Google Gemini")

    # Load your main configuration
    config = load_config("config.yml")

    # Override for comprehensive benchmarking
    config["retrieval"] = {
        "type": "dense",  # Start with dense for simplicity
        "top_k": 20,      # More results for better evaluation
        "score_threshold": 0.0  # Get all results, no filtering
    }

    config["evaluation"] = {
        "k_values": [1, 3, 5, 10, 20],  # More comprehensive evaluation
        "metrics": {
            "retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"]
        }
    }

    # Initialize components
    runner = BenchmarkRunner(config)
    adapter = StackOverflowBenchmarkAdapter(
        dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data"
    )

    # Run benchmark with more queries for full evaluation
    print("📊 Running benchmark...")
    results = runner.run_benchmark(
        adapter=adapter,
        max_queries=100  # Increased from 5 to 100 for comprehensive evaluation
    )

    # Print results
    print("\n📊 BENCHMARK RESULTS:")
    print(f"Dataset: {results['dataset']}")
    print(f"Total Queries: {results['config']['total_queries']}")
    print(f"Avg Time: {results['performance']['avg_retrieval_time_ms']:.2f}ms")

    print("\n🎯 Metrics:")
    for metric_name in ['precision@5', 'precision@10', 'recall@5', 'recall@10', 'mrr']:
        if metric_name in results['metrics']:
            stats = results['metrics'][metric_name]
            print(
                f"  {metric_name:12}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    return results


if __name__ == "__main__":
    run_simple_benchmark()
