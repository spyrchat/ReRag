"""Full dataset benchmark - uses ALL queries from StackOverflow dataset."""

import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from benchmarks.benchmarks_runner import BenchmarkRunner
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
from config.config_loader import load_config


def run_full_dataset_benchmark():
    """Run benchmark with ALL StackOverflow questions in the dataset."""

    print("🚀 FULL DATASET BENCHMARK - Using ALL queries")
    print("🔍 Gemini embeddings | 📊 Top-10 retrieval | 🗂️ ALL dataset queries")

    # Load your main configuration
    config = load_config("config.yml")

    # Configuration for full dataset benchmark
    config["retrieval"] = {
        "type": "dense",           # ✅ Gemini dense embeddings
        "top_k": 10,              # ✅ Top 10 documents
        "score_threshold": 0.0     # ✅ No filtering - get all results
    }

    config["evaluation"] = {
        "k_values": [1, 3, 5, 10], # ✅ Comprehensive evaluation
        "metrics": {
            "retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"]
        }
    }

    # Create adapter that loads ALL queries
    class FullDatasetAdapter(StackOverflowBenchmarkAdapter):
        def load_queries(self, split: str = "test"):
            """Load ALL questions from question.csv."""
            import pandas as pd

            question_file = self.dataset_path / "question.csv"
            print(f"📂 Loading ALL queries from {question_file}")

            try:
                df = pd.read_csv(question_file)
                total_questions = len(df)
                print(f"📊 Found {total_questions} questions in dataset")

                queries = []
                valid_queries = 0
                
                for idx, row in df.iterrows():
                    # No limit - process ALL rows
                    if pd.isna(row['question_title']) or not row['question_title']:
                        continue

                    from benchmark_contracts import BenchmarkQuery
                    query = BenchmarkQuery(
                        query_id=f"full_so_{row['question_id']}",
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
                            "source": "full_stackoverflow_dataset"
                        }
                    )
                    queries.append(query)
                    valid_queries += 1

                print(f"✅ Loaded {valid_queries} valid queries from {total_questions} total questions")
                print(f"📊 Will evaluate retrieval on ALL {valid_queries} queries")
                
                return queries

            except Exception as e:
                print(f"❌ Error loading full dataset: {e}")
                import traceback
                traceback.print_exc()
                return self._create_dummy_queries()

    # Initialize components
    runner = BenchmarkRunner(config)
    adapter = FullDatasetAdapter(
        dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data"
    )

    # Run benchmark on ALL queries (no max_queries limit)
    print("📊 Running benchmark on FULL dataset...")
    print("⏱️  This may take a while - processing all queries...")
    
    results = runner.run_benchmark(
        adapter=adapter
        # No max_queries parameter = ALL queries will be processed
    )

    # Print comprehensive results
    print("\n" + "="*60)
    print("📊 FULL DATASET BENCHMARK RESULTS")
    print("="*60)
    
    print(f"🗂️  Dataset: {results['dataset']}")
    print(f"📊 Total Queries Processed: {results['config']['total_queries']}")
    print(f"⏱️  Average Time per Query: {results['performance']['avg_retrieval_time_ms']:.2f}ms")
    print(f"🔧 Pipeline Components: {', '.join(results['config']['components'])}")
    print(f"🎯 Retrieval Strategy: {results['config']['retrieval_strategy']}")

    # Performance summary
    total_time_seconds = results['performance']['total_time_ms'] / 1000
    queries_per_second = results['config']['total_queries'] / total_time_seconds
    
    print(f"\n⚡ Performance Summary:")
    print(f"   Total Processing Time: {total_time_seconds:.1f} seconds ({total_time_seconds/60:.1f} minutes)")
    print(f"   Processing Rate: {queries_per_second:.2f} queries/second")

    print(f"\n🎯 Retrieval Quality Metrics:")
    for metric_name in ['precision@1', 'precision@3', 'precision@5', 'precision@10', 
                       'recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr']:
        if metric_name in results['metrics']:
            stats = results['metrics'][metric_name]
            print(f"   {metric_name:15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")

    # Additional analysis
    print(f"\n📈 Statistical Summary:")
    print(f"   Median precision@5: {results['metrics']['precision@5']['median']:.4f}")
    print(f"   Median recall@5: {results['metrics']['recall@5']['median']:.4f}")
    print(f"   Mean Reciprocal Rank: {results['metrics']['mrr']['mean']:.4f}")

    return results


def run_sample_benchmark(sample_size=100):
    """Run benchmark on a sample of the dataset for quick testing."""
    
    print(f"🧪 SAMPLE BENCHMARK - Using {sample_size} queries")
    
    config = load_config("config.yml")
    config["retrieval"] = {
        "type": "dense",
        "top_k": 10,
        "score_threshold": 0.0
    }
    config["evaluation"] = {
        "k_values": [1, 5, 10],
        "metrics": {"retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"]}
    }

    runner = BenchmarkRunner(config)
    adapter = StackOverflowBenchmarkAdapter(
        dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data"
    )

    results = runner.run_benchmark(adapter=adapter, max_queries=sample_size)
    
    print(f"\n📊 SAMPLE RESULTS ({sample_size} queries):")
    print(f"Avg Time: {results['performance']['avg_retrieval_time_ms']:.2f}ms")
    for metric in ['precision@5', 'precision@10', 'mrr']:
        if metric in results['metrics']:
            stats = results['metrics'][metric]
            print(f"{metric:12}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        # Quick sample run
        run_sample_benchmark(100)
    else:
        # Full dataset run
        run_full_dataset_benchmark()
