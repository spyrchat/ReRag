"""Full dataset benchmark runner for comprehensive retrieval evaluation."""

import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from config.config_loader import load_config
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
from benchmarks.benchmarks_runner import BenchmarkRunner
import pandas as pd
import json
import time
from pathlib import Path


def run_full_dataset_benchmark():
    """Run benchmark against the full dataset in Qdrant for retrieval evaluation."""

    print("🚀 Starting FULL DATASET Benchmark - Retrieval Evaluation")
    print("📊 Evaluating against all 6,210 documents in Qdrant")

    # Load your main configuration
    config = load_config("config.yml")

    # Configure for comprehensive evaluation
    config["retrieval"] = {
        "type": "dense",  # Start with dense, can test hybrid later
        "top_k": 20,      # Retrieve more documents for better evaluation
        "score_threshold": 0.0  # Don't filter by score, get all results
    }

    config["evaluation"] = {
        "k_values": [1, 3, 5, 10, 20],  # More comprehensive k values
        "metrics": {
            "retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"]
        }
    }

    # Create a comprehensive dataset adapter
    class FullDatasetBenchmarkAdapter(StackOverflowBenchmarkAdapter):
        """Adapter that uses real StackOverflow data for comprehensive evaluation."""
        
        def __init__(self, dataset_path: str, max_queries: int = None):
            super().__init__(dataset_path)
            self.max_queries = max_queries
            
        def load_queries(self, split: str = "test"):
            """Load comprehensive set of queries from question.csv."""
            
            question_file = self.dataset_path / "question.csv"
            print(f"📂 Loading full dataset from {question_file}")
            
            try:
                df = pd.read_csv(question_file)
                print(f"📊 Found {len(df)} questions in dataset")
                
                queries = []
                max_to_load = self.max_queries or len(df)
                
                print(f"🔄 Processing {min(max_to_load, len(df))} questions...")
                
                for idx, row in df.iterrows():
                    if idx >= max_to_load:
                        break
                    
                    # Skip if no title
                    if pd.isna(row['question_title']) or not str(row['question_title']).strip():
                        continue
                    
                    # Skip very short titles (likely low quality)
                    title = str(row['question_title']).strip()
                    if len(title) < 10:
                        continue
                    
                    from benchmark_contracts import BenchmarkQuery
                    
                    # For retrieval evaluation, we'll use the question as query
                    # and see how well the system retrieves related content
                    query = BenchmarkQuery(
                        query_id=f"full_so_{row['question_id']}",
                        query_text=title,
                        expected_answer=str(row['question_body'])[:500] if not pd.isna(row['question_body']) else None,
                        relevant_doc_ids=None,  # We'll evaluate based on relevance scores
                        difficulty=self._assess_difficulty_from_text(title),
                        category=self._extract_category_from_tags(row.get('tags', '')),
                        metadata={
                            "original_question_id": row['question_id'],
                            "question_type": str(row.get('question_type', 'unknown')),
                            "tags": str(row.get('tags', '')),
                            "source": "full_stackoverflow_dataset",
                            "has_body": not pd.isna(row['question_body']),
                            "body_length": len(str(row['question_body'])) if not pd.isna(row['question_body']) else 0
                        }
                    )
                    queries.append(query)
                
                print(f"✅ Loaded {len(queries)} valid queries from full dataset")
                return queries
                
            except Exception as e:
                print(f"❌ Error loading full dataset: {e}")
                return []
        
        def _assess_difficulty_from_text(self, title: str) -> str:
            """Assess difficulty based on title complexity."""
            title_lower = title.lower()
            
            # Complex technical terms suggest harder questions
            complex_terms = ['algorithm', 'optimization', 'performance', 'concurrency', 
                           'threading', 'async', 'architecture', 'design pattern']
            
            # Simple terms suggest easier questions  
            simple_terms = ['how to', 'what is', 'difference between', 'simple', 'basic']
            
            complex_count = sum(1 for term in complex_terms if term in title_lower)
            simple_count = sum(1 for term in simple_terms if term in title_lower)
            
            if complex_count > simple_count:
                return "hard"
            elif simple_count > 0:
                return "easy"
            else:
                return "medium"
        
        def _extract_category_from_tags(self, tags_str: str) -> str:
            """Extract primary category from tags."""
            if pd.isna(tags_str) or not str(tags_str).strip():
                return "programming"
            
            tags_lower = str(tags_str).lower()
            
            # Language categories
            if any(lang in tags_lower for lang in ['python', 'java', 'javascript', 'c#', 'c++']):
                return "programming_language"
            elif any(web in tags_lower for web in ['html', 'css', 'web', 'frontend', 'backend']):
                return "web_development"  
            elif any(data in tags_lower for data in ['sql', 'database', 'data']):
                return "database"
            elif any(mobile in tags_lower for mobile in ['android', 'ios', 'mobile']):
                return "mobile_development"
            else:
                return "programming"

    # Initialize components
    runner = BenchmarkRunner(config)
    
    # Test with different dataset sizes
    test_sizes = [50, 200, 500]  # Start small and increase
    
    all_results = {}
    
    for test_size in test_sizes:
        print(f"\n🧪 Testing with {test_size} queries...")
        
        adapter = FullDatasetBenchmarkAdapter(
            dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data",
            max_queries=test_size
        )
        
        # Run benchmark
        start_time = time.time()
        results = runner.run_benchmark(adapter=adapter)
        end_time = time.time()
        
        # Store results
        all_results[f"{test_size}_queries"] = {
            **results,
            "total_benchmark_time_s": end_time - start_time,
            "queries_per_second": test_size / (end_time - start_time)
        }
        
        # Print summary for this size
        print(f"\n📊 RESULTS FOR {test_size} QUERIES:")
        print(f"Total Time: {end_time - start_time:.1f}s")
        print(f"Avg Time per Query: {results['performance']['avg_retrieval_time_ms']:.2f}ms")
        print(f"Queries/Second: {test_size / (end_time - start_time):.2f}")
        
        print("🎯 Key Metrics:")
        for metric_name in ['precision@5', 'precision@10', 'recall@5', 'mrr']:
            if metric_name in results['metrics']:
                stats = results['metrics'][metric_name]
                print(f"  {metric_name:12}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Final comprehensive summary
    print(f"\n🏆 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 60)
    
    for size, result in all_results.items():
        print(f"\n📊 {size.upper()}:")
        print(f"  Total Time: {result['total_benchmark_time_s']:.1f}s")
        print(f"  Queries/Sec: {result['queries_per_second']:.2f}")
        print(f"  Avg Latency: {result['performance']['avg_retrieval_time_ms']:.2f}ms")
        
        # Show trend in performance
        if 'precision@5' in result['metrics']:
            precision = result['metrics']['precision@5']['mean']
            print(f"  Precision@5: {precision:.3f}")
    
    # Save detailed results
    results_file = Path("benchmark_results_full_dataset.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print(f"🎯 Total documents in Qdrant: 6,210")
    print(f"🔍 Retrieval strategy: {config['retrieval']['type']}")
    print(f"📈 Evaluation complete!")
    
    return all_results


def run_single_size_benchmark(num_queries: int = 100):
    """Run benchmark with specific number of queries."""
    
    print(f"🚀 Running benchmark with {num_queries} queries")
    
    config = load_config("config.yml")
    config["retrieval"] = {
        "type": "dense",
        "top_k": 10,
        "score_threshold": 0.0
    }
    config["evaluation"] = {
        "k_values": [1, 5, 10, 20],
        "metrics": {"retrieval": ["precision@k", "recall@k", "mrr", "ndcg@k"]}
    }
    
    class SingleSizeAdapter(StackOverflowBenchmarkAdapter):
        def load_queries(self, split: str = "test"):
            question_file = self.dataset_path / "question.csv"
            df = pd.read_csv(question_file)
            
            queries = []
            for idx, row in df.iterrows():
                if len(queries) >= num_queries:
                    break
                    
                if pd.isna(row['question_title']) or len(str(row['question_title']).strip()) < 10:
                    continue
                
                from benchmark_contracts import BenchmarkQuery
                query = BenchmarkQuery(
                    query_id=f"so_{row['question_id']}",
                    query_text=str(row['question_title']).strip(),
                    expected_answer=None,
                    relevant_doc_ids=None,
                    difficulty="medium",
                    category="programming"
                )
                queries.append(query)
            
            print(f"✅ Loaded {len(queries)} queries")
            return queries
    
    runner = BenchmarkRunner(config)
    adapter = SingleSizeAdapter("/home/spiros/Desktop/Thesis/datasets/sosum/data")
    
    results = runner.run_benchmark(adapter)
    
    print(f"\n📊 BENCHMARK RESULTS ({num_queries} queries):")
    print(f"Avg Time: {results['performance']['avg_retrieval_time_ms']:.2f}ms")
    for metric in ['precision@5', 'precision@10', 'recall@5', 'mrr']:
        if metric in results['metrics']:
            stats = results['metrics'][metric]
            print(f"{metric:12}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single size benchmark
        num_queries = int(sys.argv[1])
        run_single_size_benchmark(num_queries)
    else:
        # Full progressive benchmark
        run_full_dataset_benchmark()
