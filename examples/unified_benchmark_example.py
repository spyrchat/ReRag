#!/usr/bin/env python3
"""
Unified configuration benchmark example.
Demonstrates how to run benchmarks with the new unified config system.
"""

import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from config.config_loader import load_config, load_config_with_overrides
from benchmarks.benchmarks_runner import BenchmarkRunner
from benchmarks.benchmark_contracts import BenchmarkAdapter, BenchmarkQuery
from pathlib import Path


class StackOverflowTestAdapter(BenchmarkAdapter):
    """Simple test adapter for demonstrating the benchmark system."""
    
    @property
    def name(self) -> str:
        return "StackOverflow-Test"
    
    @property 
    def tasks(self):
        """Supported benchmark tasks."""
        from benchmarks.benchmark_contracts import BenchmarkTask
        return [BenchmarkTask.RETRIEVAL]
    
    def get_ground_truth(self, query_id: str):
        """Get ground truth for evaluation."""
        return {"relevant_docs": [], "expected_answer": None}
    
    def load_queries(self, split: str = "test"):
        """Load a few sample queries for testing."""        
        return [
            BenchmarkQuery(
                query_id="test_1",
                query_text="How to use Python pandas for data analysis?",
                relevant_doc_ids=["123", "456"],
                expected_answer="Pandas is a powerful library for data manipulation..."
            ),
            BenchmarkQuery(
                query_id="test_2", 
                query_text="What is the difference between list and tuple in Python?",
                relevant_doc_ids=["789", "101"],
                expected_answer="Lists are mutable while tuples are immutable..."
            ),
            BenchmarkQuery(
                query_id="test_3",
                query_text="How to handle exceptions in Python?",
                relevant_doc_ids=["111", "222"], 
                expected_answer="Use try-except blocks to handle exceptions..."
            )
        ]


def main():
    """Run benchmark with unified configuration."""
    
    print("🚀 Unified Configuration Benchmark Example")
    print("=" * 50)
    
    # Load base configuration
    config = load_config("/home/spiros/Desktop/Thesis/Thesis/config.yml")
    print(f"✅ Loaded base configuration")
    
    # Test with different strategies
    strategies = ["dense", "hybrid"]  # Skip sparse for now due to collection issues
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy} retrieval strategy...")
        
        # Create strategy-specific overrides
        overrides = {
            "benchmark": {
                "retrieval": {
                    "strategy": strategy,
                    "top_k": 10
                }
            }
        }
        
        # Merge configuration with overrides
        strategy_config = load_config_with_overrides(
            "/home/spiros/Desktop/Thesis/Thesis/config.yml", 
            overrides
        )
        
        try:
            # Initialize benchmark runner
            runner = BenchmarkRunner(strategy_config)
            print(f"  ✅ Initialized {strategy} benchmark runner")
            
            # Create test adapter
            adapter = StackOverflowTestAdapter()
            
            # Run benchmark with limited queries
            results = runner.run_benchmark(adapter, max_queries=3)
            
            # Display results
            print(f"  📊 Results for {strategy} strategy:")
            print(f"    - Dataset: {results['dataset']}")
            print(f"    - Strategy: {results['config']['retrieval_strategy']}")
            print(f"    - Total queries: {results['config']['total_queries']}")
            
            if 'performance' in results:
                perf = results['performance']
                print(f"    - Avg retrieval time: {perf['avg_retrieval_time_ms']:.2f}ms")
                
            if 'metrics' in results and results['metrics']:
                print(f"    - Available metrics: {list(results['metrics'].keys())}")
            else:
                print(f"    - No metrics computed (expected for test queries)")
                
        except Exception as e:
            print(f"  ❌ Error with {strategy} strategy: {e}")
    
    print(f"\n🎉 Benchmark example completed!")


if __name__ == "__main__":
    main()
