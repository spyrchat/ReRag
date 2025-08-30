#!/usr/bin/env python3
"""
Test all retrievers in the full benchmark pipeline to ensure they work correctly.
"""

import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from config.config_loader import load_config
from benchmarks.benchmarks_runner import BenchmarkRunner
from benchmarks.benchmarks_adapters import FullDatasetAdapter


def test_retriever_in_pipeline(retriever_type: str, max_queries: int = 5):
    """Test a specific retriever in the full benchmark pipeline."""
    print(f"\n🧪 Testing {retriever_type} retriever in full pipeline...")
    
    # Load config and override retriever type
    config = load_config('/home/spiros/Desktop/Thesis/Thesis/config.yml')
    
    # Set the retriever type in multiple places to ensure it's picked up
    config['default_retriever'] = retriever_type
    
    # Also set in retrieval config for consistency
    if 'retrieval' not in config:
        config['retrieval'] = {}
    config['retrieval']['type'] = retriever_type
    
    # Setup adapter
    dataset_path = '/home/spiros/Desktop/Thesis/datasets/sosum/data'
    adapter = FullDatasetAdapter(dataset_path)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(config)
    
    try:
        # Run small benchmark
        results = runner.run_benchmark(adapter=adapter, max_queries=max_queries)
        
        # Print results
        print(f"✅ {retriever_type} pipeline works!")
        print(f"   Queries processed: {results['config']['total_queries']}")
        print(f"   Avg time: {results['performance']['avg_retrieval_time_ms']:.2f}ms")
        
        metrics = results.get('metrics', {})
        for metric_name in ['precision@5', 'recall@5', 'mrr']:
            if metric_name in metrics:
                mean_val = metrics[metric_name]['mean']
                print(f"   {metric_name}: {mean_val:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ {retriever_type} pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all retrievers in the full pipeline."""
    print("🚀 Testing All Retrievers in Full Pipeline")
    print("="*60)
    
    retrievers = ['dense', 'sparse', 'hybrid']
    results = {}
    
    for retriever_type in retrievers:
        success = test_retriever_in_pipeline(retriever_type)
        results[retriever_type] = success
    
    # Summary
    print(f"\n📊 PIPELINE TEST SUMMARY:")
    print("="*30)
    for retriever_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {retriever_type.title()} Retriever: {status}")
    
    all_pass = all(results.values())
    if all_pass:
        print(f"\n🎉 All retrievers work in the full pipeline!")
        print(f"✅ Ready for optimization experiments!")
    else:
        print(f"\n⚠️  Some retrievers failed. Check logs above.")


if __name__ == "__main__":
    main()
