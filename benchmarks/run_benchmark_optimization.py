#!/usr/bin/env python3
"""
Simple benchmark runner for easy optimization experiments.
"""

import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from benchmarks.benchmark_optimizer import BenchmarkOptimizer

def main():
    print("🔬 RAG Benchmark Optimizer")
    print("="*50)
    
    optimizer = BenchmarkOptimizer()
    
    print("\nAvailable options:")
    print("1. Run quick test (10 queries)")
    print("2. Run single scenario")
    print("3. Run all scenarios")
    print("4. Compare previous results")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Quick test
        print("\n🚀 Running quick test...")
        config = optimizer.load_benchmark_config("benchmark_scenarios/quick_test.yml")
        result = optimizer.run_optimization_scenario("quick_test", config)
        optimizer._print_scenario_summary("quick_test", result)
        
    elif choice == "2":
        # Single scenario
        print("\nAvailable scenarios:")
        scenarios = [
            "dense_baseline.yml",
            "dense_high_recall.yml", 
            "dense_high_precision.yml",
            "sparse_bm25.yml",
            "hybrid_retrieval.yml",
            "quick_test.yml"
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. {scenario}")
        
        scenario_choice = input("\nEnter scenario number: ").strip()
        try:
            scenario_idx = int(scenario_choice) - 1
            scenario_file = scenarios[scenario_idx]
            
            print(f"\n🚀 Running scenario: {scenario_file}")
            config = optimizer.load_benchmark_config(f"benchmark_scenarios/{scenario_file}")
            result = optimizer.run_optimization_scenario(scenario_file.replace('.yml', ''), config)
            optimizer._print_scenario_summary(scenario_file.replace('.yml', ''), result)
            
        except (ValueError, IndexError):
            print("❌ Invalid scenario choice")
            
    elif choice == "3":
        # All scenarios
        print("\n🚀 Running all scenarios...")
        results = optimizer.run_multiple_scenarios("benchmark_scenarios")
        
        if results:
            optimizer.compare_scenarios()
            optimizer.save_results()
        
    elif choice == "4":
        # Compare results
        print("\n📊 Comparing previous results...")
        try:
            import yaml
            with open('benchmark_optimization_results.yml', 'r') as f:
                data = yaml.safe_load(f)
                optimizer.results_history = data.get('scenarios', [])
            optimizer.compare_scenarios()
        except FileNotFoundError:
            print("❌ No previous results found. Run some benchmarks first!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
