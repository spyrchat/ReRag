"""
Flexible benchmark runner with configurable optimization parameters.
Supports multiple benchmark scenarios for hyperparameter optimization.
"""

from config.config_loader import load_config
from benchmarks.benchmark_contracts import BenchmarkQuery
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter, FullDatasetAdapter
from benchmarks.benchmarks_runner import BenchmarkRunner
import sys
import os
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')


class BenchmarkOptimizer:
    """Flexible benchmark runner for optimization experiments."""

    def __init__(self, base_config_path: str = "config.yml"):
        """Initialize with base configuration."""
        self.base_config = load_config(base_config_path)
        self.results_history = []

    def load_benchmark_config(self, benchmark_config_path: str) -> Dict[str, Any]:
        """Load benchmark-specific configuration."""
        with open(benchmark_config_path, 'r') as f:
            benchmark_config = yaml.safe_load(f)

        # Merge with base config
        config = self.base_config.copy()
        config.update(benchmark_config)

        return config

    def run_optimization_scenario(self, scenario_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single optimization scenario."""
        print(f"\n🚀 Running optimization scenario: {scenario_name}")
        print(f"📊 Config: {config.get('description', 'No description')}")

        # Setup benchmark runner
        runner = BenchmarkRunner(config)

        # Setup data adapter based on config
        dataset_config = config.get('dataset', {})
        dataset_path = dataset_config.get(
            'path', '/home/spiros/Desktop/Thesis/datasets/sosum/data')

        if dataset_config.get('use_ground_truth', True):
            # Use FullDatasetAdapter for ground truth evaluation
            adapter = FullDatasetAdapter(dataset_path)
        else:
            # Use custom adapter for real questions without ground truth
            adapter = self._create_real_data_adapter(dataset_path)

        # Run benchmark
        print(f"📈 Running with max_queries: {config.get('max_queries', 10)}")
        results = runner.run_benchmark(
            adapter=adapter,
            max_queries=config.get('max_queries', 10)
        )

        # Add scenario metadata
        results['scenario_name'] = scenario_name
        results['scenario_config'] = config

        # Store results
        self.results_history.append(results)

        return results

    def _create_real_data_adapter(self, dataset_path: str):
        """Create adapter for real data without ground truth."""
        class RealDataAdapter(StackOverflowBenchmarkAdapter):
            def load_queries(self, split: str = "test"):
                import pandas as pd

                question_file = Path(dataset_path) / "question.csv"
                df = pd.read_csv(question_file)

                queries = []
                for idx, row in df.iterrows():
                    if idx >= 50:  # Limit for testing
                        break

                    if pd.isna(row['question_title']):
                        continue

                    query = BenchmarkQuery(
                        query_id=f"real_so_{row['question_id']}",
                        query_text=str(row['question_title']),
                        expected_answer=None,
                        relevant_doc_ids=None,  # No ground truth
                        difficulty="medium",
                        category="programming",
                        metadata={"source": "real_stackoverflow"}
                    )
                    queries.append(query)

                return queries

        return RealDataAdapter(dataset_path)

    def run_multiple_scenarios(self, scenarios_dir: str = "benchmark_scenarios") -> List[Dict[str, Any]]:
        """Run multiple optimization scenarios from a directory."""
        scenarios_path = Path(scenarios_dir)
        if not scenarios_path.exists():
            print(f"❌ Scenarios directory not found: {scenarios_path}")
            return []

        results = []
        for scenario_file in scenarios_path.glob("*.yml"):
            scenario_name = scenario_file.stem
            config = self.load_benchmark_config(str(scenario_file))

            try:
                result = self.run_optimization_scenario(scenario_name, config)
                results.append(result)

                # Print quick summary
                self._print_scenario_summary(scenario_name, result)

            except Exception as e:
                print(f"❌ Failed scenario {scenario_name}: {e}")
                continue

        return results

    def _print_scenario_summary(self, scenario_name: str, results: Dict[str, Any]):
        """Print a quick summary of scenario results."""
        print(f"\n📊 {scenario_name} Results:")
        print(f"   Queries: {results['config']['total_queries']}")
        print(
            f"   Avg Time: {results['performance']['avg_retrieval_time_ms']:.2f}ms")

        # Print key metrics
        metrics = results.get('metrics', {})
        for metric_name in ['precision@5', 'recall@5', 'mrr']:
            if metric_name in metrics:
                mean_val = metrics[metric_name]['mean']
                print(f"   {metric_name}: {mean_val:.3f}")

    def compare_scenarios(self) -> Dict[str, Any]:
        """Compare all run scenarios."""
        if not self.results_history:
            print("❌ No scenarios run yet")
            return {}

        print(
            f"\n🔬 OPTIMIZATION COMPARISON ({len(self.results_history)} scenarios)")
        print("="*80)

        comparison = {
            'scenarios': [],
            'best_precision': {'scenario': None, 'value': 0},
            'best_recall': {'scenario': None, 'value': 0},
            'best_mrr': {'scenario': None, 'value': 0},
            'fastest': {'scenario': None, 'time': float('inf')}
        }

        for result in self.results_history:
            scenario_name = result['scenario_name']
            metrics = result.get('metrics', {})
            avg_time = result['performance']['avg_retrieval_time_ms']

            scenario_summary = {
                'name': scenario_name,
                'precision@5': metrics.get('precision@5', {}).get('mean', 0),
                'recall@5': metrics.get('recall@5', {}).get('mean', 0),
                'mrr': metrics.get('mrr', {}).get('mean', 0),
                'avg_time_ms': avg_time,
                'config': result['scenario_config']
            }

            comparison['scenarios'].append(scenario_summary)

            # Track best performers
            if scenario_summary['precision@5'] > comparison['best_precision']['value']:
                comparison['best_precision'] = {
                    'scenario': scenario_name, 'value': scenario_summary['precision@5']}

            if scenario_summary['recall@5'] > comparison['best_recall']['value']:
                comparison['best_recall'] = {
                    'scenario': scenario_name, 'value': scenario_summary['recall@5']}

            if scenario_summary['mrr'] > comparison['best_mrr']['value']:
                comparison['best_mrr'] = {
                    'scenario': scenario_name, 'value': scenario_summary['mrr']}

            if avg_time < comparison['fastest']['time']:
                comparison['fastest'] = {
                    'scenario': scenario_name, 'time': avg_time}

            # Print scenario details
            print(f"📋 {scenario_name}:")
            print(f"   Precision@5: {scenario_summary['precision@5']:.3f}")
            print(f"   Recall@5: {scenario_summary['recall@5']:.3f}")
            print(f"   MRR: {scenario_summary['mrr']:.3f}")
            print(f"   Avg Time: {avg_time:.2f}ms")
            print(
                f"   Config: {result['scenario_config'].get('description', 'N/A')}")
            print()

        # Print best performers
        print(f"🏆 BEST PERFORMERS:")
        print(
            f"   Best Precision@5: {comparison['best_precision']['scenario']} ({comparison['best_precision']['value']:.3f})")
        print(
            f"   Best Recall@5: {comparison['best_recall']['scenario']} ({comparison['best_recall']['value']:.3f})")
        print(
            f"   Best MRR: {comparison['best_mrr']['scenario']} ({comparison['best_mrr']['value']:.3f})")
        print(
            f"   Fastest: {comparison['fastest']['scenario']} ({comparison['fastest']['time']:.2f}ms)")

        return comparison

    def save_results(self, output_file: str = "benchmark_optimization_results.csv"):
        """Save all results to a CSV file."""
        if not self.results_history:
            print("❌ No results to save")
            return

        # Prepare data for CSV
        csv_data = []
        for result in self.results_history:
            scenario_name = result['scenario_name']
            config = result.get('scenario_config', {})
            metrics = result.get('metrics', {})
            performance = result.get('performance', {})

            row = {
                'scenario_name': scenario_name,
                'description': config.get('description', 'N/A'),
                'default_retriever': config.get('default_retriever', 'N/A'),
                'max_queries': config.get('max_queries', 0),
                'total_queries': result.get('config', {}).get('total_queries', 0),
                'avg_time_ms': performance.get('avg_retrieval_time_ms', 0),
                'min_time_ms': performance.get('min_retrieval_time_ms', 0),
                'max_time_ms': performance.get('max_retrieval_time_ms', 0),
                'precision@1_mean': metrics.get('precision@1', {}).get('mean', 0),
                'precision@1_std': metrics.get('precision@1', {}).get('std', 0),
                'precision@5_mean': metrics.get('precision@5', {}).get('mean', 0),
                'precision@5_std': metrics.get('precision@5', {}).get('std', 0),
                'precision@10_mean': metrics.get('precision@10', {}).get('mean', 0),
                'precision@10_std': metrics.get('precision@10', {}).get('std', 0),
                'recall@1_mean': metrics.get('recall@1', {}).get('mean', 0),
                'recall@1_std': metrics.get('recall@1', {}).get('std', 0),
                'recall@5_mean': metrics.get('recall@5', {}).get('mean', 0),
                'recall@5_std': metrics.get('recall@5', {}).get('std', 0),
                'recall@10_mean': metrics.get('recall@10', {}).get('mean', 0),
                'recall@10_std': metrics.get('recall@10', {}).get('std', 0),
                'mrr_mean': metrics.get('mrr', {}).get('mean', 0),
                'mrr_std': metrics.get('mrr', {}).get('std', 0),
                'ndcg@5_mean': metrics.get('ndcg@5', {}).get('mean', 0),
                'ndcg@5_std': metrics.get('ndcg@5', {}).get('std', 0),
                'ndcg@10_mean': metrics.get('ndcg@10', {}).get('mean', 0),
                'ndcg@10_std': metrics.get('ndcg@10', {}).get('std', 0),
            }

            # Add configuration details
            retrieval_config = config.get('retrieval', {})
            row['top_k'] = retrieval_config.get('top_k', 'N/A')
            row['score_threshold'] = retrieval_config.get(
                'score_threshold', 'N/A')

            # Add embedding details
            embedding_config = config.get('embedding', {})
            if isinstance(embedding_config, dict):
                row['embedding_provider'] = embedding_config.get(
                    'provider', 'N/A')
                row['embedding_model'] = embedding_config.get('model', 'N/A')
            else:
                row['embedding_provider'] = 'N/A'
                row['embedding_model'] = 'N/A'

            csv_data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)

        print(f"💾 Results saved to {output_file}")
        print(
            f"📊 Saved {len(csv_data)} scenarios with {len(df.columns)} columns")

        # Also save a summary CSV with just key metrics
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_columns = [
            'scenario_name', 'description', 'default_retriever', 'total_queries',
            'avg_time_ms', 'precision@5_mean', 'recall@5_mean', 'mrr_mean'
        ]
        summary_df = df[summary_columns]
        summary_df.to_csv(summary_file, index=False)
        print(f"📋 Summary saved to {summary_file}")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Run benchmark optimization scenarios")
    parser.add_argument('--scenario', type=str,
                        help='Single scenario config file')
    parser.add_argument('--scenarios-dir', type=str, default='benchmark_scenarios',
                        help='Directory containing scenario configs')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only compare existing results')

    args = parser.parse_args()

    optimizer = BenchmarkOptimizer()

    if args.compare_only:
        # Load existing results if available
        try:
            # Try to load from CSV first, then fallback to YAML
            if os.path.exists('benchmark_optimization_results.csv'):
                df = pd.read_csv('benchmark_optimization_results.csv')
                # Convert CSV back to results format for comparison
                optimizer.results_history = []
                for _, row in df.iterrows():
                    result = {
                        'scenario_name': row['scenario_name'],
                        'scenario_config': {
                            'description': row['description'],
                            'default_retriever': row['default_retriever'],
                            'max_queries': row['max_queries']
                        },
                        'config': {'total_queries': row['total_queries']},
                        'performance': {'avg_retrieval_time_ms': row['avg_time_ms']},
                        'metrics': {
                            'precision@5': {'mean': row['precision@5_mean']},
                            'recall@5': {'mean': row['recall@5_mean']},
                            'mrr': {'mean': row['mrr_mean']}
                        }
                    }
                    optimizer.results_history.append(result)
            else:
                # Fallback to YAML format
                with open('benchmark_optimization_results.yml', 'r') as f:
                    data = yaml.safe_load(f)
                    optimizer.results_history = data.get('scenarios', [])
            optimizer.compare_scenarios()
        except FileNotFoundError:
            print("❌ No existing results found (searched for .csv and .yml)")
        return

    if args.scenario:
        # Run single scenario
        config = optimizer.load_benchmark_config(args.scenario)
        result = optimizer.run_optimization_scenario(
            Path(args.scenario).stem, config)
        optimizer._print_scenario_summary(Path(args.scenario).stem, result)
    else:
        # Run multiple scenarios
        results = optimizer.run_multiple_scenarios(args.scenarios_dir)

        if results:
            # Compare all scenarios
            optimizer.compare_scenarios()

            # Save results
            optimizer.save_results()


if __name__ == "__main__":
    main()
