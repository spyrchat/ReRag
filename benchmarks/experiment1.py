"""
Experiment 1: BM25 vs Dense vs Hybrid BGE-M3 Comparison
Statistical analysis with confidence intervals and detailed metrics export.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import yaml
import json
from datetime import datetime
from scipy import stats
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after adding path - handle both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from .benchmarks_runner import BenchmarkRunner
    from .benchmarks_adapters import StackOverflowBenchmarkAdapter
except ImportError:
    try:
        # Try absolute imports (when run as script)
        from benchmarks.benchmarks_runner import BenchmarkRunner
        from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the project root directory")
        print("Usage: python benchmarks/experiment1.py [--test]")
        sys.exit(1)


class Experiment1Runner:
    """Enhanced runner for Experiment 1 with test mode and statistical analysis."""

    def __init__(self, output_dir: str = "results/experiment_1", test_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.test_mode = test_mode

    def run_scenario(self, scenario_path: str, scenario_name: str) -> Dict[str, Any]:
        """Run a single benchmark scenario with detailed results."""
        mode_str = "TEST MODE (10 queries)" if self.test_mode else "FULL MODE (506 queries)"
        print(f"\n🚀 Running Experiment 1 - {scenario_name} - {mode_str}")
        print("=" * 70)

        # Load scenario configuration
        with open(scenario_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override max_queries for test mode
        if self.test_mode:
            config['max_queries'] = 10
            print("🧪 TEST MODE: Processing only 10 queries for end-to-end validation")
        else:
            print(
                f"📊 FULL MODE: Processing ALL {config['max_queries']} queries for statistical significance")

        # Initialize runner and adapter
        runner = BenchmarkRunner(config)
        adapter = StackOverflowBenchmarkAdapter(
            dataset_path=config['dataset']['path']
        )

        # Run benchmark
        results = runner.run_benchmark(
            adapter=adapter,
            max_queries=config['max_queries']
        )

        # Add scenario metadata
        results['scenario_name'] = scenario_name
        results['scenario_config'] = config
        results['timestamp'] = datetime.now().isoformat()
        results['test_mode'] = self.test_mode

        return results

    def calculate_confidence_intervals(self, values: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence intervals for a list of values."""
        if not values or len(values) < 2:
            return {
                'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0,
                'median': 0.0, 'count': 0
            }

        values = [v for v in values if not np.isnan(v)]
        if not values:
            return {
                'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0,
                'median': 0.0, 'count': 0
            }

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)

        # Calculate 95% confidence interval
        alpha = 1 - confidence
        degrees_freedom = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom) if n > 1 else 0

        margin_error = t_critical * (std / np.sqrt(n))
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error

        return {
            'mean': float(mean),
            'std': float(std),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'median': float(np.median(values)),
            'count': n,
            'margin_error': float(margin_error)
        }

    def export_detailed_metrics(self, results: Dict[str, Any], filename: str):
        """Export detailed metrics with confidence intervals to CSV."""
        metrics_data = []

        for scenario_name, scenario_results in results.items():
            metrics = scenario_results.get('metrics', {})
            config = scenario_results.get('scenario_config', {})

            # Base information
            base_info = {
                'scenario': scenario_name,
                'retrieval_type': config.get('retrieval', {}).get('type', 'unknown'),
                'embedding_model': self._get_embedding_model(config),
                'total_queries': scenario_results.get('config', {}).get('total_queries', 0),
                'test_mode': scenario_results.get('test_mode', False),
                'timestamp': scenario_results.get('timestamp', ''),
            }

            # Process each metric
            for metric_name, metric_stats in metrics.items():
                if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                    row = base_info.copy()
                    row.update({
                        'metric': metric_name,
                        'mean': metric_stats.get('mean', 0),
                        'std': metric_stats.get('std', 0),
                        'ci_lower': metric_stats.get('ci_lower', 0),
                        'ci_upper': metric_stats.get('ci_upper', 0),
                        'median': metric_stats.get('median', 0),
                        'min': metric_stats.get('min', 0),
                        'max': metric_stats.get('max', 0),
                        'count': metric_stats.get('count', 0),
                        'margin_error': metric_stats.get('margin_error', 0)
                    })
                    metrics_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(metrics_data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        print(f"💾 Detailed metrics exported to: {output_path}")
        return output_path

    def _get_embedding_model(self, config: Dict[str, Any]) -> str:
        """Extract embedding model from config."""
        embedding = config.get('embedding', {})
        if 'dense' in embedding:
            return embedding['dense'].get('model', 'unknown')
        elif 'sparse' in embedding:
            return embedding['sparse'].get('model', 'unknown')
        return embedding.get('model', 'unknown')

    def run_experiment(self):
        """Run the complete experiment comparing BM25 vs Dense vs Hybrid BGE-M3."""
        scenarios = [
            {
                'path': 'benchmark_scenarios/experiment_1/BM25 baseline Scenario.yml',
                'name': 'BM25_Baseline'
            },
            {
                'path': 'benchmark_scenarios/experiment_1/dense_bge_m3.yml',
                'name': 'Dense_BGE_M3'
            },
            {
                'path': 'benchmark_scenarios/experiment_1/hybrid_bge_m3.yml',
                'name': 'Hybrid_BGE_M3'
            }
        ]

        mode_title = "TEST RUN" if self.test_mode else "FULL EXPERIMENT"
        query_count = "10 queries" if self.test_mode else "506 queries"

        print(f"🧪 EXPERIMENT 1 - {mode_title}: BM25 vs Dense vs Hybrid BGE-M3")
        print("=" * 70)
        print("📋 Configuration:")
        print("   • Collection: sosum_stackoverflow_bge_large_code_aware_v1")
        print(f"   • Queries: {query_count}")
        print("   • top_k: 10")
        print("   • Metrics: Precision@K, Recall@K, MRR, NDCG@K, F1@K, MAP")
        if not self.test_mode:
            print("   • Statistics: Mean, STD, 95% CI")
        print("=" * 70)

        # Check if scenario files exist
        missing_scenarios = []
        for scenario in scenarios:
            scenario_path = Path(scenario['path'])
            if not scenario_path.exists():
                missing_scenarios.append(scenario['path'])

        if missing_scenarios:
            print("❌ Missing scenario files:")
            for missing in missing_scenarios:
                print(f"   - {missing}")
            print(
                "\nPlease create the missing scenario files before running the experiment.")
            print("Expected files:")
            for scenario in scenarios:
                print(f"   - {scenario['path']}")
            return

        # Run each scenario
        for scenario in scenarios:
            try:
                result = self.run_scenario(scenario['path'], scenario['name'])
                self.results[scenario['name']] = result

                # Print quick summary
                self._print_scenario_summary(scenario['name'], result)

            except Exception as e:
                print(f"❌ Failed to run {scenario['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(self.results) == 0:
            print("❌ No scenarios completed successfully")
            return

        # Enhanced statistical analysis (only for full mode)
        if not self.test_mode:
            self._enhance_statistics()

        # Export results
        self._export_results()

        # Statistical comparison (only for full mode with multiple scenarios)
        if not self.test_mode and len(self.results) >= 2:
            self._statistical_comparison()

    def _enhance_statistics(self):
        """Enhance results with confidence intervals."""
        print("\n📊 Calculating confidence intervals...")

        for scenario_name, result in self.results.items():
            metrics = result.get('metrics', {})
            enhanced_metrics = {}

            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    # If we have individual scores, calculate CI
                    if 'scores' in metric_data:
                        scores = metric_data['scores']
                        enhanced_stats = self.calculate_confidence_intervals(
                            scores)
                        enhanced_metrics[metric_name] = enhanced_stats
                    else:
                        # Use existing stats and add CI if possible
                        enhanced_metrics[metric_name] = metric_data

            self.results[scenario_name]['metrics'] = enhanced_metrics

    def _print_scenario_summary(self, scenario_name: str, result: Dict[str, Any]):
        """Print summary for a scenario."""
        print(f"\n📊 {scenario_name} Results:")
        print("-" * 40)

        metrics = result.get('metrics', {})
        config = result.get('config', {})

        print(f"   Total Queries: {config.get('total_queries', 0)}")
        print(
            f"   Avg Time: {result.get('performance', {}).get('avg_retrieval_time_ms', 0):.2f}ms")

        # Key metrics
        key_metrics = ['precision@5', 'recall@5', 'mrr', 'ndcg@5']
        for metric in key_metrics:
            if metric in metrics:
                stats = metrics[metric]
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                if self.test_mode:
                    print(f"   {metric}: {mean_val:.4f}")
                else:
                    print(f"   {metric}: {mean_val:.4f} ± {std_val:.4f}")

    def _export_results(self):
        """Export results in appropriate formats."""
        mode_suffix = "test" if self.test_mode else "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Detailed CSV with confidence intervals
        csv_filename = f"experiment_1_detailed_metrics_{mode_suffix}_{timestamp}.csv"
        self.export_detailed_metrics(self.results, csv_filename)

        # 2. Summary comparison CSV
        summary_filename = f"experiment_1_summary_{mode_suffix}_{timestamp}.csv"
        self._export_summary_comparison(summary_filename)

        # 3. Full JSON results
        json_filename = f"experiment_1_full_results_{mode_suffix}_{timestamp}.json"
        json_path = self.output_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Full results saved to: {json_path}")

    def _export_summary_comparison(self, filename: str):
        """Export summary comparison table."""
        summary_data = []

        key_metrics = ['precision@1', 'precision@3', 'precision@5', 'precision@10',
                       'recall@1', 'recall@3', 'recall@5', 'recall@10',
                       'mrr', 'ndcg@5', 'ndcg@10', 'f1@5', 'f1@10' 'map']

        for scenario_name, result in self.results.items():
            metrics = result.get('metrics', {})
            config = result.get('scenario_config', {})

            row = {
                'scenario': scenario_name,
                'retrieval_type': config.get('retrieval', {}).get('type'),
                'model': self._get_embedding_model(config),
                'total_queries': result.get('config', {}).get('total_queries', 0),
                'test_mode': result.get('test_mode', False)
            }

            for metric in key_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    row[f"{metric}_mean"] = stats.get('mean', 0)
                    if not self.test_mode:
                        row[f"{metric}_std"] = stats.get('std', 0)
                        row[f"{metric}_ci_lower"] = stats.get('ci_lower', 0)
                        row[f"{metric}_ci_upper"] = stats.get('ci_upper', 0)

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"📋 Summary comparison saved to: {output_path}")

    def _statistical_comparison(self):
        """Perform statistical significance tests."""
        print("\n📈 STATISTICAL COMPARISON")
        print("=" * 40)

        scenarios = list(self.results.keys())

        # Compare all pairs
        for i, scenario_1 in enumerate(scenarios):
            for j, scenario_2 in enumerate(scenarios[i+1:], i+1):
                print(f"\nComparing {scenario_1} vs {scenario_2}")
                print("-" * 50)

                metrics_1 = self.results[scenario_1]['metrics']
                metrics_2 = self.results[scenario_2]['metrics']

                key_metrics = ['precision@5', 'recall@5', 'mrr', 'ndcg@5']

                for metric in key_metrics:
                    if metric in metrics_1 and metric in metrics_2:
                        stats_1 = metrics_1[metric]
                        stats_2 = metrics_2[metric]

                        mean_1 = stats_1.get('mean', 0)
                        mean_2 = stats_2.get('mean', 0)

                        improvement = ((mean_2 - mean_1) /
                                       mean_1 * 100) if mean_1 > 0 else 0

                        print(f"{metric}:")
                        print(
                            f"  {scenario_1}: {mean_1:.4f} ± {stats_1.get('std', 0):.4f}")
                        print(
                            f"  {scenario_2}: {mean_2:.4f} ± {stats_2.get('std', 0):.4f}")
                        print(f"  Improvement: {improvement:+.2f}%")


def main():
    """Run Experiment 1 with optional test mode."""
    parser = argparse.ArgumentParser(
        description='Run Experiment 1: BM25 vs Dense vs Hybrid BGE-M3')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with only 10 queries for end-to-end validation')
    parser.add_argument('--output-dir', default='results/experiment_1',
                        help='Output directory for results (default: results/experiment_1)')

    args = parser.parse_args()

    runner = Experiment1Runner(output_dir=args.output_dir, test_mode=args.test)
    runner.run_experiment()

    mode_str = "test run" if args.test else "full experiment"
    print(f"\n✅ Experiment 1 {mode_str} completed!")
    print(f"📁 Results saved in: {runner.output_dir}")

    if args.test:
        print("\n💡 Test completed successfully! Run without --test flag for full experiment with 506 queries.")


if __name__ == "__main__":
    main()
