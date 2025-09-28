"""
Experiment 1: Clean experiment runner with separated concerns.
"""


from utils import calculate_confidence_intervals
import sys
import yaml
from pathlib import Path
from datetime import datetime
import argparse
# Add project root to Python path first
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Experiment1Runner:
    """Clean experiment runner focused only on orchestration."""

    def __init__(self, output_dir: str = "results/experiment_1", test_mode: bool = False):
        from benchmarks.utils import calculate_confidence_intervals
        from benchmarks.report_generator import BenchmarkReportGenerator
        from benchmarks.results_exporter import BenchmarkResultsExporter
        from benchmarks.statistical_analyzer import BenchmarkStatisticalAnalyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        self.results = {}

        # Initialize components
        self.statistical_analyzer = BenchmarkStatisticalAnalyzer()
        self.results_exporter = BenchmarkResultsExporter(
            self.output_dir, test_mode)
        self.report_generator = BenchmarkReportGenerator(test_mode)

    def run_experiment(self):
        """Run the complete experiment."""
        scenarios = self._get_scenarios()

        self._print_experiment_header()

        # Run all scenarios
        for scenario in scenarios:
            if self._check_scenario_exists(scenario):
                result = self._run_single_scenario(scenario)
                if result:
                    self.results[scenario['name']] = result
                    self.report_generator.print_scenario_summary(
                        scenario['name'], result)

        if not self.results:
            print("❌ No scenarios completed successfully")
            return

        # Analyze and export results
        self._post_process_results()

    def _get_scenarios(self):
        """Define experiment scenarios."""
        return [
            {'path': 'benchmark_scenarios/experiment_1/bm25_baseline.yml',
                'name': 'BM25_Baseline'},
            {'path': 'benchmark_scenarios/experiment_1/splade_baseline.yml',
                'name': 'SPLADE_Baseline'},
            {'path': 'benchmark_scenarios/experiment_1/dense_bge_m3.yml',
                'name': 'Dense_BGE_M3'},
            {'path': 'benchmark_scenarios/experiment_1/hybrid_splade_bge_m3.yml',
                'name': 'Hybrid_SPLADE_BGE_M3'},
            {'path': 'benchmark_scenarios/experiment_1/hybrid_bm25_bge_m3.yml',
                'name': 'Hybrid_BM25_BGE_M3'}
        ]

    def _run_single_scenario(self, scenario):
        """Run a single benchmark scenario."""
        try:
            # Load and modify config
            with open(scenario['path'], 'r') as f:
                config = yaml.safe_load(f)

            if self.test_mode:
                config['max_queries'] = 10

            # Initialize and run benchmark
            qdrant_cfg = config['retrieval']['qdrant']
            from qdrant_client import QdrantClient
            from benchmarks.benchmarks_runner import BenchmarkRunner  # <-- ensure import here
            from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
            qdrant_client = QdrantClient(
                host=qdrant_cfg.get('host', 'localhost'),
                port=qdrant_cfg.get('port', 6333)
            )

            runner = BenchmarkRunner(config)
            adapter = StackOverflowBenchmarkAdapter(
                dataset_path=config['dataset']['path'],
                qdrant_client=qdrant_client,
                collection_name=qdrant_cfg['collection_name']
            )

            results = runner.run_benchmark(
                adapter=adapter, max_queries=config['max_queries'])

            # Add metadata
            results.update({
                'scenario_name': scenario['name'],
                'scenario_config': config,
                'timestamp': datetime.now().isoformat(),
                'test_mode': self.test_mode
            })

            return results

        except Exception as e:
            print(f"❌ Failed to run {scenario['name']}: {e}")
            return None

    def _post_process_results(self):
        """Analyze and export all results."""
        # Enhance with confidence intervals (only for full mode)
        if not self.test_mode:
            self._enhance_with_confidence_intervals()

        # Statistical analysis (only for full mode with multiple scenarios)
        statistical_results = None
        if not self.test_mode and len(self.results) >= 2:
            statistical_results = self.statistical_analyzer.analyze_results(
                self.results)
            self.report_generator.print_statistical_report(statistical_results)

        # Export all results
        self.results_exporter.export_all_results(
            self.results, statistical_results)

    def _print_experiment_header(self):
        """Print experiment header."""
        mode_title = "TEST RUN" if self.test_mode else "FULL EXPERIMENT"
        query_count = "10 queries" if self.test_mode else "506 queries"

        print(
            f"🧪 EXPERIMENT 1 - {mode_title}: Complete Retrieval Method Comparison")
        print("=" * 70)
        print("📋 Configuration:")
        print("   • Methods: BM25, SPLADE, Dense BGE-M3, Hybrid combinations")
        print(f"   • Queries: {query_count}")
        print("   • top_k: 20")
        print("   • Metrics: Precision@K, Recall@K, MRR, NDCG@K, F1@K, MAP")
        if not self.test_mode:
            print("   • Statistics: Mean, STD, 95% CI, Statistical significance")
        print("=" * 70)

    def _check_scenario_exists(self, scenario):
        """Check if scenario file exists."""
        scenario_path = Path(scenario['path'])
        if not scenario_path.exists():
            print(f"❌ Missing scenario file: {scenario['path']}")
            return False
        return True

    def _enhance_with_confidence_intervals(self):
        """Add confidence intervals to metrics."""
        print("\n📊 Calculating confidence intervals...")

        for scenario_name, result in self.results.items():
            metrics = result.get('metrics', {})
            enhanced_metrics = {}

            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'scores' in metric_data:
                    scores = metric_data['scores']
                    enhanced_stats = calculate_confidence_intervals(scores)
                    enhanced_metrics[metric_name] = enhanced_stats
                else:
                    enhanced_metrics[metric_name] = metric_data

            self.results[scenario_name]['metrics'] = enhanced_metrics

    # ... other helper methods (print_header, check_scenarios, etc.)


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument(
        '--output-dir', default='results/experiment_1', help='Output directory')

    args = parser.parse_args()

    runner = Experiment1Runner(output_dir=args.output_dir, test_mode=args.test)
    runner.run_experiment()


if __name__ == "__main__":
    main()
