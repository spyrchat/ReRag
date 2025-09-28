"""Export functionality for benchmark results."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from .utils import export_detailed_metrics, export_per_query_results, get_embedding_model


class BenchmarkResultsExporter:
    """Handles all export functionality for benchmark results."""

    def __init__(self, output_dir: Path, test_mode: bool = False):
        self.output_dir = output_dir
        self.test_mode = test_mode

    def export_all_results(self, results: Dict[str, Any], statistical_results: List = None):
        """Export all result formats."""
        mode_suffix = "test" if self.test_mode else "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Detailed metrics CSV
        csv_filename = f"experiment_1_detailed_metrics_{mode_suffix}_{timestamp}.csv"
        export_detailed_metrics(results, csv_filename, self.output_dir)

        # 2. Summary comparison CSV
        summary_filename = f"experiment_1_summary_{mode_suffix}_{timestamp}.csv"
        self._export_summary_comparison(results, summary_filename)

        # 3. Per-query results CSV
        per_query_filename = f"experiment_1_per_query_{mode_suffix}_{timestamp}.csv"
        export_per_query_results(results, per_query_filename, self.output_dir)

        # 4. Full JSON results
        json_filename = f"experiment_1_full_results_{mode_suffix}_{timestamp}.json"
        self._export_json_results(results, json_filename)

        # 5. Statistical analysis CSV (if available)
        if statistical_results:
            stats_filename = f"experiment_1_statistical_analysis_{mode_suffix}_{timestamp}.csv"
            self._export_statistical_results(
                statistical_results, stats_filename)

    def _export_summary_comparison(self, results: Dict[str, Any], filename: str):
        """Export summary comparison table."""
        summary_data = []
        key_metrics = ['precision@1', 'precision@3', 'precision@5',
                       'recall@1', 'recall@3', 'recall@5',
                       'mrr', 'ndcg@1', 'ndcg@3', 'ndcg@5', 'f1@3', 'f1@5', 'map']

        for scenario_name, result in results.items():
            metrics = result.get('metrics', {})
            config = result.get('scenario_config', {})
            performance = result.get('performance', {})

            row = {
                'scenario': scenario_name,
                'retrieval_type': config.get('retrieval', {}).get('type'),
                'model': get_embedding_model(config),
                'total_queries': result.get('config', {}).get('total_queries', 0),
                'test_mode': result.get('test_mode', False),
                'time_mean_ms': performance.get('mean', 0),
                'time_std_ms': performance.get('std', 0),
                'time_median_ms': performance.get('median', 0),
                'time_p95_ms': performance.get('p95', 0),
                'time_cv': performance.get('cv', 0),
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

    def _export_json_results(self, results: Dict[str, Any], filename: str):
        """Export full results as JSON."""
        json_path = self.output_dir / filename
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Full results saved to: {json_path}")

    def _export_statistical_results(self, statistical_results: List, filename: str):
        """Export statistical analysis results."""
        # Convert dataclass to dict for pandas
        data = [result.__dict__ for result in statistical_results]
        df = pd.DataFrame(data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"📊 Statistical analysis saved to: {output_path}")
