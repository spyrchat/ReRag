"""Report generation for benchmark results."""

from typing import Dict, List, Any


class BenchmarkReportGenerator:
    """Generates reports and summaries from benchmark results."""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode

    def print_scenario_summary(self, scenario_name: str, result: Dict[str, Any]):
        """Print summary for a single scenario."""
        print(f"\n📊 {scenario_name} Results:")
        print("-" * 40)

        metrics = result.get('metrics', {})
        config = result.get('config', {})
        performance = result.get('performance', {})

        print(f"   Total Queries: {config.get('total_queries', 0)}")
        print(
            f"   Time (ms): {performance.get('mean', 0):.1f}±{performance.get('std', 0):.1f} | "
            f"Median: {performance.get('median', 0):.1f} | "
            f"P95: {performance.get('p95', 0):.1f} | "
            f"CV: {performance.get('cv', 0):.3f}")

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

    def print_statistical_report(self, statistical_results: List):
        """Print comprehensive statistical analysis report."""
        print("\n📈 STATISTICAL SIGNIFICANCE ANALYSIS")
        print("=" * 60)
        print(
            f"Bonferroni-corrected α = {statistical_results[0].corrected_alpha:.6f}")
        print()

        # Group by metric
        metrics = set(r.metric for r in statistical_results)

        for metric in sorted(metrics):
            print(f"\n{metric.upper()}:")
            print("-" * 40)

            metric_results = [
                r for r in statistical_results if r.metric == metric]

            for result in metric_results:
                significance = "***" if result.bonferroni_significant else "ns"
                direction = "↑" if result.mean_diff > 0 else "↓"

                print(f"{result.method2} vs {result.method1}: "
                      f"{direction} {abs(result.mean_diff):.4f} "
                      f"(p={result.p_value:.4f}, d={result.effect_size:.3f}, "
                      f"{result.effect_magnitude}) {significance}")
