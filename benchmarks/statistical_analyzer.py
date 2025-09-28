"""Statistical analysis for benchmark results."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    method1: str
    method2: str
    metric: str
    mean_diff: float
    t_statistic: float
    p_value: float
    effect_size: float
    effect_magnitude: str
    significant: bool
    bonferroni_significant: bool
    ci_lower: float
    ci_upper: float
    corrected_alpha: float


class BenchmarkStatisticalAnalyzer:
    """Handles all statistical analysis for benchmark results."""

    def __init__(self):
        self.statistical_results = []

    def analyze_results(self, results: Dict[str, Any]) -> List[StatisticalResult]:
        """Perform comprehensive statistical analysis between methods."""
        print("\n📊 Performing statistical significance testing...")

        method_scores = self._extract_per_query_scores(results)
        statistical_results = self._perform_pairwise_tests(method_scores)
        self._apply_bonferroni_correction(statistical_results)

        self.statistical_results = statistical_results
        return statistical_results

    def _extract_per_query_scores(self, results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """Extract per-query scores for each method and metric."""
        method_scores = {}

        for scenario_name, result in results.items():
            method_scores[scenario_name] = {}

            if 'per_query_scores' in result:
                per_query = result['per_query_scores']

                for metric in ['precision@1', 'precision@5', 'recall@5', 'mrr', 'map', 'ndcg@5']:
                    method_scores[scenario_name][metric] = [
                        query_result.get(metric, 0) for query_result in per_query
                    ]

        return method_scores

    def _perform_pairwise_tests(self, method_scores: Dict[str, Dict[str, List[float]]]) -> List[StatisticalResult]:
        """Perform pairwise statistical tests between all methods."""
        statistical_results = []
        methods = list(method_scores.keys())

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                for metric in ['precision@5', 'recall@5', 'mrr', 'map']:
                    if metric in method_scores[method1] and metric in method_scores[method2]:
                        result = self._compare_methods(
                            method1, method2, metric,
                            method_scores[method1][metric],
                            method_scores[method2][metric]
                        )
                        statistical_results.append(result)

        return statistical_results

    def _compare_methods(self, method1: str, method2: str, metric: str,
                         scores1: List[float], scores2: List[float]) -> StatisticalResult:
        """Compare two methods on a specific metric."""
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(scores1, scores2)

        # Mean difference and confidence interval
        differences = [s2 - s1 for s1, s2 in zip(scores1, scores2)]
        mean_diff = np.mean(differences)

        # 95% CI for the difference
        ci = stats.t.interval(0.95, len(differences)-1,
                              loc=mean_diff, scale=stats.sem(differences))

        return StatisticalResult(
            method1=method1,
            method2=method2,
            metric=metric,
            mean_diff=mean_diff,
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_magnitude=self._interpret_effect_size(effect_size),
            significant=p_value < 0.05,
            bonferroni_significant=False,  # Will be set later
            ci_lower=ci[0],
            ci_upper=ci[1],
            corrected_alpha=0.05  # Will be updated
        )

    def _apply_bonferroni_correction(self, statistical_results: List[StatisticalResult]):
        """Apply Bonferroni correction for multiple comparisons."""
        num_tests = len(statistical_results)
        corrected_alpha = 0.05 / num_tests if num_tests > 0 else 0.05

        for result in statistical_results:
            result.bonferroni_significant = result.p_value < corrected_alpha
            result.corrected_alpha = corrected_alpha

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0
        return (mean2 - mean1) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
