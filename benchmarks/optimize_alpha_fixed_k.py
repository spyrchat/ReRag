"""
Fixed-k Alpha Optimization with Unordered Binary Ground Truth.

This script optimizes only the alpha parameter (dense-sparse fusion weight)
for a FIXED k value, using SET-BASED metrics appropriate for unordered
binary relevance ground truth.

Usage:
    python optimize_alpha_fixed_k.py \
        --scenario-yaml hybrid_bge_splade_fixed_k10.yml \
        --dataset-path /path/to/sosum/data \
        --output-dir results/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stratification import StratifiedRAGDatasetSplitter
from benchmarks_runner import BenchmarkRunner
import yaml
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import argparse
from dataclasses import dataclass
from qdrant_client import QdrantClient
import warnings


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str) -> None:
    """Save object as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_grid_float(s: str) -> List[float]:
    """Parse grid specification for float values."""
    s = s.strip()
    if ":" in s:
        # Format: "start:stop:step"
        parts = s.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        n = int(np.round((stop - start) / step)) + 1
        vals = [start + i * step for i in range(n)]
        # Ensure last value is exactly stop
        vals[-1] = stop
        return vals
    # Format: "val1,val2,val3"
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def set_alpha_topk(cfg: Dict[str, Any], alpha: float, top_k: int) -> Dict[str, Any]:
    """Set alpha and top_k in configuration (deep copy)."""
    c = copy.deepcopy(cfg)

    # Set retrieval parameters
    c.setdefault("retrieval", {})
    c["retrieval"]["top_k"] = int(top_k)
    c["retrieval"]["alpha"] = float(alpha)

    # Set in fusion config if exists
    c["retrieval"].setdefault("fusion", {})
    c["retrieval"]["fusion"]["alpha"] = float(alpha)

    # Set in sparse config if exists
    if "sparse" in c["retrieval"] and isinstance(c["retrieval"]["sparse"], dict):
        c["retrieval"]["sparse"]["alpha"] = float(alpha)

    # Ensure k is in evaluation k_values
    c.setdefault("evaluation", {}).setdefault("k_values", [])
    if top_k not in c["evaluation"]["k_values"]:
        c["evaluation"]["k_values"].append(top_k)

    return c


class SplitFilteringAdapter:
    """
    Wrap a base adapter and filter queries by allowed IDs.
    For use with CV splits.
    """

    def __init__(
        self,
        base_adapter: Any,
        allowed_query_ids: Optional[set],
        name_suffix: str = ""
    ):
        self.base = base_adapter
        self.allowed = None if allowed_query_ids is None else set(
            str(x) for x in allowed_query_ids
        )
        self.name = getattr(base_adapter, "name", "adapter") + (
            f"-{name_suffix}" if name_suffix else ""
        )

        # Forward qdrant_client and collection_name if present
        if hasattr(base_adapter, "qdrant_client"):
            self.qdrant_client = base_adapter.qdrant_client
        if hasattr(base_adapter, "collection_name"):
            self.collection_name = base_adapter.collection_name

    def load_queries(self, *args, **kwargs):
        """Load and filter queries."""
        # Forward attributes to base if needed
        if hasattr(self, "qdrant_client") and not hasattr(self.base, "qdrant_client"):
            self.base.qdrant_client = self.qdrant_client
        if hasattr(self, "collection_name") and not hasattr(self.base, "collection_name"):
            self.base.collection_name = self.collection_name

        queries = self.base.load_queries(*args, **kwargs)

        if self.allowed is None:
            return queries

        return [q for q in queries if str(q.query_id) in self.allowed]


@dataclass
class OptimizationResult:
    """Results from alpha optimization."""
    alpha_star: float
    k_fixed: int
    cv_performance: Dict[str, float]
    fold_results: List[Dict[str, Any]]
    all_alpha_records: List[Dict[str, Any]]
    final_test_results: Dict[str, Any]
    config: Dict[str, Any]


class AlphaOptimizerFixedK:
    """
    Optimize alpha for FIXED k with composite objective function.

    Supports two optimization modes:
    1. Single metric (legacy): Optimize for one metric (e.g., F1@10)
    2. Agent composite: Multi-criteria objective balancing success rate,
       precision, recall, and latency for AI agent applications
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        base_adapter: Any,
        cv_info: Dict[str, Any],
        alpha_grid: List[float],
        k_fixed: int,
        optimization_metric: str = "f1@10",
        optimization_mode: str = "agent_composite",  # NEW
        objective_weights: Optional[Dict[str, float]] = None,  # NEW
        latency_target_ms: float = 500.0,  # NEW
        latency_max_penalty_ms: float = 1000.0,  # NEW
        report_k_values: List[int] = None,
        max_queries_dev: Optional[int] = None,
        max_queries_test: Optional[int] = None,
        epsilon: float = 0.01,
        latency_tiebreak: bool = False,  # Deprecated when using composite
        prefer_balanced_alpha: bool = True,
        verbose: bool = True
    ):
        self.base_config = base_config
        self.base_adapter = base_adapter
        self.cv_info = cv_info
        self.alpha_grid = alpha_grid
        self.k_fixed = k_fixed
        self.optimization_metric = optimization_metric
        self.optimization_mode = optimization_mode

        # Composite objective parameters
        self.objective_weights = objective_weights or {
            "w_success": 0.35,        # Success@3: at least one relevant
            "w_precision_early": 0.30,  # Precision@3: quality of top results
            "w_recall": 0.20,          # Recall@k: completeness
            "w_precision_full": 0.15   # Precision@k: overall quality
        }
        self.latency_target_ms = latency_target_ms
        self.latency_max_penalty_ms = latency_max_penalty_ms

        self.report_k_values = report_k_values or [1, 3, 5, 10, 15, 20]

        # Ensure k_fixed is in report_k_values
        if k_fixed not in self.report_k_values:
            self.report_k_values.append(k_fixed)
        self.report_k_values.sort()

        self.max_queries_dev = max_queries_dev
        self.max_queries_test = max_queries_test
        self.epsilon = float(epsilon)
        self.latency_tiebreak = latency_tiebreak
        self.prefer_balanced_alpha = prefer_balanced_alpha
        self.verbose = verbose

        # Validate configuration
        self._validate_configuration()

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("FIXED-K ALPHA OPTIMIZATION")
            if self.optimization_mode == "agent_composite":
                print("(Multi-Criteria Objective for AI Agent)")
            else:
                print("(Single Metric Optimization)")
            print(f"{'=' * 70}")
            print(f"Fixed k = {self.k_fixed}")

            if self.optimization_mode == "agent_composite":
                print(f"Optimization mode = {self.optimization_mode}")
                print(f"Objective weights:")
                for key, val in self.objective_weights.items():
                    print(f"  {key}: {val:.2f}")
                print(f"Latency target: {self.latency_target_ms}ms")
                print(f"Latency max penalty: {self.latency_max_penalty_ms}ms")
            else:
                print(f"Optimization metric = {self.optimization_metric}")

            print(f"Alpha grid = {self.alpha_grid}")
            print(f"Report at k = {self.report_k_values}")
            print(f"{'=' * 70}\n")

    def _validate_configuration(self):
        """Validate configuration parameters."""
        if self.optimization_mode not in ["single_metric", "agent_composite"]:
            raise ValueError(
                f"Invalid optimization_mode: {self.optimization_mode}. "
                f"Must be 'single_metric' or 'agent_composite'"
            )

        if self.optimization_mode == "single_metric":
            self._validate_single_metric()

        if self.optimization_mode == "agent_composite":
            # Validate weights sum to 1.0
            weight_sum = sum(self.objective_weights.values())
            if not np.isclose(weight_sum, 1.0, atol=0.01):
                warnings.warn(
                    f"Objective weights sum to {weight_sum:.3f}, not 1.0. "
                    f"This may lead to unexpected behavior."
                )

    def _validate_single_metric(self):
        """Validate single metric for legacy mode."""
        valid_metrics = ["precision", "recall", "f1", "r_precision", "success"]
        invalid_metrics = ["ndcg", "mrr", "map"]

        metric_lower = self.optimization_metric.lower()

        for invalid in invalid_metrics:
            if invalid in metric_lower:
                warnings.warn(
                    f"\n⚠️  WARNING: Using '{self.optimization_metric}' with "
                    f"UNORDERED binary ground truth may give misleading results.\n"
                    f"Consider using 'agent_composite' mode instead.\n"
                )

        if not any(valid in metric_lower for valid in valid_metrics):
            warnings.warn(
                f"\n⚠️  WARNING: Optimization metric '{self.optimization_metric}' "
                f"may not be appropriate for unordered binary ground truth.\n"
            )

    def _compute_objective_score(
        self,
        metrics: Dict[str, Any],
        latency_ms: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute objective score based on optimization mode.

        Returns:
            Tuple of (score, breakdown_dict)
        """
        if self.optimization_mode == "single_metric":
            score = metrics.get(self.optimization_metric, {}).get("mean", 0.0)
            breakdown = {
                "metric_value": score,
                "latency_ms": latency_ms
            }
            return score, breakdown

        elif self.optimization_mode == "agent_composite":
            w = self.objective_weights

            # Extract component metrics
            success_3 = metrics.get("success@3", {}).get("mean", 0.0)
            precision_3 = metrics.get("precision@3", {}).get("mean", 0.0)
            recall_k = metrics.get(
                f"recall@{self.k_fixed}", {}).get("mean", 0.0)
            precision_k = metrics.get(
                f"precision@{self.k_fixed}", {}).get("mean", 0.0)

            # Compute quality score
            quality_score = (
                w["w_success"] * success_3 +
                w["w_precision_early"] * precision_3 +
                w["w_recall"] * recall_k +
                w["w_precision_full"] * precision_k
            )

            # Compute latency penalty
            latency_penalty = 0.0
            if latency_ms > self.latency_target_ms:
                excess_ms = min(
                    latency_ms - self.latency_target_ms,
                    self.latency_max_penalty_ms
                )
                latency_penalty = 0.1 * \
                    (excess_ms / self.latency_max_penalty_ms)

            final_score = quality_score - latency_penalty

            # Detailed breakdown for analysis
            breakdown = {
                "composite_score": final_score,
                "quality_score": quality_score,
                "latency_penalty": latency_penalty,
                "success@3": success_3,
                "precision@3": precision_3,
                f"recall@{self.k_fixed}": recall_k,
                f"precision@{self.k_fixed}": precision_k,
                "latency_ms": latency_ms
            }

            return final_score, breakdown

        else:
            raise ValueError(
                f"Unknown optimization mode: {self.optimization_mode}")

    def _fold_keys(self) -> List[str]:
        return list(self.cv_info["splits"].keys())

    def _fold_role(self, fold_key: str) -> str:
        return self.cv_info["splits"][fold_key]["role"]

    def _ids_for(self, fold_key: str, split: str) -> List[str]:
        return self.cv_info["splits"][fold_key][split]

    def optimize(self) -> OptimizationResult:
        """Run the optimization procedure with composite objective."""

        fold_results: List[Dict[str, Any]] = []
        alpha_scores: Dict[float, List[float]] = {}
        alpha_breakdowns: Dict[float, List[Dict[str, float]]] = {}

        # Stage 1: CV optimization
        for fold_key in self._fold_keys():
            if self._fold_role(fold_key) != "optimization":
                continue

            dev_ids = self._ids_for(fold_key, "dev")

            if self.verbose:
                print(f"\n{'=' * 70}")
                print(f"[Fold {fold_key}] Dev size: {len(dev_ids)}")
                print(f"Optimizing α at k={self.k_fixed}")
                print(f"{'=' * 70}")

            dev_adapter = SplitFilteringAdapter(
                self.base_adapter,
                set(dev_ids),
                name_suffix=f"{fold_key}-dev"
            )

            best_score = -np.inf
            best_alpha = None
            best_breakdown = None

            for alpha in self.alpha_grid:
                cfg = set_alpha_topk(self.base_config, alpha, self.k_fixed)

                runner = BenchmarkRunner(cfg)
                aggregated = runner.run_benchmark(
                    dev_adapter,
                    max_queries=self.max_queries_dev
                )

                rt_ms = aggregated.get("performance", {}).get(
                    "mean", float("inf"))

                # Compute composite objective score
                score, breakdown = self._compute_objective_score(
                    aggregated["metrics"],
                    rt_ms
                )

                alpha_scores.setdefault(alpha, []).append(score)
                alpha_breakdowns.setdefault(alpha, []).append(breakdown)

                if self.verbose:
                    if self.optimization_mode == "agent_composite":
                        print(
                            f"  α={alpha:.2f} | "
                            f"composite={score:.4f} | "
                            f"success@3={breakdown['success@3']:.3f} | "
                            f"prec@3={breakdown['precision@3']:.3f} | "
                            f"recall@{self.k_fixed}={breakdown[f'recall@{self.k_fixed}']:.3f} | "
                            f"rt={rt_ms:.0f}ms"
                        )
                    else:
                        print(
                            f"  α={alpha:.2f} | "
                            f"{self.optimization_metric}={score:.4f} | "
                            f"rt={rt_ms:.1f}ms"
                        )

                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_alpha = alpha
                    best_breakdown = breakdown

            if best_alpha is None:
                raise RuntimeError(
                    f"No valid scores on dev for {fold_key}"
                )

            fold_result = {
                "fold": fold_key,
                "alpha": float(best_alpha),
                "k": self.k_fixed,
                "score": float(best_score)
            }

            if best_breakdown:
                fold_result["breakdown"] = {
                    k: float(v) for k, v in best_breakdown.items()
                }

            fold_results.append(fold_result)

            if self.verbose:
                print(
                    f"\n[Fold {fold_key}] Best: α={best_alpha:.2f}, score={best_score:.4f}")

        # Stage 2: Aggregate and select winner
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("AGGREGATING RESULTS")
            print(f"{'=' * 70}")

        records: List[Dict[str, Any]] = []
        for alpha in self.alpha_grid:
            scores = alpha_scores.get(alpha, [])
            breakdowns = alpha_breakdowns.get(alpha, [])

            valid_scores = [s for s in scores if not np.isnan(s)]
            if not valid_scores:
                continue

            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores))

            # Aggregate breakdown metrics
            aggregated_breakdown = {}
            if breakdowns:
                for key in breakdowns[0].keys():
                    values = [b.get(key, 0.0) for b in breakdowns]
                    aggregated_breakdown[f"{key}_mean"] = float(
                        np.mean(values))
                    aggregated_breakdown[f"{key}_std"] = float(np.std(values))

            record = {
                "alpha": float(alpha),
                "score_mean": mean_score,
                "score_std": std_score,
                **aggregated_breakdown
            }

            records.append(record)

        if not records:
            raise RuntimeError("No valid alpha records across folds")

        # Sort and display
        records_sorted = sorted(records, key=lambda r: -r["score_mean"])

        if self.verbose:
            print(f"\nTop configurations:")
            if self.optimization_mode == "agent_composite":
                print(
                    f"{'Rank':<5} {'α':<6} {'Composite':<10} "
                    f"{'Success@3':<11} {'Prec@3':<9} {'Latency':<10}"
                )
            else:
                print(
                    f"{'Rank':<5} {'α':<6} {'Score':<12} {'Std':<8} {'Latency':<10}"
                )
            print("-" * 70)

            for i, r in enumerate(records_sorted[:10], 1):
                if self.optimization_mode == "agent_composite":
                    print(
                        f"{i:<5} {r['alpha']:<6.2f} {r['score_mean']:<10.4f} "
                        f"{r.get('success@3_mean', 0):<11.3f} "
                        f"{r.get('precision@3_mean', 0):<9.3f} "
                        f"{r.get('latency_ms_mean', 0):<10.0f}"
                    )
                else:
                    print(
                        f"{i:<5} {r['alpha']:<6.2f} {r['score_mean']:<12.4f} "
                        f"{r['score_std']:<8.4f} {r.get('latency_ms_mean', 0):<10.1f}"
                    )

        # Winner selection with tie-breaking
        best_mean = max(r["score_mean"] for r in records)

        candidates = [
            r for r in records
            if (best_mean - r["score_mean"]) <= self.epsilon * abs(best_mean)
        ]

        # Tie-breaking (only for single_metric mode)
        if self.optimization_mode == "single_metric" and self.latency_tiebreak:
            candidates.sort(key=lambda r: r.get(
                "latency_ms_mean", float("inf")))

        if self.prefer_balanced_alpha:
            candidates.sort(key=lambda r: abs(r["alpha"] - 0.5))

        candidates.sort(key=lambda r: -r["score_mean"])

        winner = candidates[0]
        alpha_star = float(winner["alpha"])

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("SELECTED CONFIGURATION")
            print(f"{'=' * 70}")
            print(f"α* = {alpha_star:.2f}")
            print(f"k (fixed) = {self.k_fixed}")
            print(
                f"Score = {winner['score_mean']:.4f} ± {winner['score_std']:.4f}")

            if self.optimization_mode == "agent_composite":
                print(f"\nComponent breakdown:")
                print(f"  Success@3:    {winner.get('success@3_mean', 0):.3f}")
                print(
                    f"  Precision@3:  {winner.get('precision@3_mean', 0):.3f}")
                print(
                    f"  Recall@{self.k_fixed}:     {winner.get(f'recall@{self.k_fixed}_mean', 0):.3f}")
                print(
                    f"  Precision@{self.k_fixed}: {winner.get(f'precision@{self.k_fixed}_mean', 0):.3f}")
                print(
                    f"  Latency:      {winner.get('latency_ms_mean', 0):.0f} ms")

            print(f"{'=' * 70}\n")

        # Stage 3: Final test evaluation
        final_fold_key = next(
            (fk for fk in self._fold_keys() if self._fold_role(fk) == "final_test"),
            None
        )

        if final_fold_key is None:
            raise RuntimeError("No 'final_test' fold found")

        test_ids = self._ids_for(final_fold_key, "test")
        test_adapter = SplitFilteringAdapter(
            self.base_adapter,
            set(test_ids),
            name_suffix=f"{final_fold_key}-test"
        )

        final_cfg = set_alpha_topk(self.base_config, alpha_star, self.k_fixed)

        # Add all report k values
        final_cfg.setdefault("evaluation", {}).setdefault("k_values", [])
        for k in self.report_k_values:
            if k not in final_cfg["evaluation"]["k_values"]:
                final_cfg["evaluation"]["k_values"].append(k)

        if self.verbose:
            print(f"{'=' * 70}")
            print("FINAL TEST EVALUATION")
            print(f"{'=' * 70}")
            print(f"Configuration: α={alpha_star:.2f}, k={self.k_fixed}")
            print(f"Evaluating at k = {self.report_k_values}")
            print(f"{'=' * 70}\n")

        final_runner = BenchmarkRunner(final_cfg)
        final_agg = final_runner.run_benchmark_with_individual_results(
            test_adapter,
            max_queries=self.max_queries_test
        )

        # Extract comprehensive metrics summary
        test_metrics = {}
        for k in self.report_k_values:
            for metric_type in ["precision", "recall", "f1", "success", "ndcg", "mrr"]:
                key = f"{metric_type}@{k}"
                if key in final_agg["metrics"]:
                    test_metrics[key] = final_agg["metrics"][key]["mean"]

        if self.verbose:
            print("\nTest set results:")
            for k in self.report_k_values:
                marker = " ←" if k == self.k_fixed else ""
                f1_key = f"f1@{k}"
                if f1_key in test_metrics:
                    print(f"  F1@{k:>2} = {test_metrics[f1_key]:.4f}{marker}")

        return OptimizationResult(
            alpha_star=alpha_star,
            k_fixed=self.k_fixed,
            cv_performance={
                "mean": winner["score_mean"],
                "std": winner["score_std"],
                **{k: v for k, v in winner.items()
                   if k not in ["alpha", "score_mean", "score_std"]}
            },
            fold_results=fold_results,
            all_alpha_records=records,
            final_test_results={
                "aggregated": final_agg,
                "metrics_summary": test_metrics
            },
            config={
                "optimization_mode": self.optimization_mode,
                "optimization_metric": self.optimization_metric,
                "objective_weights": self.objective_weights,
                "k_fixed": self.k_fixed,
                "report_k_values": self.report_k_values,
                "epsilon": self.epsilon,
                "latency_target_ms": self.latency_target_ms
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-k alpha optimization for unordered binary GT"
    )
    parser.add_argument(
        "--scenario-yaml",
        required=True,
        help="Path to scenario YAML"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to dataset (containing questions.csv)"
    )
    parser.add_argument(
        "--adapter-module",
        default="benchmarks.adapters.stackoverflow_adapter",
        help="Module path for adapter"
    )
    parser.add_argument(
        "--adapter-class",
        default="StackOverflowBenchmarkAdapter",
        help="Adapter class name"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--max-queries-dev",
        type=int,
        default=None,
        help="Max queries per dev fold (for testing)"
    )
    parser.add_argument(
        "--max-queries-test",
        type=int,
        default=None,
        help="Max queries for final test"
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Load scenario
    scenario = load_yaml(args.scenario_yaml)
    base_cfg = scenario["pipeline"]

    # Extract k_fixed from config
    k_fixed = base_cfg["retrieval"]["top_k"]
    print(f"\nFixed k = {k_fixed} (from configuration)")

    # Parse alpha grid
    alpha_grid = parse_grid_float(scenario["grid"]["alpha"])
    print(f"Alpha grid: {alpha_grid}")

    # Create stratified CV splits
    print(f"\nCreating {args.n_folds}-fold stratified splits...")
    splitter = StratifiedRAGDatasetSplitter(
        dataset_path=args.dataset_path,
        random_state=42
    )
    splitter.load_dataset()
    cv_info = splitter.create_cv_splits(n_folds=args.n_folds)

    # Initialize Qdrant client
    qdrant_cfg = base_cfg["retrieval"]["qdrant"]
    qdrant_client = QdrantClient(
        host=qdrant_cfg.get("host", "localhost"),
        port=qdrant_cfg.get("port", 6333)
    )
    collection_name = qdrant_cfg["collection_name"]

    # Initialize base adapter - load dynamically from config
    from pipelines.adapters.loader import AdapterLoader

    adapter_spec = base_cfg["dataset"].get("adapter")
    if not adapter_spec:
        # Fallback to CLI args for backwards compatibility
        adapter_spec = f"{args.adapter_module}.{args.adapter_class}"

    base_adapter = AdapterLoader.load_adapter(
        adapter_spec=adapter_spec,
        dataset_path=args.dataset_path,
        version="1.0.0",
        qdrant_client=qdrant_client,
        collection_name=collection_name
    )

    # Initialize optimizer
    optimization_metric = base_cfg["evaluation"].get(
        "optimization_metric",
        f"f1@{k_fixed}"
    )

    optimizer = AlphaOptimizerFixedK(
        base_config=base_cfg,
        base_adapter=base_adapter,
        cv_info=cv_info,
        alpha_grid=alpha_grid,
        k_fixed=k_fixed,
        optimization_mode="agent_composite",
        optimization_metric=optimization_metric,
        report_k_values=base_cfg["evaluation"]["k_values"],
        max_queries_dev=args.max_queries_dev,
        max_queries_test=args.max_queries_test,
        epsilon=scenario.get("optimization", {}).get("epsilon", 0.01),
        latency_tiebreak=scenario.get(
            "optimization", {}).get("latency_tiebreak", True),
        prefer_balanced_alpha=scenario.get(
            "optimization", {}).get("prefer_balanced_alpha", True),
        verbose=True
    )

    # Run optimization
    print("\nStarting optimization...")
    result = optimizer.optimize()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "fixed_k_alpha_optimization",
        "dataset": base_cfg["dataset"]["path"],
        "ground_truth_type": "unordered_binary",
        "alpha_star": result.alpha_star,
        "k_fixed": result.k_fixed,
        "optimization_metric": optimization_metric,
        "cv_performance": result.cv_performance,
        "fold_results": result.fold_results,
        "all_alpha_records": result.all_alpha_records,
        "final_test_metrics": result.final_test_results["metrics_summary"],
        "config": result.config
    }

    output_file = output_dir / f"optimization_results_k{k_fixed}.json"
    save_json(summary, str(output_file))

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Optimal α = {result.alpha_star:.2f}")
    print(f"Fixed k = {result.k_fixed}")
    print(f"CV {optimization_metric} = "
          f"{result.cv_performance['mean']:.4f} ± "
          f"{result.cv_performance['std']:.4f}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
