"""
2D Grid Search: Optimize both alpha and rrf_k for fixed retrieval k.

This script implements exhaustive 2D grid search with stratified cross-validation
for hyperparameter optimization in RAG systems without trainable parameters.

Key features:
- Maximizes data usage: 80% of data per optimization fold
- Stratified sampling preserves dataset characteristics
- Repeated evaluation reduces selection bias
- Independent final test set for unbiased performance estimation
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
import itertools


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
        parts = s.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        n = int(np.round((stop - start) / step)) + 1
        vals = [start + i * step for i in range(n)]
        vals[-1] = stop
        return vals
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_grid_int(spec) -> List[int]:
    """Parse grid specification for integer values."""
    if isinstance(spec, list):
        return [int(x) for x in spec]

    if isinstance(spec, str):
        spec = spec.strip()
        if ":" in spec:
            parts = spec.split(":")
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
            return list(range(start, stop + 1, step))
        return [int(x.strip()) for x in spec.split(",") if x.strip()]

    raise ValueError(f"Invalid grid specification: {spec}")


def set_hyperparameters(
    cfg: Dict[str, Any],
    alpha: float,
    rrf_k: int,
    retrieval_k: int
) -> Dict[str, Any]:
    """
    Set all hyperparameters in configuration (deep copy).

    Args:
        cfg: Base configuration
        alpha: Dense-sparse fusion weight
        rrf_k: RRF constant parameter
        retrieval_k: Retrieval depth (fixed, but needed for config)
    """
    c = copy.deepcopy(cfg)

    # Set retrieval parameters
    c.setdefault("retrieval", {})
    c["retrieval"]["top_k"] = int(retrieval_k)
    c["retrieval"]["alpha"] = float(alpha)

    # Set in fusion config
    c["retrieval"].setdefault("fusion", {})
    c["retrieval"]["fusion"]["alpha"] = float(alpha)
    c["retrieval"]["fusion"]["rrf_k"] = int(rrf_k)

    # Set in sparse config if exists (legacy support)
    if "sparse" in c["retrieval"] and isinstance(c["retrieval"]["sparse"], dict):
        c["retrieval"]["sparse"]["alpha"] = float(alpha)

    # Ensure retrieval_k is in evaluation k_values
    c.setdefault("evaluation", {}).setdefault("k_values", [])
    if retrieval_k not in c["evaluation"]["k_values"]:
        c["evaluation"]["k_values"].append(retrieval_k)

    return c


class SplitFilteringAdapter:
    """Wrap adapter and filter queries by allowed IDs."""

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

        if hasattr(base_adapter, "qdrant_client"):
            self.qdrant_client = base_adapter.qdrant_client
        if hasattr(base_adapter, "collection_name"):
            self.collection_name = base_adapter.collection_name

    def load_queries(self, *args, **kwargs):
        """Load and filter queries."""
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
    """Results from 2D grid search optimization."""
    alpha_star: float
    rrf_k_star: int
    k_fixed: int
    cv_performance: Dict[str, float]
    fold_results: List[Dict[str, Any]]
    all_config_records: List[Dict[str, Any]]
    final_test_results: Dict[str, Any]
    config: Dict[str, Any]


class TwoDimensionalGridSearchOptimizer:
    """
    2D Grid search optimizer for (alpha, rrf_k) hyperparameters.

    Μεθοδολογία:
    - Exhaustive 2D grid search over hyperparameter space
    - Stratified cross-validation for robust performance estimation
    - Each optimization fold uses 80% of data (maximizing sample size)
    - Independent final test fold (20%) for unbiased evaluation
    - Composite objective function balancing multiple metrics

    Επιστημονική Αιτιολόγηση:
    Η cross-validation χρησιμοποιείται για τη μείωση της μεροληψίας επιλογής
    (selection bias) και την ποσοτικοποίηση της στατιστικής αβεβαιότητας των
    μετρικών απόδοσης, όχι για την πρόληψη υπερπροσαρμογής εκπαιδεύσιμων
    παραμέτρων (καθώς δεν υπάρχει φάση εκπαίδευσης).
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        base_adapter: Any,
        cv_info: Dict[str, Any],
        alpha_grid: List[float],
        rrf_k_grid: List[int],
        k_fixed: int,
        optimization_mode: str = "agent_composite",
        objective_weights: Optional[Dict[str, float]] = None,
        latency_target_ms: float = 500.0,
        latency_max_penalty_ms: float = 1000.0,
        report_k_values: List[int] = None,
        max_queries_dev: Optional[int] = None,
        max_queries_test: Optional[int] = None,
        epsilon: float = 0.01,
        prefer_balanced_alpha: bool = True,
        prefer_standard_rrf: bool = True,
        standard_rrf_k: int = 60,
        verbose: bool = True
    ):
        self.base_config = base_config
        self.base_adapter = base_adapter
        self.cv_info = cv_info
        self.alpha_grid = alpha_grid
        self.rrf_k_grid = rrf_k_grid
        self.k_fixed = k_fixed
        self.optimization_mode = optimization_mode

        self.objective_weights = objective_weights or {
            "w_success": 0.35,
            "w_precision_early": 0.30,
            "w_recall": 0.20,
            "w_precision_full": 0.15
        }
        self.latency_target_ms = latency_target_ms
        self.latency_max_penalty_ms = latency_max_penalty_ms

        self.report_k_values = report_k_values or [1, 3, 5, 10, 15, 20]
        if k_fixed not in self.report_k_values:
            self.report_k_values.append(k_fixed)
        self.report_k_values.sort()

        self.max_queries_dev = max_queries_dev
        self.max_queries_test = max_queries_test
        self.epsilon = float(epsilon)
        self.prefer_balanced_alpha = prefer_balanced_alpha
        self.prefer_standard_rrf = prefer_standard_rrf
        self.standard_rrf_k = standard_rrf_k
        self.verbose = verbose

        # Calculate total combinations
        self.total_combinations = len(self.alpha_grid) * len(self.rrf_k_grid)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("2D GRID SEARCH OPTIMIZATION WITH CROSS-VALIDATION")
            print(f"{'=' * 70}")
            print(f"Fixed k = {self.k_fixed} (retrieval depth)")
            print(f"\nHyperparameter Grid:")
            print(f"  Alpha:  {len(self.alpha_grid)} values {self.alpha_grid}")
            print(f"  RRF k:  {len(self.rrf_k_grid)} values {self.rrf_k_grid}")
            print(f"  Total:  {self.total_combinations} combinations")
            print(f"\nOptimization mode: {self.optimization_mode}")
            if self.optimization_mode == "agent_composite":
                print(f"Objective weights:")
                for key, val in self.objective_weights.items():
                    print(f"  {key}: {val:.2f}")
            print(f"\nMethodology:")
            print(
                f"  - Each config evaluated on {self.cv_info['metadata']['optimization_folds']} folds")
            print(
                f"  - Each fold uses ~80% of data ({self.cv_info['metadata']['samples_per_optimization_fold']} samples)")
            print(
                f"  - Final test on independent 20% ({self.cv_info['metadata']['samples_final_test']} samples)")
            print(f"{'=' * 70}\n")

    def _compute_objective_score(
        self,
        metrics: Dict[str, Any],
        latency_ms: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Υπολογισμός σύνθετης αντικειμενικής συνάρτησης.

        Η αντικειμενική συνάρτηση συνδυάζει πολλαπλές μετρικές ποιότητας
        με ποινή για αυξημένη latency, επιτρέποντας τη βελτιστοποίηση
        για την πρακτική αποδοτικότητα του συστήματος.
        """
        w = self.objective_weights

        success_3 = metrics.get("success@3", {}).get("mean", 0.0)
        precision_3 = metrics.get("precision@3", {}).get("mean", 0.0)
        recall_k = metrics.get(f"recall@{self.k_fixed}", {}).get("mean", 0.0)
        precision_k = metrics.get(
            f"precision@{self.k_fixed}", {}).get("mean", 0.0)

        quality_score = (
            w["w_success"] * success_3 +
            w["w_precision_early"] * precision_3 +
            w["w_recall"] * recall_k +
            w["w_precision_full"] * precision_k
        )

        latency_penalty = 0.0
        if latency_ms > self.latency_target_ms:
            excess_ms = min(
                latency_ms - self.latency_target_ms,
                self.latency_max_penalty_ms
            )
            latency_penalty = 0.1 * (excess_ms / self.latency_max_penalty_ms)

        final_score = quality_score - latency_penalty

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

    def _fold_keys(self) -> List[str]:
        return list(self.cv_info["splits"].keys())

    def _fold_role(self, fold_key: str) -> str:
        return self.cv_info["splits"][fold_key]["role"]

    def _ids_for(self, fold_key: str, split: str) -> List[str]:
        return self.cv_info["splits"][fold_key][split]

    def optimize(self) -> OptimizationResult:
        """
        Εκτέλεση 2D grid search με stratified cross-validation.

        Η διαδικασία περιλαμβάνει τρία στάδια:
        1. Optimization Phase: Αξιολόγηση όλων των configurations σε πολλαπλά folds
        2. Aggregation Phase: Συνάθροιση αποτελεσμάτων και επιλογή βέλτιστης configuration
        3. Final Test Phase: Ανεξάρτητη επικύρωση στο final test set
        """

        fold_results: List[Dict[str, Any]] = []
        config_scores: Dict[Tuple[float, int], List[float]] = {}
        config_breakdowns: Dict[Tuple[float, int], List[Dict[str, float]]] = {}

        # ========================================================================
        # STAGE 1: CV OPTIMIZATION OVER 2D GRID
        # ========================================================================
        for fold_key in self._fold_keys():
            if self._fold_role(fold_key) != "optimization":
                continue

            # Συλλογή όλων των διαθέσιμων IDs για αξιολόγηση
            # Σημείωση: Με το νέο stratification, όλα τα δεδομένα βρίσκονται στο 'train'
            # και το 'dev' είναι κενό. Η συγχώνευση διατηρείται για backward compatibility.
            train_ids = self._ids_for(fold_key, "train")
            dev_ids = self._ids_for(fold_key, "dev")

            if dev_ids:
                # Backward compatibility: Εάν υπάρχουν dev_ids, συγχώνευση
                eval_ids = list(set(train_ids) | set(dev_ids))
                print(f"\n{'=' * 70}")
                print(f"[Fold {fold_key}] Evaluation size: {len(eval_ids)} "
                      f"(train: {len(train_ids)}, dev: {len(dev_ids)})")
            else:
                # Νέα αρχιτεκτονική: Όλα στο train
                eval_ids = train_ids
                print(f"\n{'=' * 70}")
                print(f"[Fold {fold_key}] Evaluation size: {len(eval_ids)} "
                      f"(80% of dataset)")

            print(f"Testing {self.total_combinations} configurations")
            print(f"{'=' * 70}")

            # Δημιουργία adapter για το evaluation set
            eval_adapter = SplitFilteringAdapter(
                self.base_adapter,
                set(eval_ids),
                name_suffix=f"{fold_key}-eval"
            )

            best_score = -np.inf
            best_alpha = None
            best_rrf_k = None
            best_breakdown = None

            eval_count = 0

            # Επανάληψη σε όλους τους συνδυασμούς (alpha, rrf_k)
            for alpha, rrf_k in itertools.product(self.alpha_grid, self.rrf_k_grid):
                eval_count += 1

                cfg = set_hyperparameters(
                    self.base_config,
                    alpha=alpha,
                    rrf_k=rrf_k,
                    retrieval_k=self.k_fixed
                )

                runner = BenchmarkRunner(cfg)
                aggregated = runner.run_benchmark(
                    eval_adapter,
                    max_queries=self.max_queries_dev
                )

                rt_ms = aggregated.get("performance", {}).get(
                    "mean", float("inf"))

                score, breakdown = self._compute_objective_score(
                    aggregated["metrics"],
                    rt_ms
                )

                config_key = (alpha, rrf_k)
                config_scores.setdefault(config_key, []).append(score)
                config_breakdowns.setdefault(config_key, []).append(breakdown)

                if self.verbose and eval_count % 10 == 0:
                    print(f"  Progress: {eval_count}/{self.total_combinations} "
                          f"(α={alpha:.2f}, rrf_k={rrf_k})")

                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_alpha = alpha
                    best_rrf_k = rrf_k
                    best_breakdown = breakdown

            if best_alpha is None or best_rrf_k is None:
                raise RuntimeError(f"No valid scores for {fold_key}")

            fold_result = {
                "fold": fold_key,
                "alpha": float(best_alpha),
                "rrf_k": int(best_rrf_k),
                "k": self.k_fixed,
                "score": float(best_score)
            }

            if best_breakdown:
                fold_result["breakdown"] = {
                    k: float(v) for k, v in best_breakdown.items()
                }

            fold_results.append(fold_result)

            if self.verbose:
                print(f"\n[Fold {fold_key}] Best configuration:")
                print(f"  α = {best_alpha:.2f}, rrf_k = {best_rrf_k}")
                print(f"  Score = {best_score:.4f}")

        # ========================================================================
        # STAGE 2: AGGREGATE AND SELECT WINNER
        # ========================================================================
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("AGGREGATING RESULTS ACROSS FOLDS")
            print(f"{'=' * 70}")

        expected_folds = sum(1 for fk in self._fold_keys()
                             if self._fold_role(fk) == "optimization")

        records: List[Dict[str, Any]] = []
        incomplete_configs = []

        for config_key, scores in config_scores.items():
            alpha, rrf_k = config_key
            breakdowns = config_breakdowns.get(config_key, [])

            valid_scores = [s for s in scores if not np.isnan(s)]
            if not valid_scores:
                continue

            num_folds_tested = len(valid_scores)

            # Παρακολούθηση configurations που δεν δοκιμάστηκαν σε όλα τα folds
            if num_folds_tested < expected_folds:
                incomplete_configs.append({
                    "alpha": alpha,
                    "rrf_k": rrf_k,
                    "folds_tested": num_folds_tested,
                    "folds_expected": expected_folds
                })
                warnings.warn(
                    f"Config (α={alpha:.2f}, rrf_k={rrf_k}) only tested on "
                    f"{num_folds_tested}/{expected_folds} folds. Skipping for fairness."
                )
                continue

            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores))

            aggregated_breakdown = {}
            if breakdowns:
                for key in breakdowns[0].keys():
                    values = [b.get(key, 0.0) for b in breakdowns]
                    aggregated_breakdown[f"{key}_mean"] = float(
                        np.mean(values))
                    aggregated_breakdown[f"{key}_std"] = float(np.std(values))

            record = {
                "alpha": float(alpha),
                "rrf_k": int(rrf_k),
                "score_mean": mean_score,
                "score_std": std_score,
                "num_folds": num_folds_tested,
                **aggregated_breakdown
            }

            records.append(record)

        if not records:
            raise RuntimeError("No valid configuration records across folds")

        # Καταγραφή incomplete configurations
        if incomplete_configs and self.verbose:
            print(
                f"\n⚠️  Skipped {len(incomplete_configs)} incomplete configurations:")
            for cfg in incomplete_configs[:5]:
                print(f"   α={cfg['alpha']:.2f}, rrf_k={cfg['rrf_k']}: "
                      f"{cfg['folds_tested']}/{cfg['folds_expected']} folds")
            if len(incomplete_configs) > 5:
                print(f"   ... and {len(incomplete_configs) - 5} more")

        # Ταξινόμηση και εμφάνιση κορυφαίων configurations
        records_sorted = sorted(records, key=lambda r: -r["score_mean"])

        if self.verbose:
            print(f"\nTop 15 configurations:")
            print(f"{'Rank':<5} {'α':<6} {'rrf_k':<7} {'Score':<10} "
                  f"{'Success@3':<11} {'Prec@3':<9} {'Latency':<10}")
            print("-" * 75)

            for i, r in enumerate(records_sorted[:15], 1):
                print(
                    f"{i:<5} {r['alpha']:<6.2f} {r['rrf_k']:<7} "
                    f"{r['score_mean']:<10.4f} "
                    f"{r.get('success@3_mean', 0):<11.3f} "
                    f"{r.get('precision@3_mean', 0):<9.3f} "
                    f"{r.get('latency_ms_mean', 0):<10.0f}"
                )

        # Επιλογή νικητή με tie-breaking
        best_mean = max(r["score_mean"] for r in records)

        candidates = [
            r for r in records
            if (best_mean - r["score_mean"]) <= self.epsilon * abs(best_mean)
        ]

        # Tie-breaking με προτεραιότητα
        candidates.sort(
            key=lambda r: (
                -r["score_mean"],  # Primary: μεγιστοποίηση score
                # Secondary
                abs(r["alpha"] - 0.5) if self.prefer_balanced_alpha else 0,
                # Tertiary
                abs(r["rrf_k"] -
                    self.standard_rrf_k) if self.prefer_standard_rrf else 0,
            )
        )

        winner = candidates[0]
        alpha_star = float(winner["alpha"])
        rrf_k_star = int(winner["rrf_k"])

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("SELECTED CONFIGURATION")
            print(f"{'=' * 70}")
            print(f"α* = {alpha_star:.2f}")
            print(f"rrf_k* = {rrf_k_star}")
            print(f"k (fixed) = {self.k_fixed}")
            print(
                f"Score = {winner['score_mean']:.4f} ± {winner['score_std']:.4f}")
            print(f"\nComponent breakdown:")
            print(f"  Success@3:    {winner.get('success@3_mean', 0):.3f}")
            print(f"  Precision@3:  {winner.get('precision@3_mean', 0):.3f}")
            print(
                f"  Recall@{self.k_fixed}:     {winner.get(f'recall@{self.k_fixed}_mean', 0):.3f}")
            print(f"  Latency:      {winner.get('latency_ms_mean', 0):.0f} ms")
            print(f"{'=' * 70}\n")

        # ========================================================================
        # STAGE 3: FINAL TEST EVALUATION
        # ========================================================================
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

        final_cfg = set_hyperparameters(
            self.base_config,
            alpha=alpha_star,
            rrf_k=rrf_k_star,
            retrieval_k=self.k_fixed
        )

        # Προσθήκη όλων των k values για reporting
        final_cfg.setdefault("evaluation", {}).setdefault("k_values", [])
        for k in self.report_k_values:
            if k not in final_cfg["evaluation"]["k_values"]:
                final_cfg["evaluation"]["k_values"].append(k)

        if self.verbose:
            print(f"{'=' * 70}")
            print("FINAL TEST EVALUATION")
            print(f"{'=' * 70}")
            print(
                f"Configuration: α={alpha_star:.2f}, rrf_k={rrf_k_star}, k={self.k_fixed}")
            print(f"Test set size: {len(test_ids)} samples (20% of dataset)")
            print(f"{'=' * 70}\n")

        final_runner = BenchmarkRunner(final_cfg)
        final_agg = final_runner.run_benchmark_with_individual_results(
            test_adapter,
            max_queries=self.max_queries_test
        )

        # Εξαγωγή σύνοψης μετρικών
        test_metrics = {}
        for k in self.report_k_values:
            for metric_type in ["precision", "recall", "f1", "success"]:
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
            rrf_k_star=rrf_k_star,
            k_fixed=self.k_fixed,
            cv_performance={
                "mean": winner["score_mean"],
                "std": winner["score_std"],
                **{k: v for k, v in winner.items()
                   if k not in ["alpha", "rrf_k", "score_mean", "score_std"]}
            },
            fold_results=fold_results,
            all_config_records=records,
            final_test_results={
                "aggregated": final_agg,
                "metrics_summary": test_metrics
            },
            config={
                "optimization_mode": self.optimization_mode,
                "objective_weights": self.objective_weights,
                "k_fixed": self.k_fixed,
                "alpha_grid": self.alpha_grid,
                "rrf_k_grid": self.rrf_k_grid,
                "report_k_values": self.report_k_values,
                "epsilon": self.epsilon,
                "cv_strategy": "optimized_stratified_cv",
                "data_usage_per_fold": "80%",
                "note": "Each configuration evaluated on multiple folds with 80% data each for robust estimation"
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="2D grid search with optimized stratified cross-validation"
    )
    parser.add_argument("--scenario-yaml", required=True,
                        help="YAML configuration file with grid specifications")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to dataset directory")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--max-queries-dev", type=int, default=None,
                        help="Maximum queries per optimization fold (optional)")
    parser.add_argument("--max-queries-test", type=int, default=None,
                        help="Maximum queries for final test (optional)")
    parser.add_argument("--output-dir", default="results/",
                        help="Output directory for results")

    args = parser.parse_args()

    # Load scenario
    scenario = load_yaml(args.scenario_yaml)
    base_cfg = scenario["pipeline"]

    k_fixed = base_cfg["retrieval"]["top_k"]
    print(f"\nFixed k = {k_fixed} (retrieval depth)")

    # Parse grids
    alpha_grid = parse_grid_float(scenario["grid"]["alpha"])
    rrf_k_grid = parse_grid_int(scenario["grid"]["rrf_k"])

    print(f"Alpha grid: {alpha_grid}")
    print(f"RRF k grid: {rrf_k_grid}")
    print(f"Total combinations: {len(alpha_grid) * len(rrf_k_grid)}")

    # Create CV splits with optimized strategy
    print(f"\nCreating {args.n_folds}-fold optimized stratified splits...")
    splitter = StratifiedRAGDatasetSplitter(
        dataset_path=args.dataset_path,
        random_state=42
    )
    splitter.load_dataset()
    cv_info = splitter.create_cv_splits(n_folds=args.n_folds)

    # Initialize Qdrant
    qdrant_cfg = base_cfg["retrieval"]["qdrant"]
    qdrant_client = QdrantClient(
        host=qdrant_cfg.get("host", "localhost"),
        port=qdrant_cfg.get("port", 6333)
    )
    collection_name = qdrant_cfg["collection_name"]

    # Load adapter
    from pipelines.adapters.loader import AdapterLoader
    adapter_spec = base_cfg["dataset"].get("adapter")
    base_adapter = AdapterLoader.load_adapter(
        adapter_spec=adapter_spec,
        dataset_path=args.dataset_path,
        version="1.0.0",
        qdrant_client=qdrant_client,
        collection_name=collection_name
    )

    # Initialize optimizer
    optimizer = TwoDimensionalGridSearchOptimizer(
        base_config=base_cfg,
        base_adapter=base_adapter,
        cv_info=cv_info,
        alpha_grid=alpha_grid,
        rrf_k_grid=rrf_k_grid,
        k_fixed=k_fixed,
        optimization_mode="agent_composite",
        report_k_values=base_cfg["evaluation"]["k_values"],
        max_queries_dev=args.max_queries_dev,
        max_queries_test=args.max_queries_test,
        epsilon=scenario.get("optimization", {}).get("epsilon", 0.01),
        verbose=True
    )

    # Run optimization
    print("\nStarting 2D grid search optimization...")
    result = optimizer.optimize()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "2d_grid_search_optimized_cv",
        "hyperparameters": {
            "alpha_star": result.alpha_star,
            "rrf_k_star": result.rrf_k_star,
            "k_fixed": result.k_fixed
        },
        "search_space": {
            "alpha_grid": result.config["alpha_grid"],
            "rrf_k_grid": result.config["rrf_k_grid"],
            "total_combinations": len(result.config["alpha_grid"]) * len(result.config["rrf_k_grid"])
        },
        "cv_performance": result.cv_performance,
        "fold_results": result.fold_results,
        "all_config_records": result.all_config_records,
        "final_test_metrics": result.final_test_results["metrics_summary"],
        "config": result.config,
        "methodology": {
            "description": "Optimized stratified cross-validation maximizing data usage",
            "optimization_folds": cv_info["metadata"]["optimization_folds"],
            "samples_per_fold": cv_info["metadata"]["samples_per_optimization_fold"],
            "final_test_samples": cv_info["metadata"]["samples_final_test"],
            "total_evaluations": len(result.config["alpha_grid"]) * len(result.config["rrf_k_grid"]) * cv_info["metadata"]["optimization_folds"]
        }
    }

    output_file = output_dir / f"2d_optimization_alpha_rrfk_k{k_fixed}.json"
    save_json(summary, str(output_file))

    print(f"\n{'=' * 70}")
    print("2D GRID SEARCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"Optimal configuration:")
    print(f"  α* = {result.alpha_star:.2f}")
    print(f"  rrf_k* = {result.rrf_k_star}")
    print(f"  k (fixed) = {result.k_fixed}")
    print(
        f"\nCV performance: {result.cv_performance['mean']:.4f} ± {result.cv_performance['std']:.4f}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
