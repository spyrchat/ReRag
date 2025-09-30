"""
5-fold CV tuning for (alpha, top_k) with objective F1@5.

- Uses dev split inside each optimization fold to select (alpha, top_k) by maximizing mean F1@5.
- Aggregates across optimization folds to pick (alpha*, top_k*).
- Evaluates once on the held-out final_test fold with (alpha*, top_k*).
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
import importlib
import argparse


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_grid_float(s: str) -> List[float]:
    s = s.strip()
    if ":" in s:
        start, stop, step = map(float, s.split(":"))
        n = int(np.floor((stop - start) / step + 0.5)) + 1
        vals = [start + i * step for i in range(n)]
        vals[-1] = stop
        return vals
    return [float(x) for x in s.split(",") if x.strip()]


def parse_grid_int(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def set_alpha_topk(cfg: Dict[str, Any], alpha: float, top_k: int) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("retrieval", {})
    c["retrieval"]["top_k"] = int(top_k)
    c["retrieval"]["alpha"] = float(alpha)
    c["retrieval"].setdefault("fusion", {})
    c["retrieval"]["fusion"]["alpha"] = float(alpha)
    if "sparse" in c["retrieval"] and isinstance(c["retrieval"]["sparse"], dict):
        c["retrieval"]["sparse"]["alpha"] = float(alpha)
    c.setdefault("evaluation", {}).setdefault("k_values", [])
    if 5 not in c["evaluation"]["k_values"]:
        c["evaluation"]["k_values"].append(5)
    return c


def build_adapter(module_path: str, class_name: str, kwargs_json: Optional[str], qdrant_client=None, collection_name=None) -> Any:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if qdrant_client is not None:
        kwargs["qdrant_client"] = qdrant_client
    if collection_name is not None:
        kwargs["collection_name"] = collection_name
    adapter = cls(**kwargs)
    # Warn if qdrant_client or collection_name are missing
    if not hasattr(adapter, "qdrant_client") or getattr(adapter, "qdrant_client", None) is None:
        print("[WARNING] Adapter instance is missing qdrant_client. This may cause errors if required.")
    if not hasattr(adapter, "collection_name") or getattr(adapter, "collection_name", None) is None:
        print("[WARNING] Adapter instance is missing collection_name. This may cause errors if required.")
    return adapter


class SplitFilteringAdapter:
    """
    Wrap a base BenchmarkAdapter and filter queries by a given set of allowed IDs.
    Also ensures qdrant_client and collection_name are forwarded if present.
    """

    def __init__(self, base_adapter: Any, allowed_query_ids: Optional[set], name_suffix: str = ""):
        self.base = base_adapter
        self.allowed = None if allowed_query_ids is None else set(
            str(x) for x in allowed_query_ids)
        self.name = getattr(base_adapter, "name", "adapter") + \
            (f"-{name_suffix}" if name_suffix else "")
        # Forward qdrant_client and collection_name if present
        if hasattr(base_adapter, "qdrant_client"):
            self.qdrant_client = base_adapter.qdrant_client
        if hasattr(base_adapter, "collection_name"):
            self.collection_name = base_adapter.collection_name

    def load_queries(self, *args, **kwargs):
        # Ensure qdrant_client and collection_name are forwarded to base adapter if missing
        if hasattr(self, "qdrant_client") and not hasattr(self.base, "qdrant_client"):
            self.base.qdrant_client = self.qdrant_client
        if hasattr(self, "collection_name") and not hasattr(self.base, "collection_name"):
            self.base.collection_name = self.collection_name
        queries = self.base.load_queries(*args, **kwargs)
        if self.allowed is None:
            return queries
        return [q for q in queries if str(q.query_id) in self.allowed]


class AlphaTopkTunerF1At5:
    """
    Grid search over (alpha, top_k) with 5-fold CV.
    Objective: maximize mean F1@5 on dev of each optimization fold.
    Aggregate across optimization folds to select (alpha*, top_k*).
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        base_adapter: Any,
        cv_info: Dict[str, Any],
        alpha_grid: List[float],
        topk_grid: List[int],
        max_queries_dev: Optional[int] = None,
        max_queries_test: Optional[int] = None,
        epsilon: float = 0.0,
        prefer_smaller_topk: bool = True,
        latency_tiebreak: bool = True,
        verbose: bool = True
    ):
        self.base_config = base_config
        self.base_adapter = base_adapter
        self.cv_info = cv_info
        self.alpha_grid = alpha_grid
        self.topk_grid = topk_grid
        self.max_queries_dev = max_queries_dev
        self.max_queries_test = max_queries_test
        self.epsilon = float(epsilon)
        self.prefer_smaller_topk = prefer_smaller_topk
        self.latency_tiebreak = latency_tiebreak
        self.verbose = verbose

    def _fold_keys(self) -> List[str]:
        return list(self.cv_info["splits"].keys())

    def _fold_role(self, fold_key: str) -> str:
        return self.cv_info["splits"][fold_key]["role"]

    def _ids_for(self, fold_key: str, split: str) -> List[str]:
        return self.cv_info["splits"][fold_key][split]

    def tune(self) -> Dict[str, Any]:
        metric_key = "f1@5"
        fold_dev_bests: List[Dict[str, Any]] = []
        pair_scores: Dict[Tuple[float, int], List[float]] = {}
        pair_times: Dict[Tuple[float, int], List[float]] = {}
        for fold_key in self._fold_keys():
            if self._fold_role(fold_key) != "optimization":
                continue
            dev_ids = self._ids_for(fold_key, "dev")
            if self.verbose:
                print(
                    f"\n[Fold {fold_key}] Dev size: {len(dev_ids)} | Objective: {metric_key}")
            dev_adapter = SplitFilteringAdapter(
                self.base_adapter, set(dev_ids), name_suffix=f"{fold_key}-dev")
            best_score = -np.inf
            best_pair: Tuple[Optional[float], Optional[int]] = (None, None)
            for k in self.topk_grid:
                for alpha in self.alpha_grid:
                    cfg = set_alpha_topk(self.base_config, alpha, k)
                    runner = BenchmarkRunner(cfg)
                    aggregated = runner.run_benchmark(
                        dev_adapter, max_queries=self.max_queries_dev)
                    score = aggregated["metrics"][metric_key]["mean"] if metric_key in aggregated["metrics"] else float(
                        "nan")
                    rt_ms = aggregated["performance"]["mean"] if "performance" in aggregated and "mean" in aggregated["performance"] else float(
                        "inf")
                    pair_scores.setdefault((alpha, k), []).append(score)
                    pair_times.setdefault((alpha, k), []).append(rt_ms)
                    if self.verbose:
                        print(
                            f"  alpha={alpha:.2f} | top_k={k:>3} | {metric_key}={score:.4f} | rt={rt_ms:.1f}ms")
                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_pair = (alpha, k)
            if best_pair[0] is None:
                raise RuntimeError(
                    f"No valid {metric_key} on dev for {fold_key}. Check ground truth/K-values.")
            fold_dev_bests.append({
                "fold": fold_key,
                "alpha": float(best_pair[0]),
                "top_k": int(best_pair[1]),
                metric_key: float(best_score)
            })
            if self.verbose:
                print(
                    f"[Fold {fold_key}] Best on dev: alpha={best_pair[0]:.2f}, top_k={best_pair[1]} -> {metric_key}={best_score:.4f}")
        records: List[Dict[str, Any]] = []
        for (alpha, k), scores in pair_scores.items():
            vals = [s for s in scores if not np.isnan(s)]
            if not vals:
                continue
            mean_s, std_s = float(np.mean(vals)), float(np.std(vals))
            mean_t = float(np.mean(pair_times[(alpha, k)])) if pair_times.get(
                (alpha, k)) else float("inf")
            records.append({"alpha": float(alpha), "top_k": int(
                k), "mean": mean_s, "std": std_s, "rt_ms": mean_t})
        if not records:
            raise RuntimeError("No valid (alpha, top_k) records across folds.")
        best_mean = max(r["mean"] for r in records)
        candidates = [r for r in records if (
            best_mean - r["mean"]) <= self.epsilon * abs(best_mean)]
        if self.prefer_smaller_topk:
            candidates.sort(key=lambda r: (r["top_k"],))
        if self.latency_tiebreak:
            candidates.sort(key=lambda r: (r["rt_ms"],))
        candidates.sort(key=lambda r: (abs(r["alpha"] - 0.5),))
        winner = sorted(
            candidates, key=lambda r: (-r["mean"], r["top_k"], r["rt_ms"], abs(r["alpha"] - 0.5)))[0]
        alpha_star, k_star = float(winner["alpha"]), int(winner["top_k"])
        if self.verbose:
            print(f"\nSelected (alpha*, top_k*) => ({alpha_star:.2f}, {k_star}) | "
                  f"F1@5 mean={winner['mean']:.4f} ± {winner['std']:.4f} | rt={winner['rt_ms']:.1f}ms")
        final_fold_key = None
        for fk in self._fold_keys():
            if self._fold_role(fk) == "final_test":
                final_fold_key = fk
                break
        if final_fold_key is None:
            raise RuntimeError("No 'final_test' fold found in cv_info.")
        test_ids = self._ids_for(final_fold_key, "test")
        test_adapter = SplitFilteringAdapter(self.base_adapter, set(
            test_ids), name_suffix=f"{final_fold_key}-test")
        final_cfg = set_alpha_topk(self.base_config, alpha_star, k_star)
        eval_k = final_cfg.setdefault(
            "evaluation", {}).setdefault("k_values", [])
        for kv in (5, 10, 20):
            if kv not in eval_k:
                eval_k.append(kv)
        final_runner = BenchmarkRunner(final_cfg)
        final_agg = final_runner.run_benchmark_with_individual_results(
            test_adapter, max_queries=self.max_queries_test
        )
        return {
            "fold_dev_bests": fold_dev_bests,
            "alpha_star": alpha_star,
            "top_k_star": k_star,
            "epsilon": self.epsilon,
            "final_test_fold": final_fold_key,
            "final_test_aggregated": final_agg,
            "search_records": records
        }


def main():
    p = argparse.ArgumentParser(
        description="5-fold CV tuner for (alpha, top_k) with F1@5 objective (hybrid RAG pipeline)."
    )
    p.add_argument("--scenario-yaml", required=True,
                   help="Path to scenario YAML (declares RAG pipeline and grid space). E.g. benchmark_scenarios/experiment2/hybrid_rag.yml")
    p.add_argument("--dataset-path", required=True,
                   help="Path to dataset root (expects question.csv)")
    p.add_argument("--adapter-module", required=True,
                   help="Module path for BenchmarkAdapter implementation.")
    p.add_argument("--adapter-class", required=True,
                   help="Class name for BenchmarkAdapter.")
    p.add_argument("--adapter-kwargs", default=None,
                   help='JSON kwargs for adapter ctor, e.g. {"dataset_root": "..."}')
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of folds for CV (default: 5).")
    p.add_argument("--max-queries-dev", type=int, default=None,
                   help="Optional cap on dev queries per fold.")
    p.add_argument("--max-queries-test", type=int, default=None,
                   help="Optional cap on test queries.")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="ε-window on F1@5 for tie-breaking (fraction, e.g. 0.005).")
    p.add_argument("--save-summary", default="tuning_summary_f1at5.json",
                   help="Path to save final JSON summary.")
    args = p.parse_args()

    scenario = load_yaml(args.scenario_yaml)
    base_cfg = scenario["pipeline"]

    # Parse grid from scenario YAML (can be list or string for ranges)
    def parse_grid(val, parse_fn):
        if isinstance(val, str):
            return parse_fn(val)
        if isinstance(val, list):
            return val
        raise ValueError(f"Invalid grid value: {val}")

    alpha_grid = parse_grid(scenario["grid"]["alpha"], parse_grid_float)
    topk_grid = parse_grid(scenario["grid"]["top_k"], parse_grid_int)

    splitter = StratifiedRAGDatasetSplitter(
        dataset_path=args.dataset_path, random_state=42)
    splitter.load_dataset()
    cv_info = splitter.create_cv_splits(n_folds=args.n_folds)

    qdrant_cfg = base_cfg["retrieval"]["qdrant"]
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(host=qdrant_cfg.get(
        "host", "localhost"), port=qdrant_cfg.get("port", 6333))
    collection_name = qdrant_cfg["collection_name"]

    base_adapter = build_adapter(
        args.adapter_module, args.adapter_class, args.adapter_kwargs,
        qdrant_client=qdrant_client, collection_name=collection_name
    )

    # Local import to avoid circular
    from benchmarks.grid_search import AlphaTopkTunerF1At5
    tuner = AlphaTopkTunerF1At5(
        base_config=base_cfg,
        base_adapter=base_adapter,
        cv_info=cv_info,
        alpha_grid=alpha_grid,
        topk_grid=topk_grid,
        max_queries_dev=args.max_queries_dev,
        max_queries_test=args.max_queries_test,
        epsilon=args.epsilon,
        prefer_smaller_topk=True,
        latency_tiebreak=True,
        verbose=True
    )
    summary = tuner.tune()
    save_json(summary, args.save_summary)
    print("\n=== Final selection ===")
    print(json.dumps({
        "alpha_star": summary["alpha_star"],
        "top_k_star": summary["top_k_star"],
        "final_test_fold": summary["final_test_fold"],
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved summary to: {args.save_summary}")


if __name__ == "__main__":
    main()
