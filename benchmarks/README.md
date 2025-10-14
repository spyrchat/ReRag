# Benchmarks & Evaluation Framework

A comprehensive evaluation framework for RAG systems with support for multiple metrics, A/B testing, grid search optimization, and statistical analysis.

## 🎯 Overview

The benchmarks module provides:
- **Standardized metrics** (Recall@K, Precision@K, MRR, NDCG)
- **Grid search optimization** for hyperparameter tuning
- **Statistical analysis** with significance testing

## 📁 Module Structure

```
benchmarks/
├── 📖 README.md                        # This file
├── 🔧 benchmark_contracts.py           # Interfaces and data models
├── 🏃 benchmarks_runner.py             # Core benchmark execution
├── 📊 benchmarks_metrics.py            # Metrics computation
├── 🔌 benchmarks_adapters.py           # Dataset adapters for benchmarks
├── 📈 statistical_analyzer.py          # Advanced statistical analysis
├── 📋 report_generator.py              # HTML/PDF report generation
├── 📤 results_exporter.py              # Results export utilities
├── 🎛️ utils.py                         # Common utilities
│
├── 🧪 experiment1.py                   # Dense vs Sparse comparison
├── 🧪 experiment3.py                   # Hybrid optimization
├── 🔍 optimize_2d_grid_alpha_rrfk.py   # Alpha parameter optimization
├── 🏃 run_benchmark_optimization.py    # Grid search runner
├── 🏃 run_real_benchmark.py            # Real dataset benchmarks
└── 📊 stratification.py                # Dataset stratification
```

## 🚀 Quick Start

### 1. Basic Benchmark
```bash
# Run benchmark with StackOverflow dataset
python -m benchmarks.run_real_benchmark

# Configuration is hardcoded in the script.
# To customize, edit benchmarks/run_real_benchmark.py and modify:
# - config["retrieval"]["type"] = "dense" | "sparse" | "hybrid"
# - config["evaluation"]["k_values"] = [1, 5, 10]
```

### 2. Run Experiments
```bash
# Run Experiment 1 (Dense vs Sparse comparison)
python -m benchmarks.experiment1 --output-dir results/exp1

# Run Experiment 3 (Hybrid optimization)
python -m benchmarks.experiment3 --output-dir results/exp3 --test

# Run 2D grid optimization for alpha and RRF-K
python -m benchmarks.optimize_2d_grid_alpha_rrfk \
  --scenario-yaml benchmark_scenarios/your_scenario.yml \
  --dataset-path datasets/sosum/data \
  --n-folds 5 \
  --output-dir results/grid_search
```

### 3. Interactive Optimization
```bash
# Interactive benchmark optimizer (menu-driven)
python -m benchmarks.run_benchmark_optimization

# Follow the interactive prompts to:
# 1. Run quick test
# 2. Run single scenario
# 3. Run all scenarios
# 4. Compare previous results
```

## 📊 Supported Metrics

### Information Retrieval Metrics
- **Recall@K**: Proportion of relevant items retrieved in top K
- **Precision@K**: Proportion of retrieved items that are relevant
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant item
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality with position discount

### Efficiency Metrics
- **Query Latency**: Average time per query
- **Throughput**: Queries per second
- **Memory Usage**: Peak memory consumption
- **Index Size**: Storage requirements

### Statistical Tests
- **Paired t-test**: Compare two retrieval strategies
- **Wilcoxon signed-rank test**: Non-parametric alternative
- **Cohen's d**: Effect size measurement
- **Confidence intervals**: Statistical significance bounds

## 🎛️ Configuration

### Benchmark Configuration Example
```yaml
# benchmarks/configs/optimization.yml
benchmark:
  name: "hybrid_optimization"
  description: "Optimize alpha and RRF-K parameters for hybrid retrieval"
  
datasets:
  - name: "stackoverflow"
    adapter: "benchmarks.benchmarks_adapters.StackOverflowBenchmarkAdapter"
    path: "datasets/stackoverflow/"
    sample_size: 1000
    stratify_by: "difficulty"
  
retrieval:
  strategies: ["dense", "sparse", "hybrid"]
  parameters:
    alpha:
      min: 0.1
      max: 0.9
      step: 0.1
    rrfk:
      values: [10, 20, 30, 50, 100]
  
evaluation:
  metrics: ["recall@5", "recall@10", "mrr", "ndcg@10"]
  k_values: [1, 3, 5, 10, 15, 20]
  
output:
  format: ["json", "csv", "html"]
  charts: true
  statistical_tests: true
```

## 🏃‍♂️ Running Benchmarks

### 1. Dataset Preparation
```python
from benchmarks.benchmarks_adapters import StackOverflowBenchmarkAdapter
from pathlib import Path

# Prepare your dataset
adapter = StackOverflowBenchmarkAdapter(dataset_path=Path("datasets/stackoverflow/"))
queries = adapter.load_queries()
# Ground truth is optional for exploratory benchmarks
```

### 2. Configure Retrieval System
```python
from components.retrieval_pipeline import RetrievalPipelineFactory

# Create retrieval pipeline
config = {
    'embedding': {'strategy': 'hybrid'},
    'qdrant': {'collection': 'my_collection'}
}
pipeline = RetrievalPipelineFactory.create_pipeline(config)
```

### 3. Run Benchmark
```python
from benchmarks.benchmarks_runner import BenchmarkRunner

# Initialize and run benchmark
runner = BenchmarkRunner(config)
results = runner.run_benchmark(
    queries=queries,
    ground_truth=ground_truth,
    strategies=['dense', 'sparse', 'hybrid'],
    k_values=[1, 3, 5, 10]
)
```

### 4. Analyze Results
```python
from benchmarks.statistical_analyzer import StatisticalAnalyzer

# Perform statistical analysis
analyzer = StatisticalAnalyzer(results)
analysis = analyzer.compare_strategies(
    strategy_a="dense",
    strategy_b="hybrid",
    metric="recall@5"
)
print(f"p-value: {analysis['p_value']}")
print(f"effect_size: {analysis['effect_size']}")
```

## 📈 Experiments

### Experiment 1: Dense vs Sparse Retrieval
Compares dense (vector) vs sparse (keyword) retrieval.

```bash
# Run experiment with StackOverflow dataset
python -m benchmarks.experiment1

# To customize configuration, edit experiment1.py
# Currently uses hardcoded dataset and parameters
```

### Experiment 3: Hybrid Optimization
Optimizes hybrid retrieval parameters (alpha, RRF-K) using grid search.

```bash
# Run hybrid optimization experiment
python -m benchmarks.experiment3

# For 2D grid search of alpha and RRF-K parameters:
python -m benchmarks.optimize_2d_grid_alpha_rrfk

# To customize alpha ranges and RRF-K values, edit the script
```

### Custom Experiments
Create your own experiments by extending the base classes:

```python
from benchmarks.benchmark_contracts import BenchmarkExperiment

class MyCustomExperiment(BenchmarkExperiment):
    def setup(self):
        """Configure experiment parameters"""
        pass
    
    def run(self):
        """Execute experiment logic"""
        pass
    
    def analyze(self):
        """Analyze and report results"""
        pass
```

## 📊 Results & Reporting

### Result Formats
- **JSON**: Machine-readable detailed results
- **CSV**: Tabular data for spreadsheet analysis
- **HTML**: Interactive dashboards with charts
- **PDF**: Publication-ready reports

### Generated Reports Include:
- **Performance Summary**: Key metrics across strategies
- **Statistical Analysis**: Significance tests and effect sizes
- **Visualizations**: Charts showing performance trends
- **Recommendations**: Optimal parameter suggestions

### Example Report Generation
Reports are automatically generated by experiment scripts. To use the report generator programmatically:

```python
from benchmarks.report_generator import BenchmarkReportGenerator

# Create report generator
generator = BenchmarkReportGenerator(test_mode=False)

# Print scenario summary
generator.print_scenario_summary(scenario_name="my_experiment", result=results)

# Print statistical report
generator.print_statistical_report(statistical_results)
```

## 🔧 Advanced Features

### Stratified Sampling
Ensure representative evaluation across different query types:

```python
from benchmarks.stratification import StratifiedSampler

sampler = StratifiedSampler()
stratified_queries = sampler.sample(
    queries=all_queries,
    strata_column="difficulty",
    sample_size=1000,
    proportional=True
)
```

### Statistical Comparison
Compare retrieval strategies using statistical analysis:

```python
from benchmarks.statistical_analyzer import StatisticalAnalyzer

# Use the existing statistical analyzer for comparisons
analyzer = StatisticalAnalyzer(benchmark_results)
comparison = analyzer.compare_strategies(
    strategy_a="dense",
    strategy_b="hybrid",
    metric="recall@5"
)

print(f"p-value: {comparison['p_value']}")
print(f"Significant difference: {comparison['is_significant']}")
```

### Custom Metrics
Add domain-specific evaluation metrics:

```python
from benchmarks.benchmarks_metrics import BenchmarkMetrics

class CustomMetrics(BenchmarkMetrics):
    def code_relevance_score(self, retrieved_docs, query):
        """Custom metric for code-related queries"""
        # Implementation here
        pass
```



## 🎯 Best Practices

1. **Stratified Sampling**: Use representative samples for reliable results
2. **Statistical Testing**: Always check for statistical significance
3. **Multiple Runs**: Average results across multiple runs for stability
4. **Baseline Comparison**: Include simple baselines (e.g., BM25)
5. **Error Analysis**: Examine failed queries to understand limitations
6. **Resource Monitoring**: Track memory and compute usage
7. **Reproducibility**: Set random seeds and document environment



This benchmarking framework enables rigorous evaluation and optimization of your RAG system, ensuring peak performance across different datasets and use cases.
