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
├── 🔍 optimize_2d_grid_alpha_rrfk.py   # Alpha and rff_k parameter optimization
├── 📊 stratification.py                # Dataset stratification
├── 🤖 llm_judge.py                     # LLM-as-Judge evaluation framework
├── 📝 generate_ground_truth.py         # Ground truth generation with RAG pipeline
└── 📋 llm_as_judge_eval.py             # Batch LLM-as-Judge evaluation
```

## 🚀 Quick Start


### 1. Run Experiments
```bash
# Run Experiment 1 (Dense vs Sparse comparison)
python benchmarks/experiment1.py

# Run 2D grid optimization for alpha and RRF-K
python benchmarks/optimize_2d_grid_alpha_rrfk.py \
  --scenario-yaml benchmark_scenarios/experiment_2/your_scenario.yml \
  --dataset-path datasets/sosum/data \
  --output-dir results/grid_search
```

### 2. LLM-as-Judge Evaluation

#### Step 1: Generate Ground Truth
Generate answers using your RAG pipeline for evaluation:

```bash
python benchmarks/generate_ground_truth.py
```

This will:
- Load all questions from the SOSUM dataset
- Run the RAG pipeline (retrieval + generation) for each question
- Save results to JSON with retrieved context and generated answers
- Output includes Self-RAG statistics (iterations, convergence, hallucination corrections)

**Output**: `results/<experiment>/ground_truth_intermediate.json`

#### Step 2: Evaluate with LLM-as-Judge
Evaluate the generated answers using an LLM judge:

```bash
python benchmarks/llm_as_judge_eval.py
```

**Configuration** (edit in script):
```python
PROVIDER = "openai"       # openai | anthropic
MODEL_NAME = "gpt-4o"     # gpt-4o, gpt-4o-mini, claude-3-5-sonnet-20241022
INPUT_PATH = "results/test_self_rag/ground_truth_intermediate.json"
OUTPUT_PATH = "results/llm_judge_scores/llm_judge_scores.jsonl"
```

**Evaluation Dimensions** (1-5 scale):
- **Faithfulness**: Answer is grounded in provided context
- **Relevance**: Answer directly addresses the question
- **Helpfulness**: Answer is clear, complete, and actionable

**Output**: JSONL file with scores and justifications for each question
```


## 📊 Supported Metrics

### Information Retrieval Metrics
- **Recall@K**: Proportion of relevant items retrieved in top K
- **Precision@K**: Proportion of retrieved items that are relevant
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant item
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality with position discount

### LLM-as-Judge Metrics
- **Faithfulness** (1-5): Answer is grounded in provided context without hallucinations
- **Relevance** (1-5): Answer directly addresses the question asked
- **Helpfulness** (1-5): Answer is clear, complete, and actionable

### Efficiency Metrics
- **Query Latency**: Average time per query

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



## 🎯 Best Practices

1. **Stratified Sampling**: Use representative samples for reliable results
2. **Statistical Testing**: Always check for statistical significance
3. **Multiple Runs**: Average results across multiple runs for stability
4. **Baseline Comparison**: Include simple baselines (e.g., BM25)
5. **Error Analysis**: Examine failed queries to understand limitations
6. **Resource Monitoring**: Track memory and compute usage
7. **Reproducibility**: Set random seeds and document environment



This benchmarking framework enables rigorous evaluation and optimization of your RAG system, ensuring peak performance across different datasets and use cases.
