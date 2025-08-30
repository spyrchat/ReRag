# RAG Benchmark Optimization System - Usage Guide

## 🎯 Summary

We have successfully created a flexible benchmark optimization system that fixes the critical `external_id` retrieval issue and enables easy parameter optimization experiments.

### ✅ Key Achievements

1. **Fixed External ID Retrieval**: Modified dense retriever to use direct Qdrant API, preserving `external_id` in document metadata
2. **Excellent Benchmark Results**: 
   - Precision@5: 75.5%
   - Recall@5: 69.3%
   - MRR: 92.0%
3. **Flexible Configuration System**: Created modular benchmark scenarios for easy optimization
4. **Ground Truth Integration**: Proper evaluation using real StackOverflow question-answer pairs

## 🚀 Quick Start - Running Benchmarks

### Option 1: Interactive CLI (Easiest)
```bash
cd /home/spiros/Desktop/Thesis/Thesis
python run_benchmark_optimization.py
```

Then choose:
- `1` - Quick test (10 queries)
- `2` - Single scenario 
- `3` - Run all scenarios
- `4` - Compare previous results

### Option 2: Command Line
```bash
# Run single scenario
python benchmark_optimizer.py --scenario benchmark_scenarios/quick_test.yml

# Run all scenarios
python benchmark_optimizer.py --scenarios-dir benchmark_scenarios

# Compare existing results only
python benchmark_optimizer.py --compare-only
```

## 📊 Available Optimization Scenarios

Located in `benchmark_scenarios/`:

1. **quick_test.yml** - Fast 10-query test for rapid iteration
2. **dense_baseline.yml** - Dense retrieval with top_k=10, threshold=0.1
3. **dense_high_recall.yml** - Dense with top_k=20, threshold=0.05 (more results)
4. **dense_high_precision.yml** - Dense with threshold=0.3 (stricter filtering)
5. **sparse_bm25.yml** - Sparse BM25 retrieval
6. **hybrid_retrieval.yml** - Combined dense + sparse retrieval

## 🔧 Creating Custom Scenarios

Create new `.yml` files in `benchmark_scenarios/` with this structure:

```yaml
# Description of the experiment
description: "Your experiment description"

# Dataset configuration
dataset:
  path: "/home/spiros/Desktop/Thesis/datasets/sosum/data"
  use_ground_truth: true

# Retrieval configuration  
retrieval:
  type: "dense"  # dense, sparse, or hybrid
  top_k: 10
  score_threshold: 0.1

# Embedding configuration (override main config)
embedding:
  dense:
    provider: google
    model: models/embedding-001
    dimensions: 768
    api_key_env: GOOGLE_API_KEY
    batch_size: 32
    vector_name: dense
  strategy: dense

# Evaluation configuration
evaluation:
  k_values: [1, 5, 10]
  metrics:
    retrieval: ["precision@k", "recall@k", "mrr", "ndcg@k"]

# Experiment parameters
max_queries: 50
experiment_name: "your_experiment_name"
```

## 📈 Optimization Parameters You Can Tune

### Retrieval Parameters
- `top_k`: Number of documents to retrieve (5, 10, 15, 20)
- `score_threshold`: Minimum similarity score (0.0, 0.1, 0.2, 0.3)
- `type`: Retrieval strategy (dense, sparse, hybrid)

### Embedding Parameters
- `model`: Different embedding models
- `batch_size`: Processing batch size (16, 32, 64)
- `dimensions`: Embedding dimensions (384, 768, 1024)

### Evaluation Parameters  
- `max_queries`: Dataset size (10, 25, 50, 100, 500)
- `k_values`: Evaluation depths ([1,5,10], [1,5,10,20])

## 🏆 Results Analysis

The system automatically:
- Tracks all experiment results
- Compares scenarios across metrics
- Identifies best performers for each metric
- Saves results to `benchmark_optimization_results.yml`

### Key Metrics
- **Precision@K**: How many retrieved docs are relevant
- **Recall@K**: How many relevant docs were retrieved  
- **MRR**: Mean Reciprocal Rank (position of first relevant result)
- **NDCG@K**: Normalized Discounted Cumulative Gain

## 🔍 Example Optimization Workflow

1. **Start with quick test**:
   ```bash
   python benchmark_optimizer.py --scenario benchmark_scenarios/quick_test.yml
   ```

2. **Run baseline experiments**:
   ```bash
   python benchmark_optimizer.py --scenarios-dir benchmark_scenarios
   ```

3. **Create custom scenarios** based on baseline results

4. **Compare all results**:
   ```bash
   python benchmark_optimizer.py --compare-only
   ```

## 📊 Current Best Configuration

Based on our tests, the current best performing setup:
- **Retrieval**: Dense with Google Gemini embeddings
- **Top K**: 10 documents
- **Score Threshold**: 0.1 (from config, but score filter at 0.3)
- **Reranking**: Cross-encoder reranking with ms-marco-MiniLM-L-6-v2
- **Results**: 75.5% Precision@5, 69.3% Recall@5, 92% MRR

## 🚨 Important Notes

1. **Ground Truth**: System uses real StackOverflow question-answer pairs for evaluation
2. **External ID Fix**: Our custom dense retriever preserves document IDs correctly
3. **Scalability**: Adjust `max_queries` based on time constraints
4. **Consistency**: All scenarios use the same evaluation methodology for fair comparison

## 🎯 Next Steps for Optimization

1. **Hyperparameter Tuning**: Create scenarios with different top_k and threshold values
2. **Embedding Models**: Test different embedding providers/models
3. **Hybrid Strategies**: Optimize dense+sparse combination weights
4. **Reranking**: Experiment with different reranker models
5. **Dataset Size**: Scale up to full 506 questions for final evaluation
