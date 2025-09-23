# AutoRAG Multi-Arm Bandit Optimization

This implementation provides hyperparameter optimization for your modular RAG system using multi-arm bandit algorithms.

## Key Features

### 🎯 Multi-Arm Bandit Algorithms
- **UCB (Upper Confidence Bound)**: Balances exploration/exploitation with confidence bounds
- **Thompson Sampling**: Bayesian approach using posterior sampling
- **Epsilon-Greedy**: Simple exploration with configurable decay

### 🔧 Modular Integration
- Leverages your existing `BenchmarkOptimizer` for evaluation
- Uses your configuration system (`config.yml`, benchmark scenarios)
- Supports grid search, random search, and predefined configurations
- Integrates with your component architecture (retrievers, rerankers, embeddings)

### 📊 Smart Optimization
- Early stopping to prevent overfitting
- Convergence tracking and visualization
- Support for different search spaces per component
- Automatic configuration merging

## Quick Start

### 1. Basic Usage
```bash
# Run optimization with default settings
python benchmarks/run_autorag_optimization.py

# Use specific algorithm and search space
python benchmarks/run_autorag_optimization.py \
    --algorithm ucb \
    --space-name grid_search_space \
    --max-iterations 100
```

### 2. Algorithm-Specific Examples

**UCB Algorithm (Best for balanced exploration/exploitation):**
```bash
python benchmarks/run_autorag_optimization.py \
    --algorithm ucb \
    --ucb-confidence 2.0 \
    --space-name retrieval_focused_space
```

**Thompson Sampling (Best for Bayesian optimization):**
```bash
python benchmarks/run_autorag_optimization.py \
    --algorithm thompson \
    --space-name random_search_space
```

**Epsilon-Greedy (Best for simple exploration):**
```bash
python benchmarks/run_autorag_optimization.py \
    --algorithm epsilon_greedy \
    --epsilon 0.1 \
    --epsilon-decay 0.01
```

### 3. Component-Specific Optimization

**Focus on Retrieval:**
```bash
python benchmarks/run_autorag_optimization.py \
    --space-name retrieval_focused_space \
    --max-iterations 50
```

**Focus on Reranking:**
```bash
python benchmarks/run_autorag_optimization.py \
    --space-name reranking_focused_space \
    --algorithm thompson
```

**Focus on Embeddings:**
```bash
python benchmarks/run_autorag_optimization.py \
    --space-name embedding_focused_space \
    --algorithm ucb
```

## Search Space Configuration

### Grid Search Example
```yaml
grid_search_space:
  search_type: grid
  parameters:
    benchmark.retrieval.top_k: [10, 20, 50]
    benchmark.retrieval.strategy: ["dense", "hybrid"]
    reranker.enabled: [true, false]
```

### Random Search Example
```yaml
random_search_space:
  search_type: random
  n_random_arms: 100
  parameters:
    benchmark.retrieval.search_params.score_threshold:
      type: uniform
      low: 0.0
      high: 0.5
    benchmark.retrieval.top_k: [5, 10, 20, 50, 100]
```

### Predefined Configurations
```yaml
predefined_search_space:
  search_type: predefined
  predefined_configs:
    - name: "high_recall_config"
      parameters:
        benchmark.retrieval.top_k: 100
        reranker.enabled: true
```

## Advanced Usage

### Custom Optimization Workflow
```python
from benchmarks.autorag_bandit_optimizer import AutoRAGBanditOptimizer, UCBAlgorithm

# Create custom optimizer
optimizer = AutoRAGBanditOptimizer(
    base_config_path="config.yml",
    hyperparameter_space_path="custom_space.yml",
    algorithm=UCBAlgorithm(confidence_level=2.5),
    max_iterations=200,
    early_stopping_patience=15
)

# Run optimization
result = optimizer.optimize()

# Access results
print(f"Best config: {result.best_arm.config}")
print(f"Performance: {result.final_performance}")
```

### Integration with Existing Benchmarks
The optimizer automatically integrates with your existing benchmark scenarios:
- Uses your `BenchmarkOptimizer` for evaluation
- Leverages existing scenario configurations
- Merges with your base `config.yml`
- Supports all your current metrics (precision, recall, F1, MRR, NDCG)

## Output Analysis

The optimizer generates:
1. **YAML Results**: Complete optimization results with best configuration
2. **CSV History**: Iteration-by-iteration convergence data
3. **Performance Metrics**: Detailed evaluation of best configuration

### Example Output Structure
```yaml
algorithm: "UCB-2.0"
best_configuration:
  arm_id: "arm_123456"
  config:
    benchmark.retrieval.top_k: 50
    reranker.enabled: true
  avg_reward: 0.8234
  num_pulls: 15
optimization_summary:
  total_iterations: 75
  total_evaluation_time: 1234.56
  final_performance:
    f1: {1: 0.7234, 5: 0.8234, 10: 0.8456}
    precision: {1: 0.9123, 5: 0.8567, 10: 0.7890}
```

## Benefits of Multi-Arm Bandit Approach

### vs Grid Search
- **Efficiency**: Doesn't waste time on poor configurations
- **Adaptivity**: Focuses on promising regions automatically
- **Early Stopping**: Natural convergence detection

### vs Random Search  
- **Intelligence**: Learns from previous evaluations
- **Guidance**: Balances exploration with exploitation
- **Efficiency**: Converges faster to good solutions

### vs Manual Tuning
- **Systematic**: Principled exploration strategy
- **Scalable**: Handles large parameter spaces
- **Reproducible**: Consistent optimization process

## Performance Tips

1. **Start Small**: Begin with predefined configurations to validate setup
2. **Use UCB**: Generally best balance of performance and interpretability
3. **Component Focus**: Optimize one component at a time for better insights
4. **Early Stopping**: Set appropriate patience to avoid overoptimization
5. **Multiple Runs**: Run with different algorithms and compare results

## Monitoring and Debugging

The optimizer provides detailed logging:
- Iteration progress and arm selection
- Reward values and performance metrics
- Exploration/exploitation balance
- Early stopping triggers

Use the convergence history to analyze:
- Learning curves
- Exploration patterns
- Algorithm performance
- Hyperparameter sensitivity
