# Cross-Validation Split Usage in RAG Optimization

## Overview

This document explains how the train/dev/test splits are used in your RAG hyperparameter optimization pipeline.

---

## ⚠️ IMPORTANT: No Training in RAG

**Key Point**: Your RAG system does **NOT require training**. Therefore, the traditional machine learning train/dev/test paradigm needs to be adapted.

### Traditional ML Pipeline
```
Train Set (60%) → Train model weights/parameters
Dev Set (20%)   → Tune hyperparameters
Test Set (20%)  → Final unbiased evaluation
```

### Your RAG Pipeline (Updated)
```
Train+Dev Set (80%) → Evaluate hyperparameter configurations
Test Set (20%)      → Final unbiased evaluation
```

---

## Current 5-Fold Structure

### Fold 0-3 (Optimization Folds)
Each fold is split 80/20:
- **Train+Dev (80%)**: Used to evaluate (α, rrf_k) configurations
- **Test (20%)**: Held out (not used during optimization)

### Fold 4 (Final Test Fold)
- **Train+Dev (0%)**: Not created (fold is 100% test)
- **Test (100%)**: Used only for final evaluation of selected configuration

---

## How Data Flows Through Optimization

### Stage 1: Hyperparameter Search (Folds 0-3)

```python
For each optimization fold (0, 1, 2, 3):
    # Get train+dev queries (80% of fold)
    train_ids = fold.get_train_ids()
    dev_ids = fold.get_dev_ids()
    eval_ids = train_ids + dev_ids  # Merge: 60% + 20% = 80%
    
    # Test ALL (alpha, rrf_k) combinations on this 80%
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for rrf_k in [30, 60, 90, 120, 150]:
            # Evaluate configuration on 80% of fold
            score = evaluate_config(alpha, rrf_k, eval_ids)
            
            # Track score across folds
            config_scores[(alpha, rrf_k)].append(score)
```

**What happens**:
1. Each of 30 configurations is tested on **Fold 0's 80%**
2. Best config for Fold 0 is recorded
3. Process repeats for Folds 1, 2, 3
4. Each config ends up with **4 scores** (one per optimization fold)

### Stage 2: Configuration Selection

```python
# Aggregate scores across 4 optimization folds
for config in all_configs:
    mean_score = mean(config.scores_from_folds_0_1_2_3)
    std_score = std(config.scores_from_folds_0_1_2_3)
    
# Select configuration with highest mean score
winner = max(configs, key=lambda c: c.mean_score)
alpha_star = winner.alpha
rrf_k_star = winner.rrf_k
```

**What happens**:
- Each config's 4 fold scores are averaged
- Configuration with highest average wins
- Std deviation measures stability across folds

### Stage 3: Final Test Evaluation

```python
# Evaluate winner on held-out Fold 4 (100% test, never seen before)
final_test_ids = fold_4.get_test_ids()  # All queries in fold 4

final_performance = evaluate_config(
    alpha=alpha_star,
    rrf_k=rrf_k_star,
    query_ids=final_test_ids  # Unbiased test set
)
```

**What happens**:
- Selected (α*, rrf_k*) is tested on **Fold 4** (never used in optimization)
- Provides unbiased estimate of real-world performance
- This is the number you report in your thesis!

---

## Example with Numbers

Assume dataset has **500 queries total**:

### Fold 0 (Optimization)
- **Train: 300 queries (60%)** → Used for config evaluation
- **Dev: 100 queries (20%)** → Used for config evaluation  
- **Test: 100 queries (20%)** → Held out (unused)
- **Evaluation set: 400 queries (train+dev = 80%)**

### Fold 1 (Optimization)
- **Train: 300 queries (60%)** → Used for config evaluation
- **Dev: 100 queries (20%)** → Used for config evaluation
- **Test: 100 queries (20%)** → Held out (unused)
- **Evaluation set: 400 queries (train+dev = 80%)**

### Fold 2 (Optimization)
- **Train: 300 queries (60%)** → Used for config evaluation
- **Dev: 100 queries (20%)** → Used for config evaluation
- **Test: 100 queries (20%)** → Held out (unused)
- **Evaluation set: 400 queries (train+dev = 80%)**

### Fold 3 (Optimization)
- **Train: 300 queries (60%)** → Used for config evaluation
- **Dev: 100 queries (20%)** → Used for config evaluation
- **Test: 100 queries (20%)** → Held out (unused)
- **Evaluation set: 400 queries (train+dev = 80%)**

### Fold 4 (Final Test)
- **Train: 0 queries (0%)** → N/A
- **Dev: 0 queries (0%)** → N/A
- **Test: 100 queries (100%)** → Used for final evaluation only

**Total evaluations per config**: 4 folds × 400 queries = **1,600 query evaluations**

---

## Why Use Train+Dev Together?

### ✅ **Advantages**

1. **More data for evaluation**
   - 80% vs 20% = 4× more queries per fold
   - Reduces variance in score estimates
   - More reliable configuration selection

2. **Better statistical power**
   - More samples → tighter confidence intervals
   - Easier to distinguish between configurations
   - More representative of full dataset

3. **No wasted data**
   - Train set is actually used (not sitting idle)
   - Maximizes use of available queries

4. **Appropriate for non-trainable systems**
   - RAG doesn't need training, so train/dev distinction is artificial
   - Only test set needs to be held out

### ⚠️ **Why NOT Use Test Set?**

**Never use Fold 0-3 test sets during optimization!**

Reasons:
1. **Data leakage**: Would bias configuration selection
2. **Overfitting**: Configs would be optimized to test sets
3. **Invalid evaluation**: Final test (Fold 4) would be contaminated

The test portions of Folds 0-3 remain **completely unused** and that's intentional!

---

## Comparison: Before vs After Update

### Before (Only Dev Set)
```
Fold 0: [UNUSED 60%] | [USE 20%] | [UNUSED 20%]
Fold 1: [UNUSED 60%] | [USE 20%] | [UNUSED 20%]
Fold 2: [UNUSED 60%] | [USE 20%] | [UNUSED 20%]
Fold 3: [UNUSED 60%] | [USE 20%] | [UNUSED 20%]
Fold 4: [UNUSED 0%]  | [UNUSED 0%] | [USE 100%]

Data utilization per fold: 20% for optimization
Total queries per config: 4 × 20% = 0.8× dataset size
```

### After (Train+Dev Combined)
```
Fold 0: [USE 60%] | [USE 20%] | [UNUSED 20%]
Fold 1: [USE 60%] | [USE 20%] | [UNUSED 20%]
Fold 2: [USE 60%] | [USE 20%] | [UNUSED 20%]
Fold 3: [USE 60%] | [USE 20%] | [UNUSED 20%]
Fold 4: [UNUSED 0%] | [UNUSED 0%] | [USE 100%]

Data utilization per fold: 80% for optimization
Total queries per config: 4 × 80% = 3.2× dataset size
```

**Result**: 4× more data per evaluation = more reliable optimization!

---

## When Would You Actually Use Separate Train/Dev?

You would keep train/dev separate if:

### 1. **Training a Reranker**
```python
# Train on TRAIN set
reranker = CrossEncoderReranker()
reranker.train(train_queries, train_relevance_labels)

# Tune hyperparameters on DEV set
for learning_rate in [1e-5, 1e-4, 1e-3]:
    reranker.set_lr(learning_rate)
    score = reranker.evaluate(dev_queries)
```

### 2. **Learning Fusion Weights**
```python
# Learn weights on TRAIN set
optimal_weights = learn_weights(train_queries)

# Evaluate different weight combinations on DEV set
for w_dense, w_sparse in weight_combinations:
    score = evaluate_fusion(dev_queries, w_dense, w_sparse)
```

### 3. **Fitting Threshold Parameters**
```python
# Learn optimal threshold on TRAIN set
optimal_threshold = fit_threshold(train_queries)

# Validate threshold on DEV set
performance = evaluate_with_threshold(dev_queries, optimal_threshold)
```

**For pure hyperparameter tuning (no training)**: Use train+dev together!

---

## Impact on Your Results

### Statistical Reliability
- **Before**: Each config evaluated on ~100 queries per fold (20%)
- **After**: Each config evaluated on ~400 queries per fold (80%)
- **Improvement**: Standard errors reduced by ~2× (√4 = 2)

### Confidence Intervals
- **Before**: 95% CI = ±1.96 × SE with n=100 → ±19.6% SE
- **After**: 95% CI = ±1.96 × SE with n=400 → ±9.8% SE
- **Improvement**: Confidence intervals are **half as wide**!

### Practical Impact
- More stable fold-to-fold scores
- Easier to distinguish between configurations
- Higher confidence in selected optimal configuration
- Better thesis results! 📊

---

## Summary

### What Each Split Does

| Split | Folds 0-3 (Optimization) | Fold 4 (Final Test) |
|-------|--------------------------|---------------------|
| **Train** | ✅ Used for config eval | ❌ N/A (fold is 100% test) |
| **Dev** | ✅ Used for config eval | ❌ N/A (fold is 100% test) |
| **Test** | ❌ Held out (unused) | ✅ Used for final unbiased eval |

### Key Takeaways

1. ✅ **Train+Dev are merged** for hyperparameter evaluation (80% per fold)
2. ✅ **Test sets in Folds 0-3 are ignored** (prevent data leakage)
3. ✅ **Fold 4 test set is used only once** (final unbiased evaluation)
4. ✅ **No training happens** (RAG is parameterized, not trained)
5. ✅ **More data = better optimization** (4× more queries per evaluation)

### For Your Thesis

Report:
- "5-fold cross-validation with 80/20 optimization/test split"
- "Each configuration evaluated on 4 folds (80% data per fold)"
- "Final test evaluation on held-out 5th fold (20% of dataset)"
- "Total: 3.2× dataset size worth of evaluations per configuration"

---

## References

- **Stratified K-Fold CV**: Kohavi, R. (1995). "A study of cross-validation"
- **Hyperparameter Optimization**: Bergstra & Bengio (2012). "Random search for hyper-parameter optimization"
- **Data Splitting Best Practices**: Raschka, S. (2018). "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning"
