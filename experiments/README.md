# Experiments & Analysis

Advanced experimental framework for RAG system analysis, optimization, and research insights.

## 🎯 Overview

The experiments module provides sophisticated tools for:
- **Dataset Analysis**: Deep insights into document characteristics
- **Performance Optimization**: 2D grid search for hyperparameters
- **Statistical Validation**: Rigorous experimental design
- **Publication-Quality Visualizations**: Research-ready plots and charts
- **Reproducible Research**: Standardized experimental protocols

## 📁 Directory Structure

```
experiments/
├── 📖 README.md                                    # This file
└── analysis/                                       # Analysis tools and notebooks
    ├── 📊 dataset_analyzer_clean.py                # Dataset characteristics analysis
    ├── 📓 2d_grid_optimization_analysis.ipynb      # Grid search optimization analysis
    ├── 📓 experiment1_analysis.ipynb               # Dense vs Sparse analysis
    ├── 📋 STRATIFICATION_SANITY_CHECK.md           # Dataset stratification validation
    ├── 🖼️ plots/                                    # Generated visualizations
    └── 📊 __init__.py                              # Module initialization
```

## 🔬 Analysis Tools

### 1. Dataset Analyzer (`dataset_analyzer_clean.py`)

Comprehensive tool for analyzing dataset characteristics with publication-quality visualizations.

**Features:**
- Document length distributions
- Content type analysis (code vs text)
- Language detection and distribution
- Tag frequency analysis
- Quality metrics computation
- Statistical summaries

**Usage:**
```bash
# Analyze SOSum dataset
python experiments/analysis/dataset_analyzer_clean.py \
  --dataset sosum \
  --input datasets/sosum/data/ \
  --output analysis/sosum_analysis/

# Custom analysis with specific focus
python experiments/analysis/dataset_analyzer_clean.py \
  --dataset custom \
  --input processed/my_dataset/ \
  --output analysis/my_analysis/ \
  --focus code_analysis,quality_metrics \
  --plot-style publication
```

**Generated Analysis:**
- **Document Statistics**: Length, word count, character distributions
- **Content Analysis**: Code block detection, programming languages
- **Quality Metrics**: Readability scores, duplicate detection
- **Visualizations**: Histograms, scatter plots, heatmaps
- **Summary Report**: Key insights and recommendations

**Example Output:**
```
analysis/sosum_analysis/
├── 📊 document_statistics.json          # Numerical summaries
├── 📈 length_distribution.png           # Document length histogram
├── 📊 language_distribution.png         # Programming language breakdown
├── 🎯 quality_metrics.png               # Quality score distributions
├── 📋 summary_report.html               # Comprehensive HTML report
└── 📊 correlation_matrix.png            # Feature correlations
```

### 2. Grid Search Optimization Analysis (`2d_grid_optimization_analysis.ipynb`)

Interactive Jupyter notebook for analyzing hyperparameter optimization results.

**Features:**
- 2D parameter space visualization
- Performance heatmaps
- Convergence analysis
- Optimal parameter identification
- Statistical significance testing

**Key Sections:**
1. **Data Loading**: Import optimization results
2. **Parameter Space Visualization**: 2D grid plots
3. **Performance Analysis**: Metric comparisons
4. **Optimization Paths**: Algorithm convergence
5. **Statistical Tests**: Significance validation
6. **Recommendations**: Optimal parameter suggestions

**Usage:**
```bash
# Start Jupyter notebook
jupyter notebook experiments/analysis/2d_grid_optimization_analysis.ipynb

# Or run as script
jupyter nbconvert --execute experiments/analysis/2d_grid_optimization_analysis.ipynb
```

### 3. Experiment Analysis (`experiment1_analysis.ipynb`)

Detailed analysis of specific experiments comparing retrieval strategies.

**Experiment 1: Dense vs Sparse Retrieval**
- Performance comparison across datasets
- Statistical significance testing
- Error analysis and failure cases
- Computational efficiency analysis

**Generated Insights:**
- Which strategy works best for different query types
- Performance trade-offs (accuracy vs speed)
- Dataset-specific recommendations
- Statistical confidence intervals

## 📊 Dataset Analysis Features

### Statistical Analysis
```python
from experiments.analysis.dataset_analyzer_clean import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer(
    dataset_path="datasets/sosum/data/",
    output_dir="analysis/sosum/"
)

# Run comprehensive analysis
results = analyzer.analyze_all()

# Generate specific analyses
length_stats = analyzer.analyze_document_lengths()
quality_metrics = analyzer.analyze_quality_metrics()
code_analysis = analyzer.analyze_code_content()
```

### Visualization Configuration
```python
# Publication-quality plot settings
PLOT_CONFIG = {
    'style': 'publication',
    'font_family': 'GFS Didot',
    'dpi': 400,
    'figure_size': (12, 8),
    'color_palette': 'viridis',
    'save_formats': ['png', 'pdf', 'svg']
}

analyzer = DatasetAnalyzer(plot_config=PLOT_CONFIG)
```

### Content Analysis
```python
# Analyze programming languages in code blocks
language_stats = analyzer.analyze_programming_languages()
print(f"Top languages: {language_stats['top_languages']}")
print(f"Code coverage: {language_stats['code_coverage']:.2%}")

# Quality metrics analysis
quality_analysis = analyzer.analyze_quality_metrics()
print(f"Average readability: {quality_analysis['avg_readability']:.3f}")
print(f"Duplicate rate: {quality_analysis['duplicate_rate']:.2%}")
```

## 🎛️ Optimization Experiments

### 2D Grid Search Configuration
```yaml
# experiments/configs/grid_search.yml
optimization:
  parameters:
    alpha:
      min: 0.0
      max: 1.0
      step: 0.1
    rrfk:
      values: [10, 20, 30, 50, 100, 200]
  
  objective:
    metric: "recall@5"
    direction: "maximize"
  
  constraints:
    max_latency: 500  # milliseconds
    min_throughput: 10  # queries/second
  
  validation:
    cv_folds: 5
    test_split: 0.2
    random_seed: 42
```

### Running Optimization
```python
from benchmarks.optimize_2d_grid_alpha_rrfk import GridSearchOptimizer

# Initialize optimizer
optimizer = GridSearchOptimizer(
    config_path="experiments/configs/grid_search.yml"
)

# Run optimization
results = optimizer.optimize(
    dataset="stackoverflow",
    n_trials=100,
    parallel_jobs=4
)

# Analyze results
best_params = optimizer.get_best_parameters()
performance_surface = optimizer.plot_performance_surface()
```

## 📈 Statistical Analysis

### Significance Testing
```python
from experiments.analysis.statistical_analyzer import ExperimentAnalyzer

analyzer = ExperimentAnalyzer()

# Compare two retrieval strategies
comparison = analyzer.compare_strategies(
    strategy_a="dense",
    strategy_b="hybrid",
    metric="recall@5",
    alpha=0.05
)

print(f"p-value: {comparison['p_value']:.4f}")
print(f"Effect size: {comparison['effect_size']:.3f}")
print(f"Significant: {comparison['is_significant']}")
```

### Power Analysis
```python
# Determine required sample size
power_analysis = analyzer.power_analysis(
    effect_size=0.2,
    alpha=0.05,
    power=0.8
)

print(f"Required sample size: {power_analysis['sample_size']}")
```

## 🎨 Visualization Examples

### Document Length Distribution
```python
# Generate publication-quality histogram
analyzer.plot_document_lengths(
    bins=50,
    style='publication',
    save_path='plots/document_lengths.pdf'
)
```

### Performance Heatmap
```python
# 2D parameter optimization heatmap
optimizer.plot_heatmap(
    x_param='alpha',
    y_param='rrfk',
    metric='recall@5',
    colormap='viridis'
)
```

### Strategy Comparison
```python
# Box plot comparing strategies
analyzer.plot_strategy_comparison(
    strategies=['dense', 'sparse', 'hybrid'],
    metric='recall@5',
    include_significance=True
)
```

## 🔬 Research Protocols

### Experimental Design Checklist
- [ ] **Hypothesis**: Clear, testable hypothesis
- [ ] **Sample Size**: Adequate statistical power
- [ ] **Randomization**: Proper randomization of test cases
- [ ] **Controls**: Appropriate baseline comparisons
- [ ] **Metrics**: Relevant evaluation metrics
- [ ] **Validation**: Cross-validation or holdout testing
- [ ] **Reproducibility**: Fixed random seeds, documented environment

### Result Reporting Template
```markdown
## Experiment: [Name]

### Hypothesis
[Clear statement of what you're testing]

### Method
- **Dataset**: [Dataset name and size]
- **Strategies**: [Retrieval strategies compared]
- **Metrics**: [Evaluation metrics used]
- **Validation**: [Cross-validation approach]

### Results
- **Primary Metric**: [Main result with confidence interval]
- **Statistical Test**: [Test used and p-value]
- **Effect Size**: [Practical significance measure]

### Conclusions
[Interpretation and implications]

### Limitations
[Known limitations and potential confounds]
```

## 🛠️ Custom Experiments

### Creating New Experiments
```python
from experiments.base_experiment import BaseExperiment

class MyCustomExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.name = "my_custom_experiment"
    
    def setup(self):
        """Configure experiment parameters"""
        self.datasets = self.config['datasets']
        self.strategies = self.config['strategies']
    
    def run(self):
        """Execute experiment logic"""
        results = {}
        for dataset in self.datasets:
            for strategy in self.strategies:
                result = self._run_single_experiment(dataset, strategy)
                results[f"{dataset}_{strategy}"] = result
        return results
    
    def analyze(self, results):
        """Analyze and visualize results"""
        statistical_analysis = self._statistical_analysis(results)
        visualizations = self._create_visualizations(results)
        return {
            'statistics': statistical_analysis,
            'plots': visualizations
        }
```

### Experiment Configuration
```yaml
# experiments/configs/my_experiment.yml
experiment:
  name: "my_custom_experiment"
  description: "Custom experiment for specific research question"
  
parameters:
  datasets: ["stackoverflow"]  # Currently implemented dataset
  strategies: ["dense", "sparse", "hybrid"]
  metrics: ["recall@5", "mrr", "ndcg@10"]
  
validation:
  method: "cross_validation"
  folds: 5
  random_seed: 42
  
output:
  save_results: true
  generate_plots: true
  create_report: true
```

## 📊 Analysis Reports

### Automated Report Generation
```python
from experiments.report_generator import ExperimentReportGenerator

# Generate comprehensive analysis report
generator = ExperimentReportGenerator()
report = generator.generate_report(
    experiment_results="results/my_experiment.json",
    template="templates/research_report.html",
    output_path="reports/my_experiment_analysis.html"
)
```

### Report Contents
- **Executive Summary**: Key findings and recommendations
- **Methodology**: Experimental design and validation
- **Results**: Statistical analysis with visualizations
- **Discussion**: Interpretation and implications
- **Appendix**: Detailed data and code

## 🐛 Troubleshooting

### Common Issues

**Jupyter Notebook Kernel Issues:**
```bash
# Install kernel in virtual environment
python -m ipykernel install --user --name thesis --display-name "Thesis"

# Select kernel in Jupyter
jupyter notebook --kernel=thesis
```

**Memory Issues with Large Datasets:**
```python
# Process data in chunks
analyzer = DatasetAnalyzer(chunk_size=10000)
results = analyzer.analyze_in_chunks()
```

**Plot Rendering Issues:**
```python
# Use different backend for plots
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python experiments/analysis/dataset_analyzer_clean.py --debug
```

## 🎯 Best Practices

1. **Reproducibility**: Always set random seeds
2. **Documentation**: Document all experimental choices
3. **Version Control**: Track experiment versions
4. **Statistical Rigor**: Use appropriate statistical tests
5. **Visualization**: Create clear, publication-quality plots
6. **Validation**: Use proper train/validation/test splits
7. **Error Analysis**: Examine failure cases
8. **Peer Review**: Have others review experimental design

## 🔗 Integration

### With Benchmarks
```python
# Use experiment results in benchmark optimization
from benchmarks.optimization import BenchmarkOptimizer

optimizer = BenchmarkOptimizer()
optimal_params = optimizer.use_experiment_results(
    experiment_path="results/grid_search_results.json"
)
```

### With Main Pipeline
```python
# Apply insights to production configuration
from config.optimizer import ConfigOptimizer

optimizer = ConfigOptimizer()
production_config = optimizer.optimize_from_experiments(
    experiment_results="analysis/experiment_summary.json"
)
```

This experimental framework enables rigorous scientific analysis of your RAG system, providing actionable insights for optimization and research publication.
