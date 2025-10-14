# Actual CLI Reference - Verified Against Codebase

**Last Updated:** 2025-10-08  
**Purpose:** Single source of truth for all CLI commands that actually exist

## ✅ Scripts WITH CLI Support

### 1. bin/ingest.py
**Subcommands:**
- `ingest` - Ingest a dataset
- `status` - Show pipeline status
- `cleanup` - Clean up canary collections

**Global Flags:**
- `--config`, `-c` - Configuration file path
- `--verbose`, `-v` - Verbose logging

**Ingest Subcommand Flags:**
- `adapter_type` (positional, optional) - Adapter type
- `dataset_path` (positional, optional) - Path to dataset
- `--version` - Dataset version (default: "1.0.0")
- `--split` - Dataset split (choices: train, val, test, all; default: all)
- `--dry-run` - Don't upload to vector store
- `--max-docs` - Maximum documents to process
- `--canary` - Use canary collection
- `--verify` - Run verification after ingestion

### 2. main.py
**Flags:**
- `--query` - Single query to process (enables non-interactive mode)

### 3. benchmarks/experiment1.py
**Flags:**
- `--test` - Run in test mode
- `--output-dir` - Output directory (default: 'results/experiment_1')

### 4. benchmarks/experiment3.py
**Flags:**
- `--test` - Run in test mode
- `--output-dir` - Output directory (default: 'results/experiment_3')

### 5. benchmarks/optimize_2d_grid_alpha_rrfk.py
**Flags:**
- `--scenario-yaml` (required) - Path to scenario YAML
- `--dataset-path` (required) - Path to dataset
- `--n-folds` - Number of folds (default: 5)
- `--max-queries-dev` - Max queries for dev set
- `--max-queries-test` - Max queries for test set
- `--output-dir` - Output directory (default: "results/")

### 6. benchmarks/stratification.py
**Flags:**
- `--dataset-path` (required) - Path to dataset root
- `--fold` - Fold number (default: 0)
- `--split` - Split type (choices: train, dev, test; default: test)
