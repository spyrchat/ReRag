# Actual CLI Reference - Verified Against Codebase

**Last Updated:** 2025-10-16  
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
- `--mode` - Agent mode (choices: standard, self-rag; default: standard)
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
- `--scenario-yaml` (required) - Path to scenario YAML configuration
- `--dataset-path` (required) - Path to dataset directory
- `--test-size` - Test set size as fraction (default: 0.2)
- `--random-state` - Random state for reproducibility (default: 42)
- `--max-queries-train` - Max queries for train evaluation
- `--max-queries-test` - Max queries for test evaluation
- `--output-dir` - Output directory (default: "results/")

### 6. benchmarks/stratification.py
**Flags:**
- `--dataset-path` (required) - Path to dataset root (expects question.csv)
- `--test-size` - Test set size as fraction (default: 0.2)
- `--random-state` - Random state for reproducibility (default: 42)
- `--output` - Output path for split JSON (optional)

### 7. bin/retrieval_pipeline.py
**Flags:**
- `--config`, `-c` - Path to YAML configuration file
- `--query`, `-q` - Search query
- `--top-k`, `-k` - Number of results to retrieve (default: 5)
- `--show-content` - Show document content in results
- `--list-configs` - List all available configuration files
- `--verbose`, `-v` - Enable verbose logging

### 8. bin/qdrant_inspector.py
**Global Flags:**
- `--host` - Qdrant host (default: localhost)
- `--port` - Qdrant port (default: 6333)

**Subcommands:**
- `list` - List all collections
- `browse <collection>` - Browse collection data
  - `--limit` - Number of documents to show (default: 10)
- `search <collection> <query>` - Search collection
  - `--limit` - Number of results (default: 5)
- `filter <collection> <key> <value>` - Filter by metadata
  - `--limit` - Number of results (default: 10)
- `stats <collection>` - Show collection statistics

### 9. bin/switch_agent_config.py
**Flags:**
- `config_name` (positional, optional) - Name of configuration to switch to
- `--list`, `-l` - List available configurations
