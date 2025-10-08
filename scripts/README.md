# Scripts & Utilities

Collection of utility scripts for system setup, data processing, and maintenance tasks.

## 📁 Scripts Overview

```
scripts/
├── 📖 README.md                       # This file
├── 🛠️ setup_sosum.sh                  # SOSum dataset setup script
├── 🔧 standalone_my_dataset_processor.py   # Custom dataset processor
└── 🔧 standalone_sosum_processor.py        # SOSum dataset processor
```

## 🚀 Available Scripts

### 1. SOSum Dataset Setup (`setup_sosum.sh`)

Automated setup script for the Stack Overflow Summarization (SOSum) dataset.

**What it does:**
- Downloads the SOSum dataset from GitHub
- Extracts and organizes files
- Validates data integrity
- Sets up directory structure
- Prepares dataset for ingestion pipeline

**Usage:**
```bash
# Make script executable
chmod +x scripts/setup_sosum.sh

# Run setup (creates datasets/sosum/ directory)
./scripts/setup_sosum.sh

# Custom target directory
./scripts/setup_sosum.sh /path/to/custom/location
```

**Output Structure:**
```
datasets/sosum/
├── data/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── validation.jsonl
├── metadata/
│   ├── dataset_info.json
│   └── statistics.json
└── README.md
```

### 2. Standalone SOSum Processor (`standalone_sosum_processor.py`)

Standalone processor for SOSum dataset that can be run independently of the main pipeline.

**Features:**
- Code-aware text processing
- HTML cleaning while preserving code blocks
- Question-answer pair extraction
- Tag and metadata processing
- Validation and quality checks

**Usage:**
```bash
# Process entire SOSum dataset
python scripts/standalone_sosum_processor.py \
  --input datasets/sosum/data/ \
  --output processed/sosum/ \
  --chunk-size 500 \
  --preserve-code

# Process specific file
python scripts/standalone_sosum_processor.py \
  --input datasets/sosum/data/train.jsonl \
  --output processed/sosum_train.jsonl \
  --validate-quality

# Custom processing options
python scripts/standalone_sosum_processor.py \
  --input datasets/sosum/data/ \
  --output processed/sosum/ \
  --min-length 50 \
  --max-length 5000 \
  --remove-duplicates \
  --filter-languages python,javascript
```

**Configuration Options:**
```python
{
    "chunk_size": 500,           # Maximum chunk size
    "chunk_overlap": 100,        # Overlap between chunks
    "preserve_code": True,       # Keep code blocks intact
    "clean_html": True,          # Remove HTML tags
    "min_length": 50,           # Minimum text length
    "max_length": 5000,         # Maximum text length
    "remove_duplicates": True,   # Deduplicate content
    "validate_quality": True,    # Run quality checks
    "filter_languages": []      # Filter by programming languages
}
```

### 3. Custom Dataset Processor (`standalone_my_dataset_processor.py`)

Template for processing custom datasets with your own data format.

**Features:**
- Customizable data parsing
- Flexible text processing pipeline
- Metadata extraction
- Quality validation
- Output format standardization

**Usage:**
```bash
# Process custom dataset
python scripts/standalone_my_dataset_processor.py \
  --input data/my_dataset/ \
  --output processed/my_dataset/ \
  --format json \
  --config configs/my_processing.yml

# With custom adapter
python scripts/standalone_my_dataset_processor.py \
  --input data/my_dataset/ \
  --output processed/my_dataset/ \
  --adapter my_custom_adapter.py \
  --batch-size 1000
```

**Custom Adapter Example:**
```python
from scripts.base_processor import BaseDatasetProcessor

class MyCustomProcessor(BaseDatasetProcessor):
    def parse_document(self, raw_doc):
        """Parse your custom document format"""
        return {
            'id': raw_doc['doc_id'],
            'title': raw_doc['document_title'],
            'content': raw_doc['body_text'],
            'metadata': {
                'source': raw_doc['source'],
                'date': raw_doc['created_at']
            }
        }
    
    def validate_document(self, doc):
        """Custom validation logic"""
        return len(doc['content']) > 100
```

## 🔧 Script Configuration

### Environment Variables

Scripts respect the following environment variables:

```bash
# Data directories
export DATA_DIR="/path/to/datasets"
export OUTPUT_DIR="/path/to/processed"
export CACHE_DIR="/path/to/cache"

# Processing options
export CHUNK_SIZE=500
export BATCH_SIZE=1000
export PARALLEL_WORKERS=4

# Quality thresholds
export MIN_TEXT_LENGTH=50
export MAX_TEXT_LENGTH=10000
export QUALITY_THRESHOLD=0.8
```

### Configuration Files

Create YAML configuration files for complex processing:

```yaml
# configs/processing_config.yml
processing:
  chunk_size: 500
  chunk_overlap: 100
  preserve_formatting: true
  
validation:
  min_length: 50
  max_length: 5000
  remove_duplicates: true
  quality_checks: true
  
output:
  format: "jsonl"
  include_metadata: true
  compress: false

filters:
  languages: ["python", "javascript", "java"]
  tags: ["machine-learning", "data-science"]
  min_score: 5
```

## 🏃‍♂️ Running Scripts

### Batch Processing
```bash
# Process SOSUM dataset (currently implemented)
echo "Processing SOSUM..."
python scripts/standalone_sosum_processor.py \
  --input datasets/sosum/ \
  --output processed/sosum/ \
  --config configs/sosum_config.yml

# For other datasets, use standalone_my_dataset_processor.py as a template
# Copy and customize for your specific dataset
```

### Parallel Processing
```bash
# For future use when multiple datasets are available
# Example with GNU parallel:
# parallel -j4 python scripts/standalone_{}_processor.py \
#   --input datasets/{} \
#   --output processed/{} ::: dataset1 dataset2 dataset3

# Currently, process SOSUM dataset:
python scripts/standalone_sosum_processor.py \
  --input datasets/sosum/ \
  --output processed/sosum/
```

### Monitoring Progress
```bash
# Run with progress tracking
python scripts/standalone_sosum_processor.py \
  --input datasets/sosum/ \
  --output processed/sosum/ \
  --progress-bar \
  --log-level INFO \
  --checkpoint-every 1000
```

## 📊 Output Formats

### Processed Document Format
```json
{
  "id": "doc_12345",
  "title": "How to implement async/await in Python",
  "content": "Python's asyncio library provides...",
  "chunks": [
    {
      "id": "chunk_1",
      "text": "Python's asyncio library provides...",
      "start_char": 0,
      "end_char": 500
    }
  ],
  "metadata": {
    "source": "stackoverflow",
    "tags": ["python", "asyncio", "concurrency"],
    "score": 42,
    "created_at": "2024-01-15T10:30:00Z",
    "processed_at": "2024-10-07T14:20:00Z"
  },
  "quality_metrics": {
    "readability_score": 0.85,
    "code_block_count": 3,
    "avg_sentence_length": 18.5
  }
}
```

### Statistics Report
```json
{
  "dataset_name": "sosum",
  "total_documents": 125000,
  "processed_documents": 124500,
  "skipped_documents": 500,
  "total_chunks": 450000,
  "avg_chunks_per_doc": 3.6,
  "avg_chunk_size": 485,
  "quality_metrics": {
    "avg_readability": 0.82,
    "code_coverage": 0.65,
    "duplicate_rate": 0.02
  },
  "processing_time": "00:45:32",
  "errors": []
}
```

## 🛠️ Creating Custom Scripts

### Script Template
```python
#!/usr/bin/env python3
"""
Custom dataset processor template
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDatasetProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def process_file(self, input_path: Path, output_path: Path):
        """Process a single file"""
        logger.info(f"Processing {input_path}")
        # Your processing logic here
        
    def validate_output(self, output_path: Path):
        """Validate processed output"""
        # Your validation logic here
        pass

def main():
    parser = argparse.ArgumentParser(description="Custom dataset processor")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch-size", type=int, default=1000)
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    # Initialize processor
    processor = CustomDatasetProcessor(config)
    
    # Process files
    processor.process_file(args.input, args.output)
    processor.validate_output(args.output)
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
```

### Adding to Pipeline
```python
# Register custom processor in pipeline
from pipelines.adapters.registry import AdapterRegistry

AdapterRegistry.register(
    name="my_custom_dataset",
    adapter_class=MyCustomAdapter,
    processor_script="scripts/standalone_my_dataset_processor.py"
)
```

## 🐛 Troubleshooting

### Common Issues

**Script Permission Denied:**
```bash
chmod +x scripts/setup_sosum.sh
```

**Memory Issues with Large Datasets:**
```bash
# Process in smaller batches
python scripts/standalone_sosum_processor.py \
  --batch-size 100 \
  --memory-limit 4GB
```

**Network Issues During Download:**
```bash
# Retry with timeout
./scripts/setup_sosum.sh --retry 3 --timeout 300
```

**Processing Errors:**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/standalone_sosum_processor.py --verbose
```

### Debug Mode
```bash
# Run with detailed debugging
python -u scripts/standalone_sosum_processor.py \
  --debug \
  --checkpoint-every 100 \
  --log-file debug.log
```

## 🎯 Best Practices

1. **Validation**: Always validate input and output data
2. **Checkpointing**: Save progress for long-running scripts
3. **Error Handling**: Gracefully handle processing errors
4. **Logging**: Comprehensive logging for debugging
5. **Configuration**: Use config files for complex parameters
6. **Testing**: Test scripts on small datasets first
7. **Documentation**: Document custom processing logic
8. **Performance**: Monitor memory and CPU usage

## 🔗 Integration

### With Main Pipeline
```python
# Use processed data in main pipeline
from pipelines.ingest.pipeline import IngestPipeline

pipeline = IngestPipeline(config)
pipeline.process_directory("processed/sosum/")
```

### With Benchmarks
```python
# Use processed data for benchmarking
from benchmarks.benchmarks_runner import BenchmarkRunner

runner = BenchmarkRunner(config)
results = runner.run_benchmark(
    dataset_path="processed/sosum/",
    queries_path="processed/sosum/queries.jsonl"
)
```

These scripts provide essential utilities for data preparation and processing, enabling smooth integration of custom datasets into the RAG system pipeline.
