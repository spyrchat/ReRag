# Ingesting SOSum Stack Overflow Dataset

This guide shows how to ingest the SOSum dataset using the pipeline.

## About SOSum

**SOSum** is a dataset of extractive summaries of Stack Overflow posts from:
https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools

**Dataset Statistics:**
- 506 popular Stack Overflow questions
- 2,278 total posts (questions + answers)
- 669 unique tags covered
- Median view count: 253K
- Median post score: 17
- Manual extractive summaries for answers

## Dataset Format

SOSum comes in two CSV files:

### `question.csv`
| Field | Description |
|-------|-------------|
| Question Id | Post ID of the SO question |
| Question Type | 1=conceptual, 2=how-to, 3=debug-corrective |
| Question Title | Question title as string |
| Question Body | List of sentences from question content |
| Tags | SO tags associated with question |
| Answer Posts | Comma-separated answer post IDs |

### `answer.csv`
| Field | Description |
|-------|-------------|
| Answer Id | Post ID of SO answer |
| Answer Body | List of sentences from answer content |
| Summary | Extractive summative sentences |

## Quick Start

### 1. Download the Dataset

```bash
# Clone the SOSum repository into the datasets directory
cd datasets/
git clone https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools.git sosum_source

# The CSV files are in sosum_source/data/ directory
ls sosum_source/data/
# Should show: question.csv  answer.csv

# Or download the CSV files directly
mkdir -p sosum/
# Place question.csv and answer.csv in sosum/ directory
```

### 2. Test the Adapter

```bash
# Run the example script to test everything works
python examples/ingest_sosum_example.py
```

### 3. Dry Run Ingestion

```bash
# Test with a small sample (no upload to vector store)
python bin/ingest.py ingest stackoverflow sosum/ --dry-run --max-docs 10 --verbose
```

### 4. Canary Ingestion

```bash
# Safe test with real upload to canary collection
python bin/ingest.py ingest stackoverflow sosum/ --canary --max-docs 100 --verify
```

### 5. Check Status

```bash
python bin/ingest.py status
```

### 6. Full Ingestion

```bash
# Ingest all data
python bin/ingest.py ingest stackoverflow sosum/ --config pipelines/configs/stackoverflow.yml
```

### 7. Evaluate Retrieval

```bash
# Test retrieval performance
python bin/ingest.py evaluate stackoverflow sosum/ --output-dir results/sosum/
```

## What Gets Ingested

### Document Types

1. **Questions**: Combined title + body content
   - ID format: `q_{question_id}`
   - Content: "Title: {title}\n\nQuestion: {body}"
   - Metadata: tags, question_type, related_posts

2. **Answers**: Answer body + summary (if available)
   - ID format: `a_{answer_id}`
   - Content: "Answer: {body}\n\nSummary: {summary}" (if summary exists)
   - Metadata: has_summary, summary

### Metadata Fields

- `external_id`: Unique identifier (q_123 or a_456)
- `source`: "stackoverflow_sosum"
- `post_type`: "question" or "answer"
- `doc_type`: "question" or "answer"
- `tags`: List of SO tags (questions only)
- `title`: Question title (questions only)
- `question_type`: 1, 2, or 3 (questions only)
- `has_summary`: Boolean (answers only)
- `summary`: Extractive summary text (answers only)

### Evaluation Queries

The adapter automatically generates evaluation queries:

1. **Question titles** → Should retrieve the question document
2. **Short question queries** → First 5 words of title
3. **Answer summaries** → Should retrieve the answer document

## Configuration

The pipeline uses `pipelines/configs/stackoverflow.yml`:

- **Code-aware chunking**: Preserves code blocks and functions
- **Hybrid embedding**: Dense + sparse vectors for better code retrieval
- **Smaller validation limits**: Handles extractive summaries (shorter content)
- **SOSum-specific collection**: `sosum_stackoverflow_v1`

## Expected Results

After successful ingestion:

- **Documents**: ~2,278 documents (506 questions + ~1,772 answers)
- **Chunks**: Depends on chunking strategy (likely 3,000-5,000 chunks)
- **Vectors**: Hybrid (dense + sparse) for each chunk
- **Collection**: Named `sosum_stackoverflow_v1` in Qdrant

## Troubleshooting

### Common Issues

1. **File not found**:
   ```bash
   # Make sure files exist
   ls sosum/data/question.csv sosum/data/answer.csv
   ```

2. **Parsing errors**:
   ```bash
   # Check CSV format
   head -5 sosum/data/question.csv
   head -5 sosum/data/answer.csv
   ```

3. **Import errors**:
   ```bash
   # Check dependencies
   pip install pandas pydantic langchain-core
   ```

4. **Qdrant connection**:
   ```bash
   # Check if Qdrant is running
   python bin/ingest.py status
   ```

### Debug Commands

```bash
# Verbose logging
python bin/ingest.py ingest stackoverflow sosum/ --dry-run --verbose

# Check logs
tail -f logs/ingestion.log

# Test specific number of docs
python bin/ingest.py ingest stackoverflow sosum/ --dry-run --max-docs 5
```

## Integration with Retrieval

After ingestion, you can test retrieval:

```python
from retrievers.router import RetrieverRouter
from config.config_loader import load_config

config = load_config("pipelines/configs/stackoverflow.yml")
retriever = RetrieverRouter(config)

# Test queries
results = retriever.search("Python list comprehension example", top_k=5)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Doc: {result['metadata']['external_id']}")
    print(f"Content: {result['content'][:100]}...")
    print()
```

## Next Steps

1. **Add more datasets**: Use the same adapter pattern for other SO datasets
2. **Custom evaluation**: Add domain-specific evaluation queries
3. **Tune chunking**: Experiment with chunk sizes for code content
4. **Hybrid weights**: Tune dense vs sparse retrieval weights
5. **Summary utilization**: Use extractive summaries for enhanced retrieval
