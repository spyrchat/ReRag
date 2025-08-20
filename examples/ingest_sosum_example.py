#!/usr/bin/env python3
"""
Example script showing how to ingest the SOSum Stack Overflow dataset.

SOSum Dataset: https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools

Usage:
    1. Download SOSum dataset from GitHub
    2. Extract to a directory (e.g., /path/to/sosum/)
    3. Run this script to ingest the data
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.ingest.pipeline import IngestionPipeline
from pipelines.contracts import DatasetSplit
from config.config_loader import load_config


def main():
    """Example ingestion of SOSum dataset."""
    
    # Configuration
    sosum_path = "../datasets/sosum"  # Updated path
    config_path = "pipelines/configs/stackoverflow.yml"
    
    print("🔧 SOSum Dataset Ingestion Example")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path(sosum_path).exists():
        print(f" Dataset not found at {sosum_path}")
        print("\n To download the SOSum dataset:")
        print("   1. git clone https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools.git")
        print("   2. Update sosum_path in this script to point to the cloned directory")
        return
    
    # Load configuration
    config = load_config(config_path)
    
    # Create adapter
    print(f" Loading SOSum dataset from: {sosum_path}")
    adapter = StackOverflowAdapter(sosum_path, version="1.0.0")
    
    # Test adapter by reading a few rows
    print("\n Testing adapter - reading first 5 rows:")
    rows = list(adapter.read_rows())[:5]
    for i, row in enumerate(rows):
        print(f"  {i+1}. {row.post_type}: {row.external_id}")
        if row.title:
            print(f"     Title: {row.title[:60]}...")
        print(f"     Body: {row.body[:80]}...")
        if row.summary:
            print(f"     Summary: {row.summary[:60]}...")
        print()
    
    # Test document conversion
    print("Testing document conversion:")
    documents = adapter.to_documents(rows)
    print(f"   Converted {len(rows)} rows to {len(documents)} documents")
    
    if documents:
        doc = documents[0]
        print(f"   First doc content: {doc.page_content[:100]}...")
        print(f"   Metadata keys: {list(doc.metadata.keys())}")
    
    # Test evaluation queries
    print("\n Testing evaluation queries:")
    eval_queries = adapter.get_evaluation_queries()
    print(f"   Generated {len(eval_queries)} evaluation queries")
    
    if eval_queries:
        query = eval_queries[0]
        print(f"   First query: {query['query'][:60]}...")
        print(f"   Expected docs: {query['expected_docs']}")
    
    # Dry run ingestion
    print("\n Running dry-run ingestion (first 10 documents):")
    pipeline = IngestionPipeline(config=config)
    
    try:
        record = pipeline.ingest_dataset(
            adapter=adapter,
            split=DatasetSplit.ALL,
            dry_run=True,
            max_documents=10
        )
        
        print(" Dry run completed successfully!")
        print(f"   Dataset: {record.dataset_name}")
        print(f"   Documents: {record.total_documents}")
        print(f"   Chunks: {record.total_chunks}")
        print(f"   Success rate: {record.successful_chunks/record.total_chunks*100:.1f}%")
        
    except Exception as e:
        print(f" Dry run failed: {e}")
        return
    
    # Instructions for full ingestion
    print("\n To run full ingestion:")
    print("   # Test with canary (safe)")
    print(f"   python bin/ingest.py ingest stackoverflow {sosum_path} --canary --max-docs 100")
    print()
    print("   # Full ingestion")
    print(f"   python bin/ingest.py ingest stackoverflow {sosum_path} --config {config_path}")
    print()
    print("   # Evaluate retrieval")
    print(f"   python bin/ingest.py evaluate stackoverflow {sosum_path} --output-dir results/sosum/")


if __name__ == "__main__":
    main()
