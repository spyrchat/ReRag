#!/usr/bin/env python3
"""
Test full ingestion pipeline with StackOverflow data.
"""
from pipelines.contracts import DatasetSplit
from pipelines.ingest.pipeline import IngestionPipeline
import os
import sys

# Add the project root to the Python path
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')


def test_full_ingestion():
    print("=== Testing Full Ingestion Pipeline ===")

    # Set up Google API key if needed
    # Note: You may need to set GOOGLE_API_KEY environment variable
    # export GOOGLE_API_KEY="your_api_key_here"

    config_path = "pipelines/configs/stackoverflow_hybrid.yml"
    print(f"📋 Config: {config_path}")

    try:
        # Create pipeline
        pipeline = IngestionPipeline(config_path)
        print("✅ Pipeline initialized successfully")

        # Test with a small batch first (first 50 documents)
        print("\n📥 Testing ingestion with small batch (first 50 documents)...")

        # Create adapter
        from pipelines.adapters.stackoverflow import StackOverflowAdapter
        adapter = StackOverflowAdapter(
            dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data",
            version="v1.0.0"
        )

        # Run ingestion
        result = pipeline.ingest_dataset(
            adapter=adapter,
            split=DatasetSplit.ALL,
            dry_run=True,  # Start with dry run to test without uploading
            max_documents=10  # Process first 10 final documents
        )

        print(f"✅ Ingestion completed!")
        print(f"📊 Results: {result}")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

        # Check if it's an API key issue
        if "api" in str(e).lower() or "key" in str(e).lower():
            print(
                "\n💡 Tip: Make sure you have set the GOOGLE_API_KEY environment variable:")
            print("export GOOGLE_API_KEY='your_google_gemini_api_key'")


if __name__ == "__main__":
    test_full_ingestion()
