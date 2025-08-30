"""Debug the StackOverflow adapter to see why it's reading 0 documents."""

import pandas as pd
from pathlib import Path
from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.contracts import DatasetSplit


def debug_adapter():
    """Debug the StackOverflow adapter step by step."""
    data_path = Path("/home/spiros/Desktop/Thesis/datasets/sosum/data")

    print("🔍 Debugging StackOverflow Adapter")
    print(f"Data path: {data_path}")
    print(f"Path exists: {data_path.exists()}")

    # Check files
    if data_path.exists():
        files = list(data_path.iterdir())
        print(f"Files in directory: {[f.name for f in files]}")

        # Check question.csv
        question_file = data_path / "question.csv"
        answer_file = data_path / "answer.csv"

        print(f"\n📄 Checking question.csv:")
        print(f"  Exists: {question_file.exists()}")
        if question_file.exists():
            try:
                df = pd.read_csv(question_file)
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  First row preview:")
                if len(df) > 0:
                    for col in df.columns[:5]:  # Show first 5 columns
                        print(
                            f"    {col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
            except Exception as e:
                print(f"  Error reading CSV: {e}")

        print(f"\n📄 Checking answer.csv:")
        print(f"  Exists: {answer_file.exists()}")
        if answer_file.exists():
            try:
                df = pd.read_csv(answer_file)
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
            except Exception as e:
                print(f"  Error reading CSV: {e}")

    # Test adapter directly
    print(f"\n🔧 Testing adapter directly:")
    try:
        adapter = StackOverflowAdapter(
            # Fixed: use dataset_path, not data_path
            dataset_path=str(data_path),
            version="v1.0.0"
        )
        print("✅ Adapter created successfully")

        # Try reading rows
        print("📖 Attempting to read rows...")
        rows = list(adapter.read_rows(split=DatasetSplit.ALL))
        print(f"📊 Total rows read: {len(rows)}")

        if len(rows) > 0:
            print("📝 Sample row:")
            sample_row = rows[0]
            print(f"  Type: {type(sample_row)}")
            print(f"  External ID: {sample_row.external_id}")
            # Print first few attributes
            for attr in dir(sample_row):
                if not attr.startswith('_'):
                    try:
                        value = getattr(sample_row, attr)
                        if not callable(value):
                            print(f"  {attr}: {str(value)[:100]}...")
                    except:
                        pass
        else:
            print("❌ No rows read from adapter")

    except Exception as e:
        print(f"❌ Error testing adapter: {e}")
        import traceback
        traceback.print_exc()


def test_pipeline_integration():
    """Test how the pipeline calls the adapter."""
    print(f"\n🔧 Testing Pipeline Integration:")

    try:
        from pipelines.ingest.pipeline import IngestionPipeline

        # Use config file path, not config dict
        config_path = "pipelines/configs/stackoverflow_hybrid.yml"

        print(f"📋 Config path: {config_path}")

        # Create pipeline with config path
        pipeline = IngestionPipeline(config_path)
        print("✅ Pipeline created successfully")

        # Test document reading directly
        print("📖 Testing direct document reading...")

        # Create adapter
        adapter = StackOverflowAdapter(
            dataset_path="/home/spiros/Desktop/Thesis/datasets/sosum/data",
            version="v1.0.0"
        )

        # Read rows
        rows = list(adapter.read_rows(split=DatasetSplit.ALL))
        print(f"📊 Adapter returned {len(rows)} rows")

        # Convert to documents
        print("📄 Converting to documents...")
        documents = adapter.to_documents(
            rows, DatasetSplit.ALL)  # Test with ALL rows, not just first 10
        print(f"📊 Converted to {len(documents)} documents")

        if documents:
            sample_doc = documents[0]
            print(f"📝 Sample document:")
            print(f"  Content length: {len(sample_doc.page_content)}")
            print(f"  Metadata keys: {list(sample_doc.metadata.keys())}")
            print(f"  Content preview: {sample_doc.page_content[:200]}...")

    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_adapter()
    test_pipeline_integration()
