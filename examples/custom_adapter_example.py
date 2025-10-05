#!/usr/bin/env python3
"""
Example: Create and use a custom adapter without modifying core code.

This demonstrates how to add new dataset adapters to the ingestion pipeline
without changing any existing code - just create your adapter and reference it!
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Iterable
from pipelines.contracts import DatasetAdapter, BaseRow, DatasetSplit
from pipelines.adapters.loader import AdapterLoader
from langchain_core.documents import Document


# ============================================================================
# Step 1: Define your custom adapter (no modifications to existing code!)
# ============================================================================

class CustomCSVRow(BaseRow):
    """Row schema for custom CSV dataset."""
    title: str
    content: str
    category: str = "general"


class CustomCSVAdapter(DatasetAdapter):
    """
    Example custom adapter for CSV files.

    This adapter can be used WITHOUT modifying any core code!
    """

    def __init__(self, dataset_path: str, version: str = "1.0.0", **kwargs):
        """
        Initialize adapter.

        Args:
            dataset_path: Path to CSV file or directory
            version: Dataset version
            **kwargs: Custom parameters (e.g., delimiter, encoding)
        """
        self.dataset_path = Path(dataset_path)
        self._version = version

        # Use kwargs for custom configuration
        self.delimiter = kwargs.get("delimiter", ",")
        self.encoding = kwargs.get("encoding", "utf-8")

        print(f"📁 Initialized CustomCSVAdapter:")
        print(f"   Path: {self.dataset_path}")
        print(f"   Version: {version}")
        print(f"   Delimiter: {self.delimiter}")
        print(f"   Encoding: {self.encoding}")

    @property
    def source_name(self) -> str:
        return "custom_csv_dataset"

    @property
    def version(self) -> str:
        return self._version

    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[CustomCSVRow]:
        """Read CSV rows (simplified example)."""
        # In real implementation, you'd read actual CSV files here
        print(
            f"📖 Reading rows from {self.dataset_path} (split: {split.value})")

        # Example: return some dummy rows
        yield CustomCSVRow(
            external_id="row_1",
            title="Example Document 1",
            content="This is the content of document 1",
            category="tech"
        )
        yield CustomCSVRow(
            external_id="row_2",
            title="Example Document 2",
            content="This is the content of document 2",
            category="science"
        )

    def to_documents(self, rows: List[CustomCSVRow], split: DatasetSplit) -> List[Document]:
        """Convert rows to LangChain documents."""
        print(f"📝 Converting {len(rows)} rows to documents")

        documents = []
        for row in rows:
            doc = Document(
                page_content=f"{row.title}\n\n{row.content}",
                metadata={
                    "source": self.source_name,
                    "external_id": row.external_id,
                    "split": split.value,
                    "category": row.category,
                    "dataset_version": self.version
                }
            )
            documents.append(doc)

        return documents

    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Return evaluation queries."""
        return [
            {"query": "Example query 1", "expected_doc_id": "row_1"},
            {"query": "Example query 2", "expected_doc_id": "row_2"}
        ]


# ============================================================================
# Step 2: Use the adapter without modifying core code
# ============================================================================

def main():
    """Demonstrate dynamic adapter loading."""
    print("=" * 70)
    print("🚀 Dynamic Adapter Loading Demo")
    print("=" * 70)
    print()

    # Method 1: Register a shortcut for convenience (optional)
    print("1️⃣  Registering custom adapter shortcut...")
    AdapterLoader.register_shortcut(
        "custom_csv",
        "examples.custom_adapter_example.CustomCSVAdapter"
    )
    print("   ✅ Registered 'custom_csv' shortcut")
    print()

    # Method 2: Load using shortcut
    print("2️⃣  Loading adapter using shortcut...")
    adapter1 = AdapterLoader.load_adapter(
        "custom_csv",
        "/path/to/data",
        version="1.0.0",
        delimiter="|",  # Custom parameter!
        encoding="utf-8"
    )
    print("   ✅ Loaded successfully!")
    print()

    # Method 3: Load using full class path (no registration needed!)
    print("3️⃣  Loading adapter using full class path...")
    adapter2 = AdapterLoader.load_adapter(
        "examples.custom_adapter_example.CustomCSVAdapter",
        "/another/path",
        version="2.0.0",
        delimiter=","
    )
    print("   ✅ Loaded successfully!")
    print()

    # Method 4: Use the adapter
    print("4️⃣  Using the adapter...")
    print()

    # Read rows
    rows = list(adapter1.read_rows(DatasetSplit.ALL))
    print(f"   Read {len(rows)} rows")

    # Convert to documents
    documents = adapter1.to_documents(rows, DatasetSplit.ALL)
    print(f"   Converted to {len(documents)} documents")
    print()

    # Show example document
    if documents:
        print("   📄 Example document:")
        print(f"      Content: {documents[0].page_content[:80]}...")
        print(f"      Metadata: {documents[0].metadata}")
    print()

    print("=" * 70)
    print("✅ Success! You can now use custom adapters without code changes!")
    print("=" * 70)
    print()
    print("💡 To use this adapter with the CLI:")
    print()
    print("   # Option 1: Use shortcut (after registering)")
    print("   python bin/ingest.py ingest custom_csv /path/to/data")
    print()
    print("   # Option 2: Use full class path (no registration needed!)")
    print("   python bin/ingest.py ingest \\")
    print("     'examples.custom_adapter_example.CustomCSVAdapter' \\")
    print("     /path/to/data \\")
    print("     --config my_config.yml")
    print()
    print("📖 See docs/DYNAMIC_ADAPTERS.md for more details")
    print()


if __name__ == "__main__":
    main()
