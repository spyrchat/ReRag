#!/usr/bin/env python3
"""
Debug script to check the row order and types.
"""
from pipelines.contracts import DatasetSplit
from pipelines.adapters.stackoverflow import StackOverflowAdapter
import os
import sys

# Add the project root to the Python path
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')


def debug_row_order():
    print("=== Debugging Row Order ===")

    # Initialize adapter
    dataset_path = "/home/spiros/Desktop/Thesis/datasets/sosum/data"
    adapter = StackOverflowAdapter(dataset_path=dataset_path)

    # Read rows and check order
    rows = list(adapter.read_rows(DatasetSplit.ALL))
    print(f"Total rows: {len(rows)}")

    # Check first 20 rows
    print("\nFirst 20 rows:")
    for i, row in enumerate(rows[:20]):
        print(f"  Row {i+1}: {row.post_type} (ID: {row.external_id})")

    # Count questions vs answers in first 50
    first_50_questions = [row for row in rows[:50]
                          if row.post_type == "question"]
    first_50_answers = [row for row in rows[:50] if row.post_type == "answer"]

    print(f"\nFirst 50 rows breakdown:")
    print(f"  Questions: {len(first_50_questions)}")
    print(f"  Answers: {len(first_50_answers)}")

    # Test converting first 50 to documents
    print(f"\nTesting to_documents() with first 50 rows...")
    documents = adapter.to_documents(rows[:50], DatasetSplit.ALL)
    print(f"Documents created: {len(documents)}")

    # Test converting ALL rows to documents
    print(f"\nTesting to_documents() with ALL rows...")
    documents_all = adapter.to_documents(rows, DatasetSplit.ALL)
    print(f"Documents created: {len(documents_all)}")


if __name__ == "__main__":
    debug_row_order()
