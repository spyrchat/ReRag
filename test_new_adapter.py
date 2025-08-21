#!/usr/bin/env python3
"""
Test the new answer-focused adapter approach.
"""

from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.contracts import DatasetSplit

def test_adapter():
    # Initialize adapter
    adapter = StackOverflowAdapter("/home/spiros/Desktop/Thesis/Thesis/datasets/sosum")
    
    print("=== Testing New Answer-Focused Adapter ===")
    
    # Test reading rows
    print("\n1. Reading rows...")
    rows = list(adapter.read_rows(DatasetSplit.ALL))
    print(f"Total rows read: {len(rows)}")
    
    question_count = sum(1 for r in rows if r.post_type == "question")
    answer_count = sum(1 for r in rows if r.post_type == "answer")
    print(f"Questions: {question_count}")
    print(f"Answers: {answer_count}")
    
    # Test document conversion
    print("\n2. Converting to documents...")
    documents = adapter.to_documents(rows, DatasetSplit.ALL)
    print(f"Documents created: {len(documents)}")
    
    if documents:
        print(f"\nFirst document preview:")
        doc = documents[0]
        print(f"External ID: {doc.metadata.get('external_id')}")
        print(f"Post type: {doc.metadata.get('post_type')}")
        print(f"Has question context: {doc.metadata.get('has_question_context', False)}")
        print(f"Tags: {doc.metadata.get('tags', [])}")
        print(f"Content preview: {doc.page_content[:200]}...")
        print(f"Metadata keys: {list(doc.metadata.keys())}")
    
    # Test evaluation queries
    print("\n3. Testing evaluation queries...")
    queries = adapter.get_evaluation_queries()
    print(f"Evaluation queries: {len(queries)}")
    
    if queries:
        print(f"\nFirst query preview:")
        query = queries[0]
        print(f"Query: {query['query'][:100]}...")
        print(f"Expected docs: {query['expected_docs']}")
        print(f"Query type: {query['query_type']}")
        print(f"Description: {query.get('description', 'N/A')}")

if __name__ == "__main__":
    test_adapter()
