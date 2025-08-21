#!/usr/bin/env python3
"""
Detailed analysis of the new adapter to check question-answer linking.
"""

from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.contracts import DatasetSplit

def analyze_linking():
    adapter = StackOverflowAdapter("/home/spiros/Desktop/Thesis/Thesis/datasets/sosum")
    
    print("=== Analyzing Question-Answer Linking ===")
    
    # Get all documents
    rows = list(adapter.read_rows(DatasetSplit.ALL))
    documents = adapter.to_documents(rows, DatasetSplit.ALL)
    
    # Count documents with/without question context
    with_context = 0
    without_context = 0
    sample_with_context = None
    sample_without_context = None
    
    for doc in documents:
        if doc.metadata.get('has_question_context'):
            with_context += 1
            if sample_with_context is None:
                sample_with_context = doc
        else:
            without_context += 1
            if sample_without_context is None:
                sample_without_context = doc
    
    print(f"Documents with question context: {with_context}")
    print(f"Documents without question context: {without_context}")
    print(f"Context rate: {with_context/len(documents)*100:.1f}%")
    
    if sample_with_context:
        print(f"\n=== SAMPLE WITH CONTEXT ===")
        print(f"ID: {sample_with_context.metadata['external_id']}")
        print(f"Tags: {sample_with_context.metadata.get('tags', [])}")
        print(f"Title: {sample_with_context.metadata.get('title', 'N/A')}")
        print(f"Content preview:\n{sample_with_context.page_content[:300]}...")
    
    if sample_without_context:
        print(f"\n=== SAMPLE WITHOUT CONTEXT ===")
        print(f"ID: {sample_without_context.metadata['external_id']}")
        print(f"Content preview:\n{sample_without_context.page_content[:300]}...")
    
    # Check evaluation queries
    queries = adapter.get_evaluation_queries()
    print(f"\n=== Evaluation Queries ===")
    print(f"Total queries: {len(queries)}")
    
    query_types = {}
    for q in queries:
        qtype = q['query_type']
        query_types[qtype] = query_types.get(qtype, 0) + 1
    
    for qtype, count in query_types.items():
        print(f"  {qtype}: {count}")

if __name__ == "__main__":
    analyze_linking()
