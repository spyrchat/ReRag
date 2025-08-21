#!/usr/bin/env python3
"""
Debug Stack Overflow adapter document creation.
"""
import sys
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.contracts import DatasetSplit

def debug_document_creation():
    """Debug what happens during document creation."""
    
    adapter = StackOverflowAdapter("datasets/sosum", "1.0.0")
    
    # Read a few rows
    rows = list(adapter.read_rows())
    
    print(f"📊 Total rows: {len(rows)}")
    
    # Take first 5 rows for debugging
    test_rows = rows[:5]
    
    for i, row in enumerate(test_rows):
        print(f"\n🔍 Row {i+1}:")
        print(f"   Type: {row.post_type}")
        print(f"   ID: {row.external_id}")
        print(f"   Title: '{row.title}'")
        print(f"   Body length: {len(row.body)}")
        print(f"   Body content: {repr(row.body[:100])}")
        
        # Create content manually to see what happens
        if row.post_type == "question":
            content = f"Title: {row.title}\n\nQuestion: {row.body}"
            doc_type = "question"
        else:  # answer
            content = row.body
            if row.summary:
                content = f"Answer: {row.body}\n\nSummary: {row.summary}"
            doc_type = "answer"
        
        print(f"   Generated content length: {len(content)}")
        print(f"   Content preview: {repr(content[:100])}")
        print(f"   Content stripped: '{content.strip()}'")
        print(f"   Is empty after strip: {not content.strip()}")
    
    # Now try the actual conversion
    print(f"\n📋 Converting all rows to documents...")
    docs = adapter.to_documents(rows)
    print(f"   Total documents created: {len(docs)}")
    
    if docs:
        print(f"\n📄 First document:")
        doc = docs[0]
        print(f"   Content: {doc.page_content[:200]}...")
        print(f"   Metadata keys: {list(doc.metadata.keys())}")

if __name__ == "__main__":
    debug_document_creation()
