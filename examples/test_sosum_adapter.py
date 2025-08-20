#!/usr/bin/env python3
"""
Simple test of SOSum adapter without full pipeline dependencies.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_sosum_adapter():
    """Test SOSum adapter basic functionality."""
    try:
        from pipelines.adapters.stackoverflow import StackOverflowAdapter
        print(" Successfully imported StackOverflowAdapter")
    except ImportError as e:
        print(f" Import failed: {e}")
        return False
    
    # Test adapter creation
    sosum_path = "../datasets/sosum"
    
    if not Path(sosum_path).exists():
        print(f" Dataset not found at {sosum_path}")
        return False
    
    try:
        adapter = StackOverflowAdapter(sosum_path, version="1.0.0")
        print(f" Created adapter for: {adapter.source_name}")
    except Exception as e:
        print(f" Adapter creation failed: {e}")
        return False
    
    # Test reading rows
    try:
        print(" Testing row reading...")
        rows = list(adapter.read_rows())
        print(f" Read {len(rows)} rows total")
        
        # Count by type
        questions = [r for r in rows if r.post_type == "question"]
        answers = [r for r in rows if r.post_type == "answer"]
        
        print(f"   Questions: {len(questions)}")
        print(f"   Answers: {len(answers)}")
        
        # Show sample
        if rows:
            sample = rows[0]
            print(f"\n Sample row ({sample.post_type}):")
            print(f"   ID: {sample.external_id}")
            if sample.title:
                print(f"   Title: {sample.title[:60]}...")
            print(f"   Body: {sample.body[:80]}...")
            if sample.summary:
                print(f"   Summary: {sample.summary[:60]}...")
            print(f"   Tags: {sample.tags}")
        
    except Exception as e:
        print(f" Row reading failed: {e}")
        return False
    
    # Test document conversion
    try:
        print("\n Testing document conversion...")
        sample_rows = rows[:3]  # Test with first 3 rows
        documents = adapter.to_documents(sample_rows)
        print(f" Converted {len(sample_rows)} rows to {len(documents)} documents")
        
        if documents:
            doc = documents[0]
            print(f"\n Sample document:")
            print(f"   Content length: {len(doc.page_content)} chars")
            print(f"   Content preview: {doc.page_content[:100]}...")
            print(f"   Metadata keys: {list(doc.metadata.keys())}")
            print(f"   External ID: {doc.metadata.get('external_id')}")
            print(f"   Post type: {doc.metadata.get('post_type')}")
        
    except Exception as e:
        print(f" Document conversion failed: {e}")
        return False
    
    # Test evaluation queries
    try:
        print("\n Testing evaluation queries...")
        eval_queries = adapter.get_evaluation_queries()
        print(f" Generated {len(eval_queries)} evaluation queries")
        
        if eval_queries:
            query = eval_queries[0]
            print(f"\n Sample evaluation query:")
            print(f"   Query: {query['query'][:60]}...")
            print(f"   Expected docs: {query['expected_docs']}")
            print(f"   Query type: {query['query_type']}")
            print(f"   Query ID: {query['query_id']}")
        
    except Exception as e:
        print(f"Evaluation query generation failed: {e}")
        return False
    
    print(f"\nAll tests passed! SOSum adapter is working correctly.")
    print(f"   Ready to ingest {len(questions)} questions and {len(answers)} answers")
    
    return True

if __name__ == "__main__":
    print(" SOSum Adapter Test")
    print("====================")
    
    success = test_sosum_adapter()
    
    if success:
        print("\n Next steps:")
        print("   # Dry run ingestion (needs full pipeline dependencies)")
        print("   python bin/ingest.py ingest stackoverflow datasets/sosum --dry-run --max-docs 10")
    else:
        print("\n Fix the issues above before proceeding")
    
    sys.exit(0 if success else 1)
