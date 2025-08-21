#!/usr/bin/env python3
"""
Minimal SOSum ingestion test without heavy ML dependencies.
Tests the adapter and basic pipeline functionality.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_sosum_adapter_minimal():
    """Test SOSum adapter with minimal dependencies."""
    print("🧪 SOSum Adapter Minimal Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        from pipelines.adapters.stackoverflow import StackOverflowAdapter
        from pipelines.contracts import DatasetSplit
        print("✅ Successfully imported adapter classes")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test adapter creation
    sosum_path = "../datasets/sosum"
    if not Path(sosum_path).exists():
        print(f"❌ Dataset not found at {sosum_path}")
        return False
    
    try:
        adapter = StackOverflowAdapter(sosum_path, version="1.0.0")
        print(f"✅ Created adapter: {adapter.source_name} v{adapter.version}")
    except Exception as e:
        print(f"❌ Adapter creation failed: {e}")
        return False
    
    # Test row reading
    try:
        print("\n🔍 Testing row reading...")
        rows = list(adapter.read_rows(DatasetSplit.ALL))
        
        questions = [r for r in rows if r.post_type == "question"]
        answers = [r for r in rows if r.post_type == "answer"]
        
        print(f"✅ Successfully read {len(rows)} total rows")
        print(f"   Questions: {len(questions)}")
        print(f"   Answers: {len(answers)}")
        
        # Show sample
        if rows:
            sample = rows[0]
            print(f"\n📋 Sample {sample.post_type}:")
            print(f"   ID: {sample.external_id}")
            print(f"   Title: {sample.title[:50]}..." if sample.title else "   (No title)")
            print(f"   Body: {sample.body[:80]}...")
            print(f"   Tags: {sample.tags}")
            if sample.summary:
                print(f"   Summary: {sample.summary[:50]}...")
        
    except Exception as e:
        print(f"❌ Row reading failed: {e}")
        return False
    
    # Test document conversion
    try:
        print("\n📄 Testing document conversion...")
        sample_rows = rows[:5]
        documents = adapter.to_documents(sample_rows)
        
        print(f"✅ Converted {len(sample_rows)} rows to {len(documents)} documents")
        
        if documents:
            doc = documents[0]
            print(f"\n📄 Sample document:")
            print(f"   Content length: {len(doc.page_content)} characters")
            print(f"   Content preview: {doc.page_content[:100]}...")
            print(f"   Metadata: {list(doc.metadata.keys())}")
        
    except Exception as e:
        print(f"❌ Document conversion failed: {e}")
        return False
    
    # Test evaluation queries
    try:
        print("\n❓ Testing evaluation queries...")
        eval_queries = adapter.get_evaluation_queries()
        
        print(f"✅ Generated {len(eval_queries)} evaluation queries")
        
        if eval_queries:
            query = eval_queries[0]
            print(f"\n❓ Sample evaluation query:")
            print(f"   Query: {query['query'][:60]}...")
            print(f"   Expected docs: {query['expected_docs']}")
            print(f"   Query type: {query['query_type']}")
        
    except Exception as e:
        print(f"❌ Evaluation query generation failed: {e}")
        return False
    
    print(f"\n🎉 All adapter tests passed!")
    print(f"   Dataset: {len(questions)} questions + {len(answers)} answers")
    print(f"   Ready for full pipeline ingestion")
    
    return True

def test_lightweight_processing():
    """Test lightweight document processing."""
    print("\n🔧 Testing Lightweight Processing")
    print("=" * 40)
    
    try:
        from pipelines.adapters.stackoverflow import StackOverflowAdapter
        
        adapter = StackOverflowAdapter("../datasets/sosum")
        
        # Process first 10 documents
        rows = list(adapter.read_rows())[:10]
        documents = adapter.to_documents(rows)
        
        print(f"📄 Processed {len(documents)} documents for ingestion")
        
        # Create simple JSON export
        import json
        from datetime import datetime
        
        output_data = []
        for doc in documents:
            output_data.append({
                "id": doc.metadata["external_id"],
                "content": doc.page_content,
                "metadata": doc.metadata,
                "processed_at": datetime.now().isoformat()
            })
        
        # Save to file
        output_file = "output/sosum_sample.json"
        Path("output").mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved sample data to: {output_file}")
        print(f"   Ready for vector store ingestion")
        
        return True
        
    except Exception as e:
        print(f"❌ Lightweight processing failed: {e}")
        return False

def suggest_next_steps():
    """Suggest next steps for full ingestion."""
    print("\n🚀 Next Steps for Full Ingestion")
    print("=" * 40)
    
    print("Option 1: Install CPU-only PyTorch (smaller):")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("   pip install sentence-transformers")
    print()
    
    print("Option 2: Use existing embeddings infrastructure:")
    print("   # Check if your existing embedding setup works")
    print("   python -c \"from embedding.factory import get_embedder; print('✅ Embeddings available')\"")
    print()
    
    print("Option 3: Test with mock embeddings:")
    print("   # Create a mock embedder for testing")
    print("   python bin/ingest.py ingest stackoverflow ../datasets/sosum --dry-run --max-docs 3")
    print()
    
    print("Once dependencies are resolved:")
    print("   # Full ingestion")
    print("   python bin/ingest.py ingest stackoverflow ../datasets/sosum")
    print("   # Evaluation")
    print("   python bin/ingest.py evaluate stackoverflow ../datasets/sosum")

def main():
    """Main test function."""
    success = True
    
    # Test adapter functionality
    if not test_sosum_adapter_minimal():
        success = False
    
    # Test lightweight processing
    if success and not test_lightweight_processing():
        success = False
    
    # Suggest next steps
    suggest_next_steps()
    
    if success:
        print(f"\n✅ SOSum adapter is working perfectly!")
        print(f"   Just need to resolve PyTorch installation for embeddings")
    else:
        print(f"\n❌ Some tests failed - check the output above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
