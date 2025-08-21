#!/usr/bin/env python3
"""
Test Stack Overflow adapter to see what it's actually reading.
"""
import sys
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.contracts import DatasetSplit

def test_adapter():
    """Test what the adapter is actually reading."""
    
    adapter = StackOverflowAdapter("datasets/sosum", "1.0.0")
    
    print("🔍 Testing Stack Overflow adapter...")
    print(f"📁 Dataset path: {adapter.dataset_path}")
    print(f"📄 Question file: {adapter.question_file}")
    print(f"📄 Answer file: {adapter.answer_file}")
    
    # Read raw rows
    print("\n📝 Reading raw rows...")
    rows = list(adapter.read_rows())
    
    questions = [r for r in rows if r.post_type == "question"]
    answers = [r for r in rows if r.post_type == "answer"]
    
    print(f"   Questions: {len(questions)}")
    print(f"   Answers: {len(answers)}")
    
    # Show samples
    if questions:
        print(f"\n📄 Sample Question:")
        q = questions[0]
        print(f"   ID: {q.external_id}")
        print(f"   Title: {q.title}")
        print(f"   Body: {q.body[:100]}...")
    
    if answers:
        print(f"\n💬 Sample Answer:")
        a = answers[0]
        print(f"   ID: {a.external_id}")
        print(f"   Body: {a.body[:100]}...")
    else:
        print("\n❌ No answers found!")
    
    # Convert to documents
    print(f"\n📋 Converting to documents...")
    docs = adapter.to_documents(rows)
    
    question_docs = [d for d in docs if d.metadata.get("labels", {}).get("post_type") == "question"]
    answer_docs = [d for d in docs if d.metadata.get("labels", {}).get("post_type") == "answer"]
    
    print(f"   Question documents: {len(question_docs)}")
    print(f"   Answer documents: {len(answer_docs)}")
    
    if answer_docs:
        print(f"\n💬 Sample Answer Document:")
        doc = answer_docs[0]
        print(f"   Content: {doc.page_content[:200]}...")
        print(f"   Metadata: {doc.metadata}")

if __name__ == "__main__":
    test_adapter()
