#!/usr/bin/env python3
"""
Simple test script to verify the StackOverflow adapter produces documents.
"""
from pipelines.contracts import DatasetSplit
from pipelines.adapters.stackoverflow import StackOverflowAdapter
import os
import sys

# Add the project root to the Python path
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')


def test_adapter():
    print("=== Testing StackOverflow Adapter ===")

    # Initialize adapter
    dataset_path = "/home/spiros/Desktop/Thesis/datasets/sosum/"
    adapter = StackOverflowAdapter(dataset_path=dataset_path)

    print(f"✓ Adapter initialized from: {dataset_path}")
    print(f"✓ Source name: {adapter.source_name}")
    print(f"✓ Version: {adapter.version}")

    # Test reading rows
    print("\n=== Testing read_rows() ===")
    rows = list(adapter.read_rows(DatasetSplit.ALL))
    print(f"✓ Total rows read: {len(rows)}")

    # Count questions vs answers
    questions = [row for row in rows if row.post_type == "question"]
    answers = [row for row in rows if row.post_type == "answer"]

    print(f"✓ Questions: {len(questions)}")
    print(f"✓ Answers: {len(answers)}")

    # Show a sample question and answer
    if questions:
        sample_q = questions[0]
        print(f"\nSample Question:")
        print(f"  ID: {sample_q.external_id}")
        print(f"  Title: {sample_q.title[:100]}...")
        print(f"  Tags: {sample_q.tags}")
        print(f"  Related posts: {sample_q.related_posts}")

    if answers:
        sample_a = answers[0]
        print(f"\nSample Answer:")
        print(f"  ID: {sample_a.external_id}")
        print(f"  Body length: {len(sample_a.body)}")
        print(f"  Has summary: {bool(sample_a.summary)}")

    # Test converting to documents
    print("\n=== Testing to_documents() ===")
    documents = adapter.to_documents(rows, DatasetSplit.ALL)
    print(f"✓ Documents created: {len(documents)}")

    if documents:
        # Show first document
        doc = documents[0]
        print(f"\nSample Document:")
        print(f"  Content length: {len(doc.page_content)}")
        print(f"  Metadata keys: {list(doc.metadata.keys())}")
        print(f"  External ID: {doc.metadata.get('external_id')}")
        print(f"  Post type: {doc.metadata.get('post_type')}")
        print(
            f"  Has question context: {doc.metadata.get('has_question_context', False)}")
        print(f"  Content preview: {doc.page_content[:200]}...")
    else:
        print("❌ No documents were created!")

        # Debug: Let's see what's happening with the first few answers
        print("\n=== Debug Info ===")
        for i, answer in enumerate(answers[:3]):
            print(f"\nAnswer {i+1}:")
            print(f"  ID: {answer.external_id}")
            print(f"  Body empty: {not answer.body.strip()}")

            # Check for matching questions
            answer_id_num = answer.external_id.replace('a_', '')
            matching_questions = []
            for q in questions:
                if answer_id_num in q.related_posts:
                    matching_questions.append(q.external_id)
            print(f"  Matching questions: {matching_questions}")

    return len(documents) > 0


if __name__ == "__main__":
    success = test_adapter()
    if success:
        print("\n✅ Adapter test PASSED")
    else:
        print("\n❌ Adapter test FAILED")
        sys.exit(1)
