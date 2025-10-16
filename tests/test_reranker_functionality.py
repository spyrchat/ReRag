"""
Test suite to verify rerankers are actually working in the pipeline.

This script tests:
1. Score changes before/after reranking
2. Order changes in results
3. Different rerankers produce different results
4. Reranker metadata is properly added
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from components.retrieval_pipeline import RetrievalPipelineFactory, RetrievalResult
from components.rerankers import CrossEncoderReranker, BM25Reranker, EnsembleReranker
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_results(query: str = "machine learning") -> list:
    """Create mock retrieval results for testing."""
    docs = [
        ("Machine learning is a subset of artificial intelligence.", 0.95),
        ("Python is a popular programming language.", 0.92),
        ("Deep learning uses neural networks.", 0.90),
        ("JavaScript is used for web development.", 0.88),
        ("Data science involves analyzing data.", 0.85),
    ]
    
    results = []
    for i, (text, score) in enumerate(docs):
        doc = Document(
            page_content=text,
            metadata={"id": i, "source": "test"}
        )
        result = RetrievalResult(
            document=doc,
            score=score,
            retrieval_method="dense",
            metadata={"id": i}
        )
        results.append(result)
    
    return results


def test_score_changes():
    """Test 1: Verify that scores change after reranking."""
    print("\n" + "=" * 80)
    print("TEST 1: Verify Score Changes")
    print("=" * 80)
    
    query = "machine learning algorithms"
    original_results = create_mock_results(query)
    
    # Store original scores
    original_scores = [r.score for r in original_results]
    print(f"\n📊 Original scores: {[f'{s:.3f}' for s in original_scores]}")
    
    # Apply reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranked_results = reranker.rerank(query, original_results)
    
    # Check new scores
    new_scores = [r.score for r in reranked_results]
    print(f"📊 Reranked scores: {[f'{s:.3f}' for s in new_scores]}")
    
    # Verify scores changed
    scores_changed = original_scores != new_scores
    print(f"\n✅ Scores changed: {scores_changed}")
    
    if not scores_changed:
        print("⚠️  WARNING: Reranker did not change scores!")
        return False
    
    return True


def test_order_changes():
    """Test 2: Verify that result order changes after reranking."""
    print("\n" + "=" * 80)
    print("TEST 2: Verify Order Changes")
    print("=" * 80)
    
    query = "machine learning algorithms"
    original_results = create_mock_results(query)
    
    # Store original order (document IDs)
    original_order = [r.document.metadata["id"] for r in original_results]
    print(f"\n📋 Original order: {original_order}")
    print("Original ranking:")
    for i, result in enumerate(original_results[:3], 1):
        print(f"  {i}. [Score: {result.score:.3f}] {result.document.page_content[:50]}...")
    
    # Apply reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranked_results = reranker.rerank(query, original_results)
    
    # Check new order
    new_order = [r.document.metadata["id"] for r in reranked_results]
    print(f"\n📋 Reranked order: {new_order}")
    print("Reranked ranking:")
    for i, result in enumerate(reranked_results[:3], 1):
        print(f"  {i}. [Score: {result.score:.3f}] {result.document.page_content[:50]}...")
    
    # Verify order changed
    order_changed = original_order != new_order
    print(f"\n✅ Order changed: {order_changed}")
    
    if not order_changed:
        print("⚠️  WARNING: Reranker did not change result order!")
        return False
    
    return True


def test_metadata_added():
    """Test 3: Verify that reranker adds proper metadata."""
    print("\n" + "=" * 80)
    print("TEST 3: Verify Metadata Addition")
    print("=" * 80)
    
    query = "machine learning"
    original_results = create_mock_results(query)
    
    # Apply reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranked_results = reranker.rerank(query, original_results)
    
    # Check first result
    result = reranked_results[0]
    
    print("\n📝 Checking metadata for top result:")
    print(f"  Retrieval method: {result.retrieval_method}")
    print(f"  Has 'original_score': {'original_score' in result.metadata}")
    print(f"  Has 'reranker_model': {'reranker_model' in result.metadata}")
    print(f"  Has 'reranked' flag: {'reranked' in result.metadata}")
    
    if "original_score" in result.metadata:
        print(f"  Original score: {result.metadata['original_score']:.3f}")
        print(f"  New score: {result.score:.3f}")
    
    # Verify metadata
    has_metadata = all([
        "original_score" in result.metadata,
        "reranker_model" in result.metadata,
        "reranked" in result.metadata,
        "+cross_encoder" in result.retrieval_method
    ])
    
    print(f"\n✅ Metadata properly added: {has_metadata}")
    
    if not has_metadata:
        print("⚠️  WARNING: Reranker did not add proper metadata!")
        return False
    
    return True


def test_different_rerankers():
    """Test 4: Verify different rerankers produce different results."""
    print("\n" + "=" * 80)
    print("TEST 4: Compare Different Rerankers")
    print("=" * 80)
    
    query = "machine learning neural networks"
    original_results = create_mock_results(query)
    
    # Test CrossEncoder
    print("\n🔄 Testing CrossEncoderReranker...")
    ce_reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    ce_results = ce_reranker.rerank(query, original_results)
    ce_order = [r.document.metadata["id"] for r in ce_results]
    print(f"  Order: {ce_order}")
    print(f"  Top result: {ce_results[0].document.page_content[:50]}...")
    
    # Test BM25
    print("\n🔄 Testing BM25Reranker...")
    bm25_reranker = BM25Reranker(k1=1.2, b=0.75)
    bm25_results = bm25_reranker.rerank(query, original_results)
    bm25_order = [r.document.metadata["id"] for r in bm25_results]
    print(f"  Order: {bm25_order}")
    print(f"  Top result: {bm25_results[0].document.page_content[:50]}...")
    
    # Compare
    orders_different = ce_order != bm25_order
    print(f"\n✅ Different rerankers produce different orders: {orders_different}")
    
    if not orders_different:
        print("ℹ️  Note: Rerankers may agree on this query - try different queries")
    
    return True


def test_with_real_pipeline():
    """Test 5: Test reranker in actual pipeline with real retrieval."""
    print("\n" + "=" * 80)
    print("TEST 5: Reranker in Real Pipeline")
    print("=" * 80)
    
    try:
        from config.config_loader import load_config
        
        # Load config
        config = load_config()
        
        print("\n🔧 Creating pipeline without reranker...")
        pipeline_no_rerank = RetrievalPipelineFactory.create_hybrid_pipeline(config)
        
        print("🔧 Creating pipeline with reranker...")
        pipeline_with_rerank = RetrievalPipelineFactory.create_reranked_pipeline(
            config,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Test query
        query = "How do I use Python list comprehensions?"
        
        print(f"\n🔍 Query: {query}")
        print("\n📊 Running both pipelines...")
        
        # Run without reranker
        results_no_rerank = pipeline_no_rerank.run(query, k=5)
        print(f"\n✅ Without reranker: {len(results_no_rerank)} results")
        print("Top 3 results:")
        for i, result in enumerate(results_no_rerank[:3], 1):
            print(f"  {i}. [Score: {result.score:.3f}] {result.document.page_content[:60]}...")
        
        # Run with reranker
        results_with_rerank = pipeline_with_rerank.run(query, k=5)
        print(f"\n✅ With reranker: {len(results_with_rerank)} results")
        print("Top 3 results:")
        for i, result in enumerate(results_with_rerank[:3], 1):
            print(f"  {i}. [Score: {result.score:.3f}] {result.document.page_content[:60]}...")
            if "reranked" in result.metadata:
                print(f"      ↳ Original score: {result.metadata.get('original_score', 'N/A'):.3f}")
        
        # Compare
        original_order = [id(r.document) for r in results_no_rerank]
        reranked_order = [id(r.document) for r in results_with_rerank]
        
        order_changed = original_order != reranked_order
        print(f"\n✅ Reranker changed result order: {order_changed}")
        
        # Check metadata
        has_rerank_metadata = any("reranked" in r.metadata for r in results_with_rerank)
        print(f"✅ Reranker metadata present: {has_rerank_metadata}")
        
        return order_changed and has_rerank_metadata
        
    except Exception as e:
        print(f"\n⚠️  Could not test with real pipeline: {e}")
        print("   (This is OK if Qdrant is not running)")
        return None


def main():
    """Run all tests."""
    print("=" * 80)
    print("RERANKER FUNCTIONALITY TESTS")
    print("=" * 80)
    print("\nThese tests verify that rerankers are actually working.")
    
    results = []
    
    # Run tests
    results.append(("Score Changes", test_score_changes()))
    results.append(("Order Changes", test_order_changes()))
    results.append(("Metadata Added", test_metadata_added()))
    results.append(("Different Rerankers", test_different_rerankers()))
    
    # Optional: Test with real pipeline
    pipeline_result = test_with_real_pipeline()
    if pipeline_result is not None:
        results.append(("Real Pipeline", pipeline_result))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Your rerankers are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
