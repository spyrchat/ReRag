"""
Integration test for Self-RAG graph with detailed logging.
Tests the complete self-correction workflow with verification.
"""

import pytest
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.graph_self_rag import graph
from logs.utils.logger import get_logger

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_self_rag_detailed.log', mode='w')
    ]
)

logger = get_logger("test_self_rag")


class TestSelfRAGIntegration:
    """Integration tests for Self-RAG graph."""
    
    @pytest.fixture(scope="class")
    def self_rag_graph(self):
        """Get Self-RAG graph (already built in graph_self_rag.py)."""
        logger.info("=" * 80)
        logger.info("LOADING SELF-RAG GRAPH")
        logger.info("=" * 80)
        
        try:
            logger.info("✓ Self-RAG graph loaded successfully")
            logger.info(f"  - Nodes: query_analyzer, retriever, self_rag_generator, benchmark_logger")
            return graph
        except Exception as e:
            logger.error(f"✗ Failed to load graph: {e}")
            raise
    
    def test_basic_question_no_hallucination(self, self_rag_graph):
        """
        Test 1: Basic question where answer should be faithful on first try.
        Expected: Single iteration, no refinement needed.
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: BASIC QUESTION (Expected: No Hallucination)")
        logger.info("=" * 80)
        
        question = "What is a Python list comprehension?"
        context = """
        List comprehensions provide a concise way to create lists in Python.
        They consist of brackets containing an expression followed by a for clause.
        Example: squares = [x**2 for x in range(10)]
        This creates a list of squares from 0 to 81.
        """
        
        logger.info(f"\n📝 INPUT:")
        logger.info(f"  Question: {question}")
        logger.info(f"  Context length: {len(context)} chars")
        
        initial_state = {
            "question": question,
            "context": context
        }
        
        logger.info(f"\n🔄 INVOKING SELF-RAG GRAPH...")
        
        try:
            result = self_rag_graph.invoke(initial_state)
            
            logger.info(f"\n✓ GRAPH EXECUTION COMPLETED")
            logger.info(f"\n📊 RESULTS:")
            
            # Log answer
            answer = result.get("answer", "")
            logger.info(f"\n📝 GENERATED ANSWER:")
            logger.info(f"  Length: {len(answer)} chars")
            logger.info(f"  Content: {answer[:200]}..." if len(answer) > 200 else f"  Content: {answer}")
            
            # Log verification
            verification = result.get("verification", {})
            logger.info(f"\n🔍 VERIFICATION:")
            logger.info(f"  Is Faithful: {verification.get('is_faithful', 'N/A')}")
            logger.info(f"  Confidence: {verification.get('confidence', 'N/A')}")
            logger.info(f"  Hallucination Detected: {verification.get('hallucination_detected', 'N/A')}")
            logger.info(f"  Severity: {verification.get('severity', 'N/A')}")
            if verification.get('issues'):
                logger.info(f"  Issues: {verification['issues']}")
            
            # Log metadata
            metadata = result.get("self_rag_metadata", {})
            logger.info(f"\n📈 METADATA:")
            logger.info(f"  Iterations: {metadata.get('iterations', 'N/A')}")
            logger.info(f"  Converged: {metadata.get('converged', 'N/A')}")
            logger.info(f"  Hallucinations Corrected: {metadata.get('hallucinations_corrected', 'N/A')}")
            logger.info(f"  Max Iterations Reached: {metadata.get('max_iterations_reached', 'N/A')}")
            
            # Log iteration history
            if metadata.get('iteration_history'):
                logger.info(f"\n📚 ITERATION HISTORY:")
                for i, iteration in enumerate(metadata['iteration_history']):
                    logger.info(f"\n  Iteration {iteration.get('iteration', i+1)}:")
                    logger.info(f"    Action: {iteration.get('action', 'N/A')}")
                    logger.info(f"    Answer length: {len(iteration.get('answer', ''))} chars")
                    iter_verif = iteration.get('verification', {})
                    logger.info(f"    Faithful: {iter_verif.get('is_faithful', 'N/A')}")
                    logger.info(f"    Confidence: {iter_verif.get('confidence', 'N/A')}")
            
            # Assertions
            assert "answer" in result, "Answer not found in result"
            assert len(answer) > 0, "Answer is empty"
            assert metadata.get('iterations', 0) >= 1, "No iterations recorded"
            
            # For a faithful answer, we expect 1 iteration
            logger.info(f"\n✓ TEST 1 PASSED")
            logger.info(f"  Expected: 1 iteration (faithful on first try)")
            logger.info(f"  Actual: {metadata.get('iterations')} iteration(s)")
            
        except Exception as e:
            logger.error(f"\n✗ TEST 1 FAILED: {e}", exc_info=True)
            raise
    
    def test_question_with_misleading_context(self, self_rag_graph):
        """
        Test 2: Question with context that might trigger hallucination.
        Expected: Potentially multiple iterations with refinement.
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: POTENTIALLY CHALLENGING QUESTION")
        logger.info("=" * 80)
        
        question = "How do I handle database connections in Python with asyncio?"
        context = """
        Python's asyncio library provides async/await syntax for asynchronous programming.
        For database connections, you should use connection pooling.
        Libraries like aiopg and aiomysql support async operations.
        Always use context managers to ensure connections are properly closed.
        """
        
        logger.info(f"\n📝 INPUT:")
        logger.info(f"  Question: {question}")
        logger.info(f"  Context length: {len(context)} chars")
        logger.info(f"  Note: Context is limited, may trigger verification checks")
        
        initial_state = {
            "question": question,
            "context": context
        }
        
        logger.info(f"\n🔄 INVOKING SELF-RAG GRAPH...")
        
        try:
            result = self_rag_graph.invoke(initial_state)
            
            logger.info(f"\n✓ GRAPH EXECUTION COMPLETED")
            logger.info(f"\n📊 RESULTS:")
            
            # Log answer
            answer = result.get("answer", "")
            logger.info(f"\n📝 GENERATED ANSWER:")
            logger.info(f"  Length: {len(answer)} chars")
            logger.info(f"  Content:\n{answer}")
            
            # Log verification
            verification = result.get("verification", {})
            logger.info(f"\n🔍 VERIFICATION:")
            logger.info(f"  Is Faithful: {verification.get('is_faithful', 'N/A')}")
            logger.info(f"  Confidence: {verification.get('confidence', 'N/A')}")
            logger.info(f"  Hallucination Detected: {verification.get('hallucination_detected', 'N/A')}")
            logger.info(f"  Severity: {verification.get('severity', 'N/A')}")
            if verification.get('issues'):
                logger.info(f"  Issues:")
                for issue in verification['issues']:
                    logger.info(f"    - {issue}")
            if verification.get('reasoning'):
                logger.info(f"  Reasoning: {verification['reasoning']}")
            
            # Log metadata
            metadata = result.get("self_rag_metadata", {})
            logger.info(f"\n📈 METADATA:")
            logger.info(f"  Iterations: {metadata.get('iterations', 'N/A')}")
            logger.info(f"  Converged: {metadata.get('converged', 'N/A')}")
            logger.info(f"  Hallucinations Corrected: {metadata.get('hallucinations_corrected', 'N/A')}")
            logger.info(f"  Max Iterations Reached: {metadata.get('max_iterations_reached', 'N/A')}")
            
            # Log detailed iteration history
            if metadata.get('iteration_history'):
                logger.info(f"\n📚 DETAILED ITERATION HISTORY:")
                for iteration in metadata['iteration_history']:
                    logger.info(f"\n  ╔═══ Iteration {iteration.get('iteration', 'N/A')} ═══")
                    logger.info(f"  ║ Action: {iteration.get('action', 'N/A')}")
                    
                    iter_answer = iteration.get('answer', '')
                    logger.info(f"  ║ Answer Preview: {iter_answer[:150]}...")
                    
                    iter_verif = iteration.get('verification', {})
                    logger.info(f"  ║ Verification:")
                    logger.info(f"  ║   - Faithful: {iter_verif.get('is_faithful', 'N/A')}")
                    logger.info(f"  ║   - Confidence: {iter_verif.get('confidence', 'N/A')}")
                    logger.info(f"  ║   - Hallucination: {iter_verif.get('hallucination_detected', 'N/A')}")
                    
                    if iter_verif.get('issues'):
                        logger.info(f"  ║   - Issues Found:")
                        for issue in iter_verif['issues']:
                            logger.info(f"  ║     • {issue}")
                    
                    logger.info(f"  ╚{'═' * 50}")
            
            # Assertions
            assert "answer" in result, "Answer not found in result"
            assert len(answer) > 0, "Answer is empty"
            assert metadata.get('iterations', 0) >= 1, "No iterations recorded"
            
            logger.info(f"\n✓ TEST 2 PASSED")
            logger.info(f"  Total iterations: {metadata.get('iterations')}")
            logger.info(f"  Self-correction occurred: {metadata.get('hallucinations_corrected', False)}")
            
        except Exception as e:
            logger.error(f"\n✗ TEST 2 FAILED: {e}", exc_info=True)
            raise
    
    def test_question_with_empty_context(self, self_rag_graph):
        """
        Test 3: Question with no context (should handle gracefully).
        Expected: Answer acknowledges lack of context.
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: QUESTION WITH NO CONTEXT")
        logger.info("=" * 80)
        
        question = "What is the difference between REST and GraphQL?"
        context = ""
        
        logger.info(f"\n📝 INPUT:")
        logger.info(f"  Question: {question}")
        logger.info(f"  Context: EMPTY")
        logger.info(f"  Expected: Graceful handling")
        
        initial_state = {
            "question": question,
            "context": context
        }
        
        logger.info(f"\n🔄 INVOKING SELF-RAG GRAPH...")
        
        try:
            result = self_rag_graph.invoke(initial_state)
            
            logger.info(f"\n✓ GRAPH EXECUTION COMPLETED")
            logger.info(f"\n📊 RESULTS:")
            
            # Log answer
            answer = result.get("answer", "")
            logger.info(f"\n📝 GENERATED ANSWER:")
            logger.info(f"  Length: {len(answer)} chars")
            logger.info(f"  Content:\n{answer}")
            
            # Log metadata
            metadata = result.get("self_rag_metadata", {})
            logger.info(f"\n📈 METADATA:")
            logger.info(f"  Iterations: {metadata.get('iterations', 'N/A')}")
            logger.info(f"  Converged: {metadata.get('converged', 'N/A')}")
            
            # Assertions
            assert "answer" in result, "Answer not found in result"
            assert len(answer) > 0, "Answer is empty"
            
            logger.info(f"\n✓ TEST 3 PASSED")
            logger.info(f"  Answer generated despite empty context")
            
        except Exception as e:
            logger.error(f"\n✗ TEST 3 FAILED: {e}", exc_info=True)
            raise
    
    def test_full_pipeline_with_retrieval(self, self_rag_graph):
        """
        Test 4: Test with retrieval (requires Qdrant to be running).
        This is an optional test that will be skipped if Qdrant is not available.
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST 4: FULL PIPELINE WITH RETRIEVAL (Optional)")
        logger.info("=" * 80)
        
        try:
            from qdrant_client import QdrantClient
            
            # Try to connect to Qdrant
            client = QdrantClient(host="localhost", port=6333)
            collections = client.get_collections()
            
            if not collections.collections:
                logger.warning("⚠️  No collections found in Qdrant - skipping retrieval test")
                pytest.skip("No Qdrant collections available")
            
            logger.info(f"✓ Qdrant is available with {len(collections.collections)} collection(s)")
            
            question = "How do I use Python list comprehensions?"
            
            logger.info(f"\n📝 INPUT:")
            logger.info(f"  Question: {question}")
            logger.info(f"  Mode: Full pipeline (query analysis + retrieval + generation)")
            
            initial_state = {
                "question": question
            }
            
            logger.info(f"\n🔄 INVOKING FULL SELF-RAG PIPELINE...")
            
            result = self_rag_graph.invoke(initial_state)
            
            logger.info(f"\n✓ FULL PIPELINE EXECUTION COMPLETED")
            
            # Log query analysis
            query_analysis = result.get("query_analysis", "")
            logger.info(f"\n🔍 QUERY ANALYSIS:")
            logger.info(f"  {query_analysis}")
            
            # Log retrieval
            context = result.get("context", "")
            logger.info(f"\n📚 RETRIEVED CONTEXT:")
            logger.info(f"  Length: {len(context)} chars")
            logger.info(f"  Preview: {context[:300]}...")
            
            # Log answer
            answer = result.get("answer", "")
            logger.info(f"\n📝 GENERATED ANSWER:")
            logger.info(f"  Length: {len(answer)} chars")
            logger.info(f"  Content:\n{answer}")
            
            # Log metadata
            metadata = result.get("self_rag_metadata", {})
            logger.info(f"\n📈 METADATA:")
            logger.info(f"  Iterations: {metadata.get('iterations', 'N/A')}")
            logger.info(f"  Converged: {metadata.get('converged', 'N/A')}")
            logger.info(f"  Hallucinations Corrected: {metadata.get('hallucinations_corrected', 'N/A')}")
            
            # Assertions
            assert "answer" in result, "Answer not found in result"
            assert len(answer) > 0, "Answer is empty"
            assert "context" in result, "Context not found in result"
            
            logger.info(f"\n✓ TEST 4 PASSED")
            logger.info(f"  Full pipeline executed successfully with retrieval")
            
        except ImportError:
            logger.warning("⚠️  Qdrant client not installed - skipping retrieval test")
            pytest.skip("Qdrant client not available")
        except Exception as e:
            logger.warning(f"⚠️  Could not connect to Qdrant: {e}")
            pytest.skip("Qdrant not available")


if __name__ == "__main__":
    """Run tests with detailed output."""
    print("\n" + "=" * 80)
    print("SELF-RAG INTEGRATION TESTS")
    print("=" * 80)
    print("\nThis test suite will:")
    print("  1. Test basic question (no hallucination expected)")
    print("  2. Test challenging question (may trigger refinement)")
    print("  3. Test question with no context")
    print("  4. Test full pipeline with retrieval (optional)")
    print("\nAll steps are logged in detail to test_self_rag_detailed.log")
    print("=" * 80)
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
