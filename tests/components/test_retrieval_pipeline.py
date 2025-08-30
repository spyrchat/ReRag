#!/usr/bin/env python3
"""
Unit tests for the retrieval pipeline components.
"""

import unittest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from components.retrieval_pipeline import (
    RetrievalPipeline,
    RetrievalResult,
    RetrievalPipelineFactory,
    Retriever,
    Reranker,
    Filter
)


class MockRetriever(Retriever):
    """Mock retriever for testing."""

    def __init__(self, results=None):
        self.results = results or []

    @property
    def component_name(self) -> str:
        return "mock_retriever"

    def retrieve(self, query: str, k: int = 10, **kwargs) -> list:
        return self.results[:k]


class MockReranker(Reranker):
    """Mock reranker for testing."""

    def __init__(self, score_multiplier=1.0):
        self.score_multiplier = score_multiplier

    @property
    def component_name(self) -> str:
        return "mock_reranker"

    def rerank(self, query: str, results: list, **kwargs) -> list:
        # Multiply scores by multiplier
        reranked = []
        for result in results:
            new_result = RetrievalResult(
                document=result.document,
                score=result.score * self.score_multiplier,
                retrieval_method=f"{result.retrieval_method}+reranked",
                metadata={**result.metadata, "reranked": True}
            )
            reranked.append(new_result)
        return sorted(reranked, key=lambda x: x.score, reverse=True)


class MockFilter(Filter):
    """Mock filter for testing."""

    def __init__(self, min_score=0.0):
        self.min_score = min_score

    @property
    def component_name(self) -> str:
        return "mock_filter"

    def filter(self, query: str, results: list, **kwargs) -> list:
        # Filter by minimum score
        filtered = [r for r in results if r.score >= self.min_score]
        for result in filtered:
            result.metadata["filtered"] = True
        return filtered


class TestRetrievalResult(unittest.TestCase):
    """Test the RetrievalResult data class."""

    def test_creation(self):
        """Test basic creation of RetrievalResult."""
        doc = Document(page_content="test", metadata={"id": "1"})
        result = RetrievalResult(
            document=doc,
            score=0.85,
            retrieval_method="dense",
            metadata={"source": "test"}
        )

        self.assertEqual(result.document, doc)
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.retrieval_method, "dense")
        self.assertEqual(result.metadata["source"], "test")

    def test_default_metadata(self):
        """Test default empty metadata."""
        doc = Document(page_content="test")
        result = RetrievalResult(
            document=doc,
            score=0.5,
            retrieval_method="sparse"
        )

        self.assertEqual(result.metadata, {})


class TestRetrievalPipeline(unittest.TestCase):
    """Test the RetrievalPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test documents
        self.documents = [
            Document(page_content="First document", metadata={"id": "1"}),
            Document(page_content="Second document", metadata={"id": "2"}),
            Document(page_content="Third document", metadata={"id": "3"}),
        ]

        # Create test results
        self.test_results = [
            RetrievalResult(
                document=doc,
                score=0.8 - i * 0.1,  # Decreasing scores
                retrieval_method="mock",
                metadata={"original": True}
            )
            for i, doc in enumerate(self.documents)
        ]

        # Create mock retriever
        self.retriever = MockRetriever(self.test_results)

        # Create pipeline
        self.pipeline = RetrievalPipeline(retriever=self.retriever)

    def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        self.assertEqual(self.pipeline.retriever, self.retriever)
        self.assertEqual(len(self.pipeline.components), 0)

    def test_add_component(self):
        """Test adding components to pipeline."""
        reranker = MockReranker()
        filter_comp = MockFilter(min_score=0.5)

        self.pipeline.add_component(reranker)
        self.pipeline.add_component(filter_comp)

        self.assertEqual(len(self.pipeline.components), 2)
        self.assertEqual(self.pipeline.components[0], reranker)
        self.assertEqual(self.pipeline.components[1], filter_comp)

    def test_remove_component(self):
        """Test removing components by name."""
        reranker = MockReranker()
        self.pipeline.add_component(reranker)

        self.assertEqual(len(self.pipeline.components), 1)

        removed = self.pipeline.remove_component("mock_reranker")
        self.assertTrue(removed)
        self.assertEqual(len(self.pipeline.components), 0)

        # Try removing non-existent component
        removed = self.pipeline.remove_component("non_existent")
        self.assertFalse(removed)

    def test_run_basic(self):
        """Test basic pipeline run without components."""
        results = self.pipeline.run("test query", k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].score, 0.8)
        self.assertEqual(results[1].score, 0.7)
        self.assertEqual(results[2].score, 0.6)

    def test_run_with_reranker(self):
        """Test pipeline run with reranker."""
        reranker = MockReranker(score_multiplier=2.0)
        self.pipeline.add_component(reranker)

        results = self.pipeline.run("test query", k=3)

        self.assertEqual(len(results), 3)
        # Scores should be doubled by reranker
        self.assertEqual(results[0].score, 1.6)  # 0.8 * 2
        self.assertEqual(results[1].score, 1.4)  # 0.7 * 2
        self.assertEqual(results[2].score, 1.2)  # 0.6 * 2

        # Check metadata
        for result in results:
            self.assertTrue(result.metadata["reranked"])

    def test_run_with_filter(self):
        """Test pipeline run with filter."""
        filter_comp = MockFilter(min_score=0.7)
        self.pipeline.add_component(filter_comp)

        results = self.pipeline.run("test query", k=5)

        # Only results with score >= 0.7 should remain
        self.assertEqual(len(results), 2)  # 0.8 and 0.7, not 0.6
        self.assertEqual(results[0].score, 0.8)
        self.assertEqual(results[1].score, 0.7)

        # Check metadata
        for result in results:
            self.assertTrue(result.metadata["filtered"])

    def test_run_with_multiple_components(self):
        """Test pipeline run with multiple components."""
        reranker = MockReranker(score_multiplier=1.5)
        filter_comp = MockFilter(min_score=1.0)  # Will filter after reranking

        self.pipeline.add_component(reranker)
        self.pipeline.add_component(filter_comp)

        results = self.pipeline.run("test query", k=5)

        # Scores after reranking: 1.2, 1.05, 0.9
        # After filtering (>= 1.0): 1.2, 1.05
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].score, 1.2, places=1)
        self.assertAlmostEqual(results[1].score, 1.05, places=2)

        # Check both reranked and filtered
        for result in results:
            self.assertTrue(result.metadata["reranked"])
            self.assertTrue(result.metadata["filtered"])

    def test_run_empty_results(self):
        """Test pipeline with no results."""
        empty_retriever = MockRetriever([])
        pipeline = RetrievalPipeline(retriever=empty_retriever)

        results = pipeline.run("test query")
        self.assertEqual(len(results), 0)


class TestRetrievalPipelineFactory(unittest.TestCase):
    """Test the RetrievalPipelineFactory."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            },
            "vector_db": {
                "url": "localhost:6333",
                "collection_name": "test_collection"
            }
        }

    @patch('components.retrieval_pipeline.QdrantVectorDB')
    @patch('embedding.factory.get_embedder')
    def test_create_dense_pipeline(self, mock_get_embedder, mock_qdrant):
        """Test creation of dense pipeline."""
        # Mock dependencies
        mock_embedder = Mock()
        mock_get_embedder.return_value = mock_embedder

        mock_db = Mock()
        mock_qdrant.return_value = mock_db

        pipeline = RetrievalPipelineFactory.create_dense_pipeline(self.config)

        self.assertIsInstance(pipeline, RetrievalPipeline)
        self.assertIsNotNone(pipeline.retriever)

        # Check that embedder and DB were created correctly
        mock_get_embedder.assert_called_once()
        mock_qdrant.assert_called_once()

    @patch('components.retrieval_pipeline.QdrantVectorDB')
    @patch('embedding.factory.get_embedder')
    @patch('embedding.sparse_embedder.SparseEmbedder')
    def test_create_hybrid_pipeline(self, mock_sparse, mock_get_embedder, mock_qdrant):
        """Test creation of hybrid pipeline."""
        # Mock dependencies
        mock_dense_embedder = Mock()
        mock_get_embedder.return_value = mock_dense_embedder

        mock_sparse_embedder = Mock()
        mock_sparse.return_value = mock_sparse_embedder

        mock_db = Mock()
        mock_qdrant.return_value = mock_db

        pipeline = RetrievalPipelineFactory.create_hybrid_pipeline(self.config)

        self.assertIsInstance(pipeline, RetrievalPipeline)
        self.assertIsNotNone(pipeline.retriever)

        # Check that both embedders and DB were created
        mock_get_embedder.assert_called_once()
        mock_sparse.assert_called_once()
        mock_qdrant.assert_called_once()


if __name__ == "__main__":
    unittest.main()
