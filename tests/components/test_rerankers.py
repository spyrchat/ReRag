#!/usr/bin/env python3
"""
Unit tests for reranker components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from components.retrieval_pipeline import RetrievalResult
from components.rerankers import (
    CrossEncoderReranker,
    SemanticReranker,
    BM25Reranker,
    EnsembleReranker
)


class TestCrossEncoderReranker(unittest.TestCase):
    """Test the CrossEncoderReranker component."""

    def setUp(self):
        """Set up test fixtures."""
        self.reranker = CrossEncoderReranker(model_name="test-model", top_k=3)
        self.query = "test query"

        # Create test documents
        self.documents = [
            Document(page_content="First document", metadata={"id": "1"}),
            Document(page_content="Second document", metadata={"id": "2"}),
            Document(page_content="Third document", metadata={"id": "3"}),
        ]

        # Create test results
        self.results = [
            RetrievalResult(
                document=doc,
                score=0.5 + i * 0.1,
                retrieval_method="dense",
                metadata={"original": True}
            )
            for i, doc in enumerate(self.documents)
        ]

    def test_component_name(self):
        """Test component name generation."""
        expected = "cross_encoder_reranker_test-model"
        self.assertEqual(self.reranker.component_name, expected)

    @patch('components.rerankers.CrossEncoder')
    def test_rerank_success(self, mock_cross_encoder):
        """Test successful reranking."""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        mock_cross_encoder.return_value = mock_model

        reranked = self.reranker.rerank(self.query, self.results)

        # Check results are reranked correctly
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0].score, 0.9)
        self.assertEqual(reranked[1].score, 0.8)
        self.assertEqual(reranked[2].score, 0.7)

        # Check metadata is preserved and enhanced
        for result in reranked:
            self.assertIn("original_score", result.metadata)
            self.assertIn("reranker_model", result.metadata)
            self.assertTrue(result.metadata["reranked"])

    def test_rerank_empty_results(self):
        """Test reranking with empty results."""
        reranked = self.reranker.rerank(self.query, [])
        self.assertEqual(len(reranked), 0)

    @patch('components.rerankers.CrossEncoder')
    def test_rerank_error_fallback(self, mock_cross_encoder):
        """Test fallback behavior on error."""
        mock_cross_encoder.side_effect = Exception("Model error")

        reranked = self.reranker.rerank(self.query, self.results)

        # Should return original results (up to top_k)
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0].score, 0.5)


class TestSemanticReranker(unittest.TestCase):
    """Test the SemanticReranker component."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = Mock()
        self.embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        self.embedder.embed_documents.return_value = [
            [1.0, 0.0, 0.0],  # Perfect match
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.5, 0.5, 0.0],  # Partial match
        ]

        self.reranker = SemanticReranker(embedder=self.embedder, top_k=3)

        # Create test results
        self.documents = [
            Document(page_content="First document", metadata={"id": "1"}),
            Document(page_content="Second document", metadata={"id": "2"}),
            Document(page_content="Third document", metadata={"id": "3"}),
        ]

        self.results = [
            RetrievalResult(
                document=doc,
                score=0.5,
                retrieval_method="dense",
                metadata={}
            )
            for doc in self.documents
        ]

    def test_component_name(self):
        """Test component name."""
        self.assertEqual(self.reranker.component_name, "semantic_reranker")

    @patch('sklearn.metrics.pairwise.cosine_similarity')
    def test_rerank_success(self, mock_cosine):
        """Test successful semantic reranking."""
        mock_cosine.return_value = [[1.0, 0.0, 0.7]]  # Similarity scores

        reranked = self.reranker.rerank("test query", self.results)

        # Should be sorted by similarity
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0].score, 1.0)  # Highest similarity first
        self.assertEqual(reranked[1].score, 0.7)
        self.assertEqual(reranked[2].score, 0.0)

    def test_rerank_no_embedder(self):
        """Test reranking without embedder."""
        reranker = SemanticReranker(embedder=None)
        reranked = reranker.rerank("test query", self.results)

        # Should return original results
        self.assertEqual(len(reranked), len(self.results))


class TestBM25Reranker(unittest.TestCase):
    """Test the BM25Reranker component."""

    def setUp(self):
        """Set up test fixtures."""
        self.reranker = BM25Reranker(top_k=3)

        # Create test results
        self.documents = [
            Document(page_content="Python programming language",
                     metadata={"id": "1"}),
            Document(page_content="Java programming tutorial",
                     metadata={"id": "2"}),
            Document(page_content="Machine learning with Python",
                     metadata={"id": "3"}),
        ]

        self.results = [
            RetrievalResult(
                document=doc,
                score=0.5,
                retrieval_method="dense",
                metadata={}
            )
            for doc in self.documents
        ]

    def test_component_name(self):
        """Test component name."""
        self.assertEqual(self.reranker.component_name, "bm25_reranker")

    @patch('rank_bm25.BM25Okapi')
    @patch('nltk.tokenize.word_tokenize')
    def test_rerank_success(self, mock_tokenize, mock_bm25):
        """Test successful BM25 reranking."""
        # Mock tokenization
        mock_tokenize.side_effect = lambda x: x.lower().split()

        # Mock BM25
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_scores.return_value = [0.9, 0.3, 0.6]
        mock_bm25.return_value = mock_bm25_instance

        reranked = self.reranker.rerank("Python", self.results)

        # Should be sorted by BM25 scores
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0].score, 0.9)
        self.assertEqual(reranked[1].score, 0.6)
        self.assertEqual(reranked[2].score, 0.3)


class TestEnsembleReranker(unittest.TestCase):
    """Test the EnsembleReranker component."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock rerankers
        self.reranker1 = Mock()
        self.reranker1.component_name = "reranker1"

        self.reranker2 = Mock()
        self.reranker2.component_name = "reranker2"

        self.ensemble = EnsembleReranker(
            rerankers=[self.reranker1, self.reranker2],
            weights=[0.6, 0.4],
            top_k=3
        )

        # Create test documents and results
        self.documents = [
            Document(page_content="First document", metadata={"id": "1"}),
            Document(page_content="Second document", metadata={"id": "2"}),
        ]

        self.results = [
            RetrievalResult(
                document=self.documents[0],
                score=0.5,
                retrieval_method="dense",
                metadata={}
            ),
            RetrievalResult(
                document=self.documents[1],
                score=0.4,
                retrieval_method="dense",
                metadata={}
            ),
        ]

    def test_component_name(self):
        """Test component name generation."""
        expected = "ensemble_reranker_2_models"
        self.assertEqual(self.ensemble.component_name, expected)

    def test_weights_validation(self):
        """Test that weights must match number of rerankers."""
        with self.assertRaises(ValueError):
            EnsembleReranker(
                rerankers=[self.reranker1, self.reranker2],
                weights=[1.0]  # Wrong number of weights
            )

    def test_rerank_success(self):
        """Test successful ensemble reranking."""
        # Mock reranker results
        self.reranker1.rerank.return_value = [
            RetrievalResult(
                document=self.documents[0],
                score=0.9,
                retrieval_method="reranked1",
                metadata={}
            ),
            RetrievalResult(
                document=self.documents[1],
                score=0.1,
                retrieval_method="reranked1",
                metadata={}
            ),
        ]

        self.reranker2.rerank.return_value = [
            RetrievalResult(
                document=self.documents[0],
                score=0.2,
                retrieval_method="reranked2",
                metadata={}
            ),
            RetrievalResult(
                document=self.documents[1],
                score=0.8,
                retrieval_method="reranked2",
                metadata={}
            ),
        ]

        reranked = self.ensemble.rerank("test query", self.results)

        # Check that ensemble combines scores correctly
        # Doc 0: 0.6 * 0.9 + 0.4 * 0.2 = 0.54 + 0.08 = 0.62 (normalized)
        # Doc 1: 0.6 * 0.1 + 0.4 * 0.8 = 0.06 + 0.32 = 0.38 (normalized)
        self.assertEqual(len(reranked), 2)
        self.assertGreater(reranked[0].score, reranked[1].score)

        # Check metadata is preserved
        for result in reranked:
            self.assertIn("ensemble_components", result.metadata)
            self.assertIn("ensemble_weights", result.metadata)
            self.assertTrue(result.metadata["reranked"])


if __name__ == "__main__":
    unittest.main()
