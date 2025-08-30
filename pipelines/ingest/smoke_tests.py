"""
Smoke tests for validating ingestion pipeline results.
Provides basic sanity checks after data ingestion.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from database.qdrant_controller import QdrantVectorDB
from pipelines.contracts import ChunkMeta

logger = logging.getLogger(__name__)


@dataclass
class SmokeTestResult:
    """Result of a smoke test execution."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SmokeTestRunner:
    """Runs smoke tests after ingestion to validate data quality."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize smoke test runner.

        Args:
            config: Configuration dictionary containing test settings
        """
        self.config = config
        self.sample_size = config.get("sample_size", 5)
        self.min_success_rate = config.get("min_success_rate", 0.8)
        self.vector_db = QdrantVectorDB()
        self.client = self.vector_db.get_client()

    def run_smoke_tests(self, collection_name: str, chunk_metas: List[ChunkMeta]) -> List[SmokeTestResult]:
        """
        Run all smoke tests on the ingested data.

        Args:
            collection_name: Name of the Qdrant collection to test
            chunk_metas: List of ingested chunk metadata

        Returns:
            List of smoke test results
        """
        logger.info(f"Running smoke tests on collection: {collection_name}")

        results = []

        # Test 1: Collection exists and has data
        results.append(self._test_collection_exists(collection_name))

        # Test 2: Data retrieval works
        results.append(self._test_data_retrieval(collection_name))

        # Test 3: Vector search works
        results.append(self._test_vector_search(collection_name))

        # Test 4: Sample data quality
        if chunk_metas:
            results.append(self._test_data_quality(chunk_metas))

        # Summary
        passed_tests = sum(1 for r in results if r.passed)
        total_tests = len(results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        logger.info(
            f"Smoke tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1%})")

        return results

    def _test_collection_exists(self, collection_name: str) -> SmokeTestResult:
        """Test if collection exists and has data."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if collection_name not in collection_names:
                return SmokeTestResult(
                    test_name="collection_exists",
                    passed=False,
                    message=f"Collection {collection_name} not found",
                    details={"available_collections": collection_names}
                )

            # Check if collection has data
            collection_info = self.client.get_collection(collection_name)
            point_count = collection_info.points_count

            if point_count == 0:
                return SmokeTestResult(
                    test_name="collection_exists",
                    passed=False,
                    message=f"Collection {collection_name} exists but has no data",
                    details={"point_count": point_count}
                )

            return SmokeTestResult(
                test_name="collection_exists",
                passed=True,
                message=f"Collection {collection_name} exists with {point_count} points",
                details={"point_count": point_count}
            )

        except Exception as e:
            return SmokeTestResult(
                test_name="collection_exists",
                passed=False,
                message=f"Error checking collection: {str(e)}"
            )

    def _test_data_retrieval(self, collection_name: str) -> SmokeTestResult:
        """Test basic data retrieval."""
        try:
            # Try to scroll through some points
            points = self.client.scroll(
                collection_name=collection_name,
                limit=self.sample_size,
                with_payload=True,
                with_vectors=False
            )[0]  # Get the points from the tuple

            if not points:
                return SmokeTestResult(
                    test_name="data_retrieval",
                    passed=False,
                    message="No points could be retrieved from collection"
                )

            # Check if points have required payload fields
            sample_point = points[0]
            required_fields = ["text", "doc_id", "chunk_id"]
            missing_fields = []

            for field in required_fields:
                if field not in sample_point.payload:
                    missing_fields.append(field)

            if missing_fields:
                return SmokeTestResult(
                    test_name="data_retrieval",
                    passed=False,
                    message=f"Missing required payload fields: {missing_fields}",
                    details={"available_fields": list(
                        sample_point.payload.keys())}
                )

            return SmokeTestResult(
                test_name="data_retrieval",
                passed=True,
                message=f"Successfully retrieved {len(points)} points with valid payload",
                details={"retrieved_count": len(points)}
            )

        except Exception as e:
            return SmokeTestResult(
                test_name="data_retrieval",
                passed=False,
                message=f"Error retrieving data: {str(e)}"
            )

    def _test_vector_search(self, collection_name: str) -> SmokeTestResult:
        """Test vector search functionality."""
        try:
            # Get collection info to determine vector dimensions
            collection_info = self.client.get_collection(collection_name)
            vector_config = collection_info.config.params.vectors

            # Handle both named vectors and single vector config
            if hasattr(vector_config, 'dense'):
                # Named vectors (hybrid setup)
                vector_size = vector_config.dense.size
                vector_name = "dense"
            else:
                # Single vector config
                vector_size = vector_config.size
                vector_name = None

            # Create a dummy query vector
            query_vector = [0.1] * vector_size

            # Perform search
            search_kwargs = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": 3
            }

            if vector_name:
                search_kwargs["using"] = vector_name

            search_results = self.client.search(**search_kwargs)

            if not search_results:
                return SmokeTestResult(
                    test_name="vector_search",
                    passed=False,
                    message="Vector search returned no results"
                )

            return SmokeTestResult(
                test_name="vector_search",
                passed=True,
                message=f"Vector search successful, found {len(search_results)} results",
                details={
                    "result_count": len(search_results),
                    "vector_size": vector_size,
                    "top_score": search_results[0].score if search_results else None
                }
            )

        except Exception as e:
            return SmokeTestResult(
                test_name="vector_search",
                passed=False,
                message=f"Error in vector search: {str(e)}"
            )

    def _test_data_quality(self, chunk_metas: List[ChunkMeta]) -> SmokeTestResult:
        """Test data quality of ingested chunks."""
        try:
            if not chunk_metas:
                return SmokeTestResult(
                    test_name="data_quality",
                    passed=False,
                    message="No chunk metadata provided for quality testing"
                )

            # Sample some chunks for testing
            sample_size = min(self.sample_size, len(chunk_metas))
            sample_chunks = chunk_metas[:sample_size]

            issues = []

            for chunk in sample_chunks:
                # Check text length
                if not chunk.text or len(chunk.text.strip()) < 10:
                    issues.append(
                        f"Chunk {chunk.chunk_id} has insufficient text")

                # Check required fields
                if not chunk.doc_id:
                    issues.append(f"Chunk {chunk.chunk_id} missing doc_id")

                if not chunk.source:
                    issues.append(f"Chunk {chunk.chunk_id} missing source")

            if issues:
                return SmokeTestResult(
                    test_name="data_quality",
                    passed=False,
                    message=f"Found {len(issues)} data quality issues",
                    details={"issues": issues[:10]}  # Limit to first 10 issues
                )

            return SmokeTestResult(
                test_name="data_quality",
                passed=True,
                message=f"Data quality check passed for {sample_size} chunks",
                details={"checked_chunks": sample_size}
            )

        except Exception as e:
            return SmokeTestResult(
                test_name="data_quality",
                passed=False,
                message=f"Error in data quality check: {str(e)}"
            )
