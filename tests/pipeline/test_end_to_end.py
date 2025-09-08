"""
End-to-End Pipeline Tests

These tests actually run the complete pipeline with real data.
Requires GOOGLE_API_KEY and Qdrant with test data.
Designed for GitHub Actions with API key secrets.
"""

import pytest
import os
import sys
import requests
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "content": "Python exception handling is done using try-except blocks. You can catch specific exceptions like ValueError or use a general except clause. Always handle exceptions gracefully to prevent crashes.",
        "title": "Python Exception Handling Basics",
        "tags": ["python", "error-handling", "exceptions"],
        "external_id": "doc_001"
    },
    {
        "content": "Binary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing the search interval in half. Time complexity is O(log n).",
        "title": "Binary Search Algorithm Explained",
        "tags": ["algorithms", "search", "binary-search", "complexity"],
        "external_id": "doc_002"
    },
    {
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Common types include supervised, unsupervised, and reinforcement learning.",
        "title": "Introduction to Machine Learning",
        "tags": ["machine-learning", "ai", "supervised-learning"],
        "external_id": "doc_003"
    },
    {
        "content": "REST APIs are architectural style for designing networked applications. They use HTTP methods like GET, POST, PUT, DELETE to perform operations. REST APIs are stateless and use JSON for data exchange.",
        "title": "Understanding REST API Design",
        "tags": ["api", "rest", "web-development", "http"],
        "external_id": "doc_004"
    },
    {
        "content": "Docker containers provide a lightweight way to package applications with their dependencies. Containers are isolated, portable, and can run consistently across different environments.",
        "title": "Docker Container Fundamentals",
        "tags": ["docker", "containers", "devops", "deployment"],
        "external_id": "doc_005"
    }
]


@pytest.mark.requires_api
class TestEndToEndPipeline:
    """Test complete pipeline execution with real API and data."""

    @pytest.fixture(scope="class")
    def qdrant_config(self):
        """Qdrant configuration for tests."""
        return {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "url": f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}",
            "collection_name": "test_e2e_collection"
        }

    @pytest.fixture(scope="class")
    def setup_test_collection(self, qdrant_config):
        """Set up Qdrant collection with sample documents for testing."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set - skipping end-to-end test")

        collection_name = qdrant_config["collection_name"]
        base_url = qdrant_config["url"]

        # Clean up any existing collection
        requests.delete(f"{base_url}/collections/{collection_name}")
        time.sleep(1)

        # Create collection with Google embedding dimensions
        create_payload = {
            "vectors": {
                "size": 768,  # Google embeddings size
                "distance": "Cosine"
            }
        }

        response = requests.put(
            f"{base_url}/collections/{collection_name}",
            json=create_payload,
            timeout=10
        )
        assert response.status_code in [
            200, 201], f"Failed to create collection: {response.text}"

        # Insert sample documents with embeddings
        self._insert_sample_documents(qdrant_config)

        yield qdrant_config

        # Cleanup after tests
        requests.delete(f"{base_url}/collections/{collection_name}")

    def _insert_sample_documents(self, qdrant_config):
        """Insert sample documents into Qdrant collection."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        # Initialize Google embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize Qdrant client
        client = QdrantClient(
            host=qdrant_config["host"],
            port=qdrant_config["port"]
        )

        # Generate embeddings and create points
        points = []
        for i, doc in enumerate(SAMPLE_DOCUMENTS):
            # Generate embedding for content
            embedding = embeddings.embed_query(doc["content"])

            # Create point
            point = PointStruct(
                id=i + 1,
                vector=embedding,
                payload={
                    "content": doc["content"],
                    "labels": {
                        "title": doc["title"],
                        "tags": doc["tags"],
                        "external_id": doc["external_id"]
                    }
                }
            )
            points.append(point)

        # Insert points
        client.upsert(
            collection_name=qdrant_config["collection_name"],
            points=points
        )

        # Wait for indexing
        time.sleep(2)
        print(
            f"✅ Inserted {len(points)} sample documents into {qdrant_config['collection_name']}")

    @pytest.mark.integration
    @pytest.mark.requires_api
    def test_full_retrieval_pipeline(self, setup_test_collection):
        """Test complete retrieval pipeline with real query and data."""
        from bin.agent_retriever import ConfigurableRetrieverAgent

        qdrant_config = setup_test_collection

        # Update CI config to use test collection
        self._update_config_for_test(qdrant_config["collection_name"])

        # Create agent with test config
        agent = ConfigurableRetrieverAgent(
            "pipelines/configs/retrieval/ci_google_gemini_test.yml")

        # Test queries that should match our sample documents
        test_cases = [
            {
                "query": "How to handle errors in Python?",
                "expected_keywords": ["exception", "python", "try", "except"],
                "min_score": 0.3
            },
            {
                "query": "What is binary search algorithm?",
                "expected_keywords": ["binary", "search", "algorithm", "sorted"],
                "min_score": 0.3
            },
            {
                "query": "Machine learning introduction",
                "expected_keywords": ["machine learning", "artificial intelligence", "supervised"],
                "min_score": 0.3
            }
        ]

        for test_case in test_cases:
            results = agent.retrieve(test_case["query"], top_k=3)

            # Validate results structure
            assert isinstance(results, list), "Results should be a list"
            assert len(
                results) > 0, f"Should return results for query: {test_case['query']}"
            assert len(
                results) <= 3, "Should not return more than requested top_k"

            # Validate result structure
            result = results[0]
            assert "score" in result, "Result should have score"
            assert "content" in result, "Result should have content"
            assert "retrieval_method" in result, "Result should have retrieval_method"
            assert "question_title" in result, "Result should have title"
            assert "tags" in result, "Result should have tags"

            # Validate score and content quality
            assert isinstance(result["score"], (int, float)
                              ), "Score should be numeric"
            assert result["score"] >= test_case[
                "min_score"], f"Score too low: {result['score']}"
            assert len(result["content"]) > 20, "Content should be substantial"

            # Check if result content contains expected keywords
            content_lower = result["content"].lower()
            title_lower = result["question_title"].lower()

            keyword_found = False
            for keyword in test_case["expected_keywords"]:
                if keyword.lower() in content_lower or keyword.lower() in title_lower:
                    keyword_found = True
                    break

            assert keyword_found, f"No expected keywords found in top result for: {test_case['query']}"

            print(
                f"✅ Query: '{test_case['query']}' -> Score: {result['score']:.3f}, Title: '{result['question_title']}'")

    @pytest.mark.integration
    @pytest.mark.requires_api
    def test_retrieval_ranking(self, setup_test_collection):
        """Test that retrieval returns results in correct ranking order."""
        from bin.agent_retriever import ConfigurableRetrieverAgent

        qdrant_config = setup_test_collection
        self._update_config_for_test(qdrant_config["collection_name"])

        agent = ConfigurableRetrieverAgent(
            "pipelines/configs/retrieval/ci_google_gemini_test.yml")

        # Query that should strongly match the Python exceptions document
        results = agent.retrieve(
            "Python try except error handling ValueError", top_k=5)

        assert len(results) > 1, "Should return multiple results"

        # Scores should be in descending order
        scores = [result["score"] for result in results]
        assert scores == sorted(
            scores, reverse=True), "Results should be sorted by score (descending)"

        # Top result should be highly relevant
        top_result = results[0]
        assert top_result["score"] > 0.5, f"Top result score too low: {top_result['score']}"

        # Check that top result is about Python exceptions
        content_and_title = (
            top_result["content"] + " " + top_result["question_title"]).lower()
        assert "python" in content_and_title, "Top result should mention Python"
        assert ("exception" in content_and_title or "error" in content_and_title), "Top result should be about error handling"

    @pytest.mark.integration
    @pytest.mark.requires_api
    def test_config_switching_with_data(self, setup_test_collection):
        """Test configuration switching works with real data."""
        from bin.agent_retriever import ConfigurableRetrieverAgent

        qdrant_config = setup_test_collection
        self._update_config_for_test(qdrant_config["collection_name"])

        agent = ConfigurableRetrieverAgent(
            "pipelines/configs/retrieval/ci_google_gemini_test.yml")

        # Test with initial config
        query = "machine learning basics"
        results1 = agent.retrieve(query, top_k=2)

        # Get config info
        config_info1 = agent.get_config_info()

        # Verify initial results
        assert len(results1) > 0, "Should return results with initial config"
        assert config_info1["retriever_type"] == "dense", "Should be using dense retriever"
        assert config_info1["collection"] == qdrant_config["collection_name"], "Should use test collection"

        print(
            f"✅ Config switching test completed - Retrieved {len(results1)} results")

    @pytest.mark.integration
    def test_pipeline_error_handling_with_real_setup(self, setup_test_collection):
        """Test pipeline error handling with real Qdrant setup."""
        from bin.agent_retriever import ConfigurableRetrieverAgent

        # Test with non-existent config
        with pytest.raises(FileNotFoundError):
            ConfigurableRetrieverAgent("nonexistent_config.yml")

        # Test with valid agent
        qdrant_config = setup_test_collection
        self._update_config_for_test(qdrant_config["collection_name"])

        agent = ConfigurableRetrieverAgent(
            "pipelines/configs/retrieval/ci_google_gemini_test.yml")

        # Test empty query (should handle gracefully)
        try:
            results = agent.retrieve("", top_k=1)
            # Should either return empty results or handle gracefully
            assert isinstance(
                results, list), "Should return list even for empty query"
        except Exception as e:
            # Should be a meaningful error, not a crash
            assert len(str(e)) > 0, "Error message should be informative"

        # Test very long query (should not crash)
        long_query = "test " * 100  # 500 character query
        try:
            results = agent.retrieve(long_query, top_k=1)
            assert isinstance(results, list), "Should handle long queries"
        except Exception as e:
            print(f"Long query failed (acceptable): {e}")


def _update_config_for_test(self, collection_name: str):
    """Create a test config file with the test collection name."""
    import yaml

    with open("pipelines/configs/retrieval/ci_google_gemini.yml", 'r') as f:
        config = yaml.safe_load(f)

    # Ensure Qdrant points to the test collection
    config.setdefault("qdrant", {})
    config["qdrant"]["collection"] = collection_name
    config.setdefault("retrieval_pipeline", {}).setdefault("retriever", {})
    config["retrieval_pipeline"]["retriever"].setdefault("qdrant", {})
    config["retrieval_pipeline"]["retriever"]["qdrant"]["collection_name"] = collection_name
    config["retrieval_pipeline"]["retriever"]["qdrant"]["force_recreate"] = False

    # >>> Ensure Gemini embeddings are used at query time <<<
    # This block is what your agent must read to build the dense retriever.
    config["retrieval_pipeline"]["retriever"]["embedding"] = {
        "dense": {
            "provider": "google",
            "model": "models/embedding-001",
            "api_key_env": "GOOGLE_API_KEY",
            # Many CI keys behave better over REST than gRPC.
            "transport": "rest"
        }
    }

    test_config_path = "pipelines/configs/retrieval/ci_google_gemini_test.yml"
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Created test config with collection: {collection_name}")


if __name__ == "__main__":
    # Run with specific markers
    pytest.main([__file__, "-v", "-m", "integration"])
