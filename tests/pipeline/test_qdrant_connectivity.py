"""
Qdrant Database Connectivity Tests

Tests basic Qdrant connectivity and operations without requiring embeddings.
These tests are optional and can be skipped if Qdrant is not available.
"""

import pytest
import requests
import os
from typing import Dict, Any


class TestQdrantConnectivity:
    """Test Qdrant database connectivity (optional)."""
    
    @pytest.fixture
    def qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration."""
        return {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "url": f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
        }
    
    @pytest.mark.integration
    def test_qdrant_health_endpoint(self, qdrant_config):
        """Test Qdrant health endpoint is accessible."""
        try:
            health_url = f"{qdrant_config['url']}/health"
            response = requests.get(health_url, timeout=5)
            
            # Qdrant returns 200 with health status, but sometimes health endpoint is different
            # If 404, try the collections endpoint instead as a health check
            if response.status_code == 404:
                collections_url = f"{qdrant_config['url']}/collections"
                response = requests.get(collections_url, timeout=5)
                assert response.status_code == 200
            else:
                assert response.status_code == 200
            
        except requests.ConnectionError:
            pytest.skip("Qdrant not available - skipping connectivity tests")
        except requests.Timeout:
            pytest.skip("Qdrant connection timeout - skipping connectivity tests")
    
    @pytest.mark.integration  
    def test_qdrant_collections_endpoint(self, qdrant_config):
        """Test Qdrant collections endpoint is accessible."""
        try:
            collections_url = f"{qdrant_config['url']}/collections"
            response = requests.get(collections_url, timeout=5)
            
            # Should return 200 with collections list (might be empty)
            assert response.status_code == 200
            
            data = response.json()
            assert "result" in data
            assert "collections" in data["result"]
            
        except requests.ConnectionError:
            pytest.skip("Qdrant not available - skipping connectivity tests")
        except requests.Timeout:
            pytest.skip("Qdrant connection timeout - skipping connectivity tests")
    
    @pytest.mark.integration
    def test_qdrant_collection_creation_deletion(self, qdrant_config):
        """Test basic collection creation and deletion (no embeddings)."""
        try:
            base_url = qdrant_config['url']
            test_collection = "test_minimal_collection"
            
            # Create collection with minimal config
            create_url = f"{base_url}/collections/{test_collection}"
            create_payload = {
                "vectors": {
                    "size": 768,  # Google embeddings size
                    "distance": "Cosine"
                }
            }
            
            # Clean up first if exists
            requests.delete(create_url, timeout=5)
            
            # Create collection
            response = requests.put(create_url, json=create_payload, timeout=5)
            assert response.status_code in [200, 201]
            
            # Verify collection exists
            list_response = requests.get(f"{base_url}/collections", timeout=5)
            assert list_response.status_code == 200
            
            collections = list_response.json()["result"]["collections"]
            collection_names = [c["name"] for c in collections]
            assert test_collection in collection_names
            
            # Clean up
            delete_response = requests.delete(create_url, timeout=5)
            assert delete_response.status_code == 200
            
        except requests.ConnectionError:
            pytest.skip("Qdrant not available - skipping connectivity tests")
        except requests.Timeout:
            pytest.skip("Qdrant connection timeout - skipping connectivity tests")
    
    @pytest.mark.integration
    def test_qdrant_client_import(self):
        """Test that Qdrant client can be imported."""
        try:
            from qdrant_client import QdrantClient
            
            # Should be able to create client instance (doesn't connect yet)
            client = QdrantClient(host="localhost", port=6333)
            assert client is not None
            
        except ImportError:
            pytest.skip("qdrant-client not installed - skipping client tests")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
