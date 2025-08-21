#!/usr/bin/env python3
"""
Semantic search tool using the actual embeddings to find relevant answers.
This demonstrates the real power of the RAG system.
"""
import sys
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder
from qdrant_client.http.models import SearchRequest, Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Semantic search using actual embeddings."""
    
    def __init__(self, config: Dict[str, Any], collection_name: str):
        self.config = config
        self.collection_name = collection_name
        
        # Initialize embedders based on config
        self.dense_embedder = None
        self.sparse_embedder = None
        
        embedding_strategy = config.get("embedding_strategy", "dense").lower()
        
        if embedding_strategy in ["dense", "hybrid"]:
            dense_config = config.get("embedding", {}).get("dense", {})
            self.dense_embedder = get_embedder(dense_config)
            logger.info(f"Initialized dense embedder for search")
        
        if embedding_strategy in ["sparse", "hybrid"]:
            sparse_config = config.get("embedding", {}).get("sparse", {})
            self.sparse_embedder = get_embedder(sparse_config)
            logger.info(f"Initialized sparse embedder for search")
        
        # Initialize vector database
        self.vector_db = QdrantVectorDB()
        self.client = self.vector_db.get_client()
        
        # Vector names from config
        self.dense_vector_name = config.get("qdrant", {}).get("dense_vector_name", "dense")
        self.sparse_vector_name = config.get("qdrant", {}).get("sparse_vector_name", "sparse")
    
    def search(self, query: str, limit: int = 5, use_dense: bool = True, use_sparse: bool = False) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        
        results = []
        
        try:
            if use_dense and self.dense_embedder:
                logger.info(f"Performing dense vector search for: '{query}'")
                dense_results = self._dense_search(query, limit)
                results.extend(dense_results)
            
            if use_sparse and self.sparse_embedder:
                logger.info(f"Performing sparse vector search for: '{query}'")
                sparse_results = self._sparse_search(query, limit)
                results.extend(sparse_results)
            
            # Remove duplicates and sort by score
            seen_ids = set()
            unique_results = []
            for result in sorted(results, key=lambda x: x["score"], reverse=True):
                if result["id"] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result["id"])
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def _dense_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using dense embeddings."""
        
        # Generate query embedding
        query_vector = self.dense_embedder.embed_query(query)
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=(self.dense_vector_name, query_vector),
            limit=limit,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "search_type": "dense",
                "payload": result.payload
            })
        
        return results
    
    def _sparse_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using sparse embeddings."""
        
        # Generate query embedding
        query_vector = self.sparse_embedder.embed_query(query)
        
        # Convert dict to SparseVector format
        from qdrant_client.http.models import SparseVector
        sparse_vector = SparseVector(
            indices=list(query_vector.keys()),
            values=list(query_vector.values())
        )
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=(self.sparse_vector_name, sparse_vector),
            limit=limit,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "search_type": "sparse",
                "payload": result.payload
            })
        
        return results
    
    def search_and_display(self, query: str, limit: int = 5, search_mode: str = "dense"):
        """Search and display results in a nice format."""
        
        print(f"\n{'='*80}")
        print(f"🔍 SEMANTIC SEARCH: '{query}'")
        print(f"📊 Collection: {self.collection_name}")
        print(f"🎯 Mode: {search_mode}")
        print(f"{'='*80}")
        
        use_dense = search_mode in ["dense", "hybrid"]
        use_sparse = search_mode in ["sparse", "hybrid"]
        
        results = self.search(query, limit, use_dense, use_sparse)
        
        if not results:
            print("❌ No results found.")
            return
        
        print(f"\n✅ Found {len(results)} relevant results:")
        
        for i, result in enumerate(results, 1):
            payload = result["payload"]
            
            print(f"\n{i}. SCORE: {result['score']:.4f} ({result['search_type']})")
            print(f"   📋 ID: {result['id']}")
            print(f"   🆔 External ID: {payload.get('external_id', 'N/A')}")
            print(f"   📝 Title: {payload.get('labels', {}).get('title', 'N/A')}")
            print(f"   🏷️  Tags: {payload.get('labels', {}).get('tags', [])}")
            
            # Show the actual text content
            text = payload.get('text', '')
            if len(text) > 400:
                print(f"   📄 Content: {text[:400]}...")
            else:
                print(f"   📄 Content: {text}")
        
        print(f"\n💡 This demonstrates semantic similarity - finding relevant content even without exact keyword matches!")


def main():
    """Main function to test semantic search."""
    
    # Available collections and their configs
    collections = {
        "dense": {
            "collection": "sosum_stackoverflow_minilm_v1",
            "config": "pipelines/configs/stackoverflow_minilm.yml"
        },
        "hybrid": {
            "collection": "sosum_stackoverflow_hybrid_v1_canary", 
            "config": "pipelines/configs/stackoverflow_hybrid.yml"
        }
    }
    
    print("🔍 SEMANTIC SEARCH DEMO")
    print("=" * 50)
    print("Available collections:")
    for key, info in collections.items():
        print(f"  {key}: {info['collection']}")
    
    # Test queries that should find relevant Stack Overflow content
    test_queries = [
        "How to calculate array size in Python?",
        "Getting dimensions of numpy arrays",  
        "C# datetime and age calculation",
        "Message box customization in .NET",
        "Working with arrays in programming"
    ]
    
    print(f"\nTesting with queries: {test_queries}")
    
    # Test with hybrid collection (most comprehensive)
    if "hybrid" in collections:
        config_path = Path(collections["hybrid"]["config"])
        collection_name = collections["hybrid"]["collection"]
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        searcher = SemanticSearcher(config, collection_name)
        
        for query in test_queries:
            try:
                searcher.search_and_display(query, limit=3, search_mode="dense")
                input("\nPress Enter to continue to next query...")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error with query '{query}': {e}")
                continue
    else:
        print("❌ Hybrid collection not found. Please run ingestion first.")


if __name__ == "__main__":
    main()
