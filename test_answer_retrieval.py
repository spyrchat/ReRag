#!/usr/bin/env python3
"""
Test semantic search with our new answer-focused RAG system.
"""

import os
from pathlib import Path
import yaml
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder

def test_answer_retrieval():
    print("=== Testing Answer-Focused RAG Retrieval ===")
    
    # Load config
    config_path = "pipelines/configs/stackoverflow_minilm.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    vector_db = QdrantVectorDB()
    client = vector_db.get_client()
    collection_name = "sosum_stackoverflow_minilm_v1"
    
    embedder = get_embedder(config["embedding"]["dense"])
    
    # Test queries that should retrieve helpful answers
    test_queries = [
        "How to count set bits in an integer?",
        "What is extern C in C++?",
        "How to handle exceptions in Python?",
        "What are metaclasses in Python?",
        "How to calculate age from birthdate?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Generate query embedding
        query_embedding = embedder.embed_query(query)
        
        # Search for similar answers
        search_result = client.search(
            collection_name=collection_name,
            query_vector=("dense", query_embedding),
            limit=3,
            with_payload=True
        )
        
        if search_result:
            print(f"✅ Found {len(search_result)} relevant answers:")
            
            for i, hit in enumerate(search_result, 1):
                score = hit.score
                metadata = hit.payload
                
                print(f"\n  {i}. Score: {score:.3f}")
                print(f"     Answer ID: {metadata.get('external_id', 'N/A')}")
                print(f"     Question: {metadata.get('title', 'N/A')}")
                print(f"     Tags: {metadata.get('tags', [])}")
                print(f"     Text preview: {hit.payload.get('text', '')[:200]}...")
        else:
            print("❌ No results found")

if __name__ == "__main__":
    test_answer_retrieval()
