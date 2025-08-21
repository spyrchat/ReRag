#!    print(f"Search returned {len(search_result)} points")
    
    for i, point in enumerate(search_result, 1):
        print(f"\n--- Search Result {i} ---")
        print(f"ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload keys: {list(point.payload.keys()) if point.payload else 'NO PAYLOAD'}")
        if point.payload:
            print(f"External ID: {point.payload.get('external_id', 'MISSING')}")
            
            # Check if metadata is in labels
            labels = point.payload.get('labels', {})
            print(f"Title (from labels): {labels.get('title', 'MISSING')}")
            print(f"Tags (from labels): {labels.get('tags', 'MISSING')}")
            print(f"Post Type (from labels): {labels.get('post_type', 'MISSING')}")
            print(f"Has Question Context: {labels.get('has_question_context', False)}")
            print(f"Text preview: {point.payload.get('text', 'MISSING')[:100]}...")on3
"""
Debug script to check exactly what     print(f"Search returned {len(search_result)} points")
    
    for i, point in enumerate(search_result, 1):
        print(f"\n--- Search Result {i} ---")
        print(f"ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload keys: {list(point.payload.keys()) if point.payload else 'NO PAYLOAD'}")
        if point.payload:
            print(f"External ID: {point.payload.get('external_id', 'MISSING')}")
            
            # Check if metadata is in labels
            labels = point.payload.get('labels', {})
            print(f"Title (from labels): {labels.get('title', 'MISSING')}")
            print(f"Tags (from labels): {labels.get('tags', 'MISSING')}")
            print(f"Post Type (from labels): {labels.get('post_type', 'MISSING')}")
            print(f"Has Question Context: {labels.get('has_question_context', False)}")
            print(f"Text preview: {point.payload.get('text', 'MISSING')[:100]}...")tored and retrieved.
"""

import yaml
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder

def debug_metadata():
    print("=== Debugging Metadata Storage and Retrieval ===")
    
    # Load config
    config_path = "pipelines/configs/stackoverflow_minilm.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    vector_db = QdrantVectorDB()
    client = vector_db.get_client()
    collection_name = "sosum_stackoverflow_minilm_v1"
    
    # First, get a sample point by scrolling
    print("\n1. Getting sample points via scroll...")
    scroll_result = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    
    points = scroll_result[0]
    
    for i, point in enumerate(points, 1):
        print(f"\n--- Point {i} (via scroll) ---")
        print(f"ID: {point.id}")
        print(f"Payload keys: {list(point.payload.keys())}")
        print(f"External ID: {point.payload.get('external_id', 'MISSING')}")
        
        # Check if metadata is in labels
        labels = point.payload.get('labels', {})
        print(f"Labels: {labels}")
        print(f"Title (from labels): {labels.get('title', 'MISSING')}")
        print(f"Tags (from labels): {labels.get('tags', 'MISSING')}")
        print(f"Post Type (from labels): {labels.get('post_type', 'MISSING')}")
        print(f"Text preview: {point.payload.get('text', 'MISSING')[:100]}...")
        break  # Just check first one
    
    # Now test search with payload
    print(f"\n2. Testing search with payload...")
    embedder = get_embedder(config["embedding"]["dense"])
    query = "How to count bits in a number?"
    query_embedding = embedder.embed_query(query)
    
    # Use the deprecated search method since query_points needs vector name
    search_result = client.search(
        collection_name=collection_name,
        query_vector=("dense", query_embedding),  # Specify vector name
        limit=2,
        with_payload=True
    )
    
    print(f"Search returned {len(search_result.points)} points")
    
    for i, point in enumerate(search_result.points, 1):
        print(f"\n--- Search Result {i} ---")
        print(f"ID: {point.id}")
        print(f"Score: {point.score}")
        print(f"Payload keys: {list(point.payload.keys()) if point.payload else 'NO PAYLOAD'}")
        if point.payload:
            print(f"External ID: {point.payload.get('external_id', 'MISSING')}")
            print(f"Title: {point.payload.get('title', 'MISSING')}")
            print(f"Tags: {point.payload.get('tags', 'MISSING')}")
            print(f"Post Type: {point.payload.get('post_type', 'MISSING')}")
            print(f"Text preview: {point.payload.get('text', 'MISSING')[:100]}...")

if __name__ == "__main__":
    debug_metadata()
