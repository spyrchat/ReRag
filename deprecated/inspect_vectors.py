#!/usr/bin/env python3
"""
Inspect the actual vector structure in Qdrant to understand what gets embedded.
"""
import sys
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from database.qdrant_controller import QdrantVectorDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_vector_structure():
    """Inspect the actual vector structure in collections."""
    
    vector_db = QdrantVectorDB()
    client = vector_db.get_client()
    
    collections = ["sosum_stackoverflow_hybrid_v1_canary", "sosum_stackoverflow_minilm_v1"]
    
    for collection_name in collections:
        if not client.collection_exists(collection_name):
            print(f"❌ Collection {collection_name} doesn't exist")
            continue
            
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING COLLECTION: {collection_name}")
        print(f"{'='*60}")
        
        # Get collection info
        info = client.get_collection(collection_name)
        print(f"📊 Points: {info.points_count}")
        print(f"📊 Status: {info.status}")
        
        # Check vector configuration
        print("\n🎯 VECTOR CONFIGURATION:")
        if info.config.params.vectors:
            for name, config in info.config.params.vectors.items():
                print(f"   Dense Vector '{name}': {config.size}D, distance={config.distance}")
        
        if info.config.params.sparse_vectors:
            for name, config in info.config.params.sparse_vectors.items():
                print(f"   Sparse Vector '{name}': index={config.index}")
        
        # Sample a point to see actual structure
        sample_points = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_vectors=True
        )[0]
        
        if sample_points:
            point = sample_points[0]
            print(f"\n📄 SAMPLE POINT ID: {point.id}")
            
            if hasattr(point, 'vector') and point.vector:
                print("🎯 ACTUAL VECTORS STORED:")
                for vector_name, vector_data in point.vector.items():
                    if hasattr(vector_data, '__len__'):
                        print(f"   Vector '{vector_name}': {len(vector_data)} dimensions")
                        if isinstance(vector_data, dict):
                            print(f"      Type: Sparse (dict), {len(vector_data)} non-zero entries")
                            if len(vector_data) > 0:
                                sample_entries = dict(list(vector_data.items())[:3])
                                print(f"      Sample: {sample_entries}")
                        elif isinstance(vector_data, list):
                            print(f"      Type: Dense (list), sample: {vector_data[:3]}...")
                        else:
                            print(f"      Type: {type(vector_data)}")
                    else:
                        print(f"   Vector '{vector_name}': {type(vector_data)}")
            
            # Check payload
            if hasattr(point, 'payload') and point.payload:
                print(f"\n📋 PAYLOAD FIELDS:")
                for key in point.payload.keys():
                    if key in ['text', 'dense_embedding', 'sparse_embedding']:
                        continue  # Skip large fields
                    value = point.payload[key]
                    print(f"   {key}: {value}")
                
                # Check what's embedded
                print(f"\n📝 EMBEDDED CONTENT:")
                text = point.payload.get('text', 'N/A')
                print(f"   Text (first 100 chars): {text[:100]}...")
                print(f"   Source: {point.payload.get('source', 'N/A')}")
                print(f"   External ID: {point.payload.get('external_id', 'N/A')}")
                print(f"   Document Type: {point.payload.get('labels', {}).get('doc_type', 'N/A')}")
                print(f"   Post Type: {point.payload.get('labels', {}).get('post_type', 'N/A')}")
        else:
            print("❌ No points found in collection")

if __name__ == "__main__":
    inspect_vector_structure()
