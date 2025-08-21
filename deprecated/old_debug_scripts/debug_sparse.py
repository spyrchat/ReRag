#!/usr/bin/env python3
"""
Debug script to check sparse embedder output.
"""
import sys
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

from embedding.sparse_embedder import SparseEmbedder
from fastembed import SparseTextEmbedding

# Check available models
print("Available sparse models:")
models = SparseTextEmbedding.list_supported_models()
for model in models[:5]:  # Show first 5
    print(f"  {model}")

# Test the sparse embedder directly
embedder = SparseEmbedder(model_name="Qdrant/bm25")

test_text = "Python is a programming language."
print(f"\nTesting with: {test_text}")

result = embedder.embed_query(test_text)
print(f"Result type: {type(result)}")
print(f"Result: {result}")

if hasattr(result, '__len__'):
    print(f"Length: {len(result)}")

if hasattr(result, 'items'):
    print("Has items method")
    items = list(result.items())[:5]
    print(f"First 5 items: {items}")
