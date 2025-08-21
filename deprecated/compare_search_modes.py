#!/usr/bin/env python3
"""
Quick test for sparse vs dense vs hybrid search comparison.
"""
import sys
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

import yaml
from pathlib import Path
from semantic_search_demo import SemanticSearcher

def compare_search_modes():
    """Compare different search modes on the same query."""
    
    config_path = Path("pipelines/configs/stackoverflow_hybrid.yml")
    collection_name = "sosum_stackoverflow_hybrid_v1_canary"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    searcher = SemanticSearcher(config, collection_name)
    
    test_query = "numpy array dimensions"
    
    print("\n" + "="*80)
    print("🔍 SEARCH MODE COMPARISON")
    print("="*80)
    print(f"Query: '{test_query}'")
    
    # Test dense search
    print(f"\n🎯 DENSE SEARCH:")
    searcher.search_and_display(test_query, limit=2, search_mode="dense")
    
    # Test sparse search  
    print(f"\n🎯 SPARSE SEARCH:")
    searcher.search_and_display(test_query, limit=2, search_mode="sparse")
    
    # Test hybrid search
    print(f"\n🎯 HYBRID SEARCH:")
    searcher.search_and_display(test_query, limit=2, search_mode="hybrid")

if __name__ == "__main__":
    compare_search_modes()
