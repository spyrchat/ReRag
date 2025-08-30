#!/usr/bin/env python3
"""
Simple Qdrant connectivity test for CI environments.
No embedding models, just basic database operations.
"""

import requests
import json
import sys
import time
from typing import Dict, Any


def wait_for_qdrant(max_attempts: int = 30) -> bool:
    """Wait for Qdrant to be ready."""
    print(f"⏳ Waiting for Qdrant to be ready...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get('http://localhost:6333/health', timeout=2)
            if response.status_code == 200:
                print(f"✅ Qdrant is ready! (attempt {attempt})")
                return True
        except:
            pass
        
        if attempt < max_attempts:
            print(f"⏳ Attempt {attempt}/{max_attempts}...")
            time.sleep(2)
    
    print("❌ Qdrant failed to start")
    return False


def test_qdrant_health() -> bool:
    """Test Qdrant health endpoint."""
    print("🔍 Testing Qdrant health...")
    
    try:
        response = requests.get('http://localhost:6333/health', timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_qdrant_collections_endpoint() -> bool:
    """Test Qdrant collections endpoint."""
    print("🔍 Testing Qdrant collections endpoint...")
    
    try:
        response = requests.get('http://localhost:6333/collections', timeout=5)
        if response.status_code == 200:
            print("✅ Collections endpoint accessible")
            return True
        else:
            print(f"❌ Collections endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Collections endpoint error: {e}")
        return False


def test_create_delete_collection() -> bool:
    """Test creating and deleting a simple test collection."""
    print("🔍 Testing collection creation/deletion...")
    
    collection_name = "test_ci_minimal"
    collection_config = {
        'vectors': {
            'size': 384,  # Small vector size for testing
            'distance': 'Cosine'
        }
    }
    
    try:
        # Clean up if collection exists
        requests.delete(f'http://localhost:6333/collections/{collection_name}', timeout=5)
        
        # Create collection
        create_response = requests.put(
            f'http://localhost:6333/collections/{collection_name}',
            json=collection_config,
            timeout=10
        )
        
        if create_response.status_code not in [200, 201]:
            print(f"❌ Failed to create collection: {create_response.status_code}")
            return False
        
        # Verify collection exists
        info_response = requests.get(
            f'http://localhost:6333/collections/{collection_name}',
            timeout=5
        )
        
        if info_response.status_code != 200:
            print(f"❌ Failed to get collection info: {info_response.status_code}")
            return False
        
        # Delete collection
        delete_response = requests.delete(
            f'http://localhost:6333/collections/{collection_name}',
            timeout=5
        )
        
        if delete_response.status_code not in [200, 404]:
            print(f"❌ Failed to delete collection: {delete_response.status_code}")
            return False
        
        print("✅ Collection operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Collection operations error: {e}")
        return False


def run_qdrant_tests() -> bool:
    """Run basic Qdrant connectivity tests."""
    print("🗄️ Qdrant Connectivity Tests")
    print("=" * 35)
    
    tests = [
        ("Health Check", test_qdrant_health),
        ("Collections Endpoint", test_qdrant_collections_endpoint),
        ("Collection Operations", test_create_delete_collection),
    ]
    
    passed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        
        if test_func():
            passed += 1
        else:
            failed_tests.append(test_name)
    
    total = len(tests)
    print("\n" + "=" * 35)
    print("📊 QDRANT TEST RESULTS")
    print("=" * 35)
    
    if passed == total:
        print("🎉 ALL QDRANT TESTS PASSED!")
        return True
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        print("Failed tests:")
        for test in failed_tests:
            print(f"  • {test}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qdrant connectivity")
    parser.add_argument("--wait", action="store_true", help="Wait for Qdrant first")
    parser.add_argument("--max-wait", type=int, default=30, help="Max wait attempts")
    
    args = parser.parse_args()
    
    if args.wait:
        if not wait_for_qdrant(args.max_wait):
            return 1
    
    success = run_qdrant_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
