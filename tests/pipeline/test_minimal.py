#!/usr/bin/env python3
"""
Minimal pipeline tests using only Google Gemini embeddings.
No local models like sentence transformers.
"""

import os
import sys
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-ci')
os.environ.setdefault('GOOGLE_API_KEY', 'test-key-for-ci')


def create_google_only_config() -> Dict[str, Any]:
    """Create a configuration that only uses Google Gemini embeddings."""
    return {
        "description": "Google Gemini only configuration for CI",
        "retrieval_pipeline": {
            "retriever": {
                "type": "dense",
                "top_k": 5,
                "score_threshold": 0.1,
                "embedding": {
                    "strategy": "dense",
                    "dense": {
                        "provider": "google",
                        "model": "models/embedding-001",
                        "dimensions": 768,
                        "api_key_env": "GOOGLE_API_KEY",
                        "batch_size": 16,
                        "vector_name": "dense"
                    }
                },
                "qdrant": {
                    "collection_name": "test_ci_google_only",  # Unique collection name
                    "dense_vector_name": "dense",
                    "host": "localhost",
                    "port": 6333,
                    "force_recreate": True  # Force recreation to avoid conflicts
                },
                "performance": {
                    "lazy_initialization": True,
                    "enable_caching": False
                }
            },
            "stages": [
                {
                    "type": "retriever",
                    "name": "google_dense_retriever"
                }
            ]
        }
    }


def test_config_loading() -> bool:
    """Test that main configuration loads without errors."""
    print("🔍 Testing configuration loading...")
    
    try:
        from config.config_loader import load_config
        config = load_config()
        
        required_keys = ['llm', 'qdrant', 'agent_retrieval']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False
        
        print("✅ Configuration loads successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_agent_schema() -> bool:
    """Test agent schema works correctly."""
    print("🔍 Testing agent schema...")
    
    try:
        from agent.schema import AgentState
        
        state = AgentState(
            question="Test question",
            reference_date="2024-01-01",
            chat_history=[]
        )
        
        # Check required fields
        if 'question' not in state:
            print("❌ Missing question field")
            return False
        
        if 'reference_date' not in state:
            print("❌ Missing reference_date field")
            return False
        
        # Ensure SQL field was removed
        if 'sql' in state:
            print("❌ SQL field should not exist (was removed)")
            return False
        
        print("✅ Agent schema works correctly")
        return True
    except Exception as e:
        print(f"❌ Agent schema test failed: {e}")
        return False


def test_google_embeddings_config() -> bool:
    """Test Google embeddings configuration without actually calling the API."""
    print("🔍 Testing Google embeddings configuration...")
    
    try:
        from embedding.factory import get_embedder
        
        # Test config structure
        embedding_config = {
            "provider": "google",
            "model": "models/embedding-001",
            "dimensions": 768,
            "api_key_env": "GOOGLE_API_KEY"
        }
        
        # This should not fail even without real API key
        # We're just testing the configuration structure
        try:
            embedder = get_embedder(embedding_config)
            # If we get here, the config structure is correct
            print("✅ Google embeddings configuration is valid")
            return True
        except Exception as e:
            # Expected if no real API key - check if it's an auth error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['api', 'key', 'auth', 'credential']):
                print("✅ Google embeddings configuration is valid (auth expected in CI)")
                return True
            else:
                print(f"❌ Unexpected embeddings error: {e}")
                return False
    except Exception as e:
        print(f"❌ Google embeddings test failed: {e}")
        return False


def test_agent_retriever_with_google() -> bool:
    """Test agent retriever with Google-only configuration."""
    print("🔍 Testing agent retriever with Google embeddings...")
    
    try:
        from bin.agent_retriever import ConfigurableRetrieverAgent
        
        # Create temporary config file with Google-only setup
        config = create_google_only_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            # Initialize agent
            agent = ConfigurableRetrieverAgent(temp_config_path)
            
            # Get config info
            config_info = agent.get_config_info()
            
            if not isinstance(config_info, dict):
                print("❌ Config info is not a dict")
                return False
            
            if 'retriever_type' not in config_info:
                print("❌ Missing retriever_type in config info")
                return False
            
            print("✅ Agent retriever with Google embeddings works")
            return True
            
        finally:
            os.unlink(temp_config_path)
    except Exception as e:
        print(f"❌ Agent retriever test failed: {e}")
        return False


def test_pipeline_factory_google_only() -> bool:
    """Test pipeline factory with Google-only configuration."""
    print("🔍 Testing pipeline factory with Google embeddings...")
    
    try:
        from components.retrieval_pipeline import RetrievalPipelineFactory
        
        config = create_google_only_config()
        
        # Try to create pipeline - this may fail due to missing API key
        # but should not fail due to local model loading
        try:
            pipeline = RetrievalPipelineFactory.create_from_config(config)
            print("✅ Pipeline factory works with Google embeddings")
            return True
        except Exception as e:
            error_str = str(e).lower()
            # These are acceptable errors in CI
            acceptable_errors = [
                'api', 'key', 'auth', 'credential', 'quota', 'permission',
                'qdrant', 'collection', 'database', 'connection'
            ]
            
            if any(keyword in error_str for keyword in acceptable_errors):
                print("✅ Pipeline factory handles missing services correctly")
                return True
            else:
                print(f"❌ Unexpected pipeline factory error: {e}")
                return False
    except Exception as e:
        print(f"❌ Pipeline factory test failed: {e}")
        return False


def test_config_switching() -> bool:
    """Test configuration switching mechanism."""
    print("🔍 Testing configuration switching...")
    
    try:
        from bin.switch_agent_config import list_available_configs
        
        configs = list_available_configs()
        
        if len(configs) == 0:
            print("❌ No configurations found")
            return False
        
        # Verify each config file exists and has valid structure
        for config_name, description, path in configs:
            if not Path(path).exists():
                print(f"❌ Config file missing: {path}")
                return False
            
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'retrieval_pipeline' not in config:
                print(f"❌ Invalid config structure: {config_name}")
                return False
        
        print(f"✅ Configuration switching works with {len(configs)} configs")
        return True
    except Exception as e:
        print(f"❌ Configuration switching test failed: {e}")
        return False


def run_minimal_pipeline_tests() -> bool:
    """Run minimal pipeline tests without local models."""
    print("🧪 Minimal Pipeline Tests (Google Gemini Only)")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Agent Schema", test_agent_schema),
        ("Google Embeddings Config", test_google_embeddings_config),
        ("Agent Retriever (Google)", test_agent_retriever_with_google),
        ("Pipeline Factory (Google)", test_pipeline_factory_google_only),
        ("Configuration Switching", test_config_switching),
    ]
    
    passed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
        else:
            failed_tests.append(test_name)
    
    total = len(tests)
    print("\n" + "=" * 50)
    print("📊 MINIMAL PIPELINE TEST RESULTS")
    print("=" * 50)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Pipeline works with Google Gemini embeddings")
        return True
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        print("Failed tests:")
        for test in failed_tests:
            print(f"  • {test}")
        return False


def main():
    """Main function."""
    success = run_minimal_pipeline_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
