#!/usr/bin/env python3
"""
Configuration validation tests for pipeline.
Tests YAML structure and required fields without loading models.
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List


def test_yaml_validity() -> bool:
    """Test that all YAML configuration files are valid."""
    print("🔍 Testing YAML file validity...")
    
    config_dirs = [
        'pipelines/configs/retrieval',
        'pipelines/configs/datasets',
        'pipelines/configs/examples',
        'pipelines/configs/legacy'
    ]
    
    total_files = 0
    errors = []
    
    for config_dir in config_dirs:
        config_path = Path(config_dir)
        if not config_path.exists():
            continue
        
        for yaml_file in config_path.glob('*.yml'):
            total_files += 1
            try:
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
            except Exception as e:
                errors.append(f"{yaml_file}: {e}")
    
    if errors:
        print(f"❌ YAML validation failed:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print(f"✅ All {total_files} YAML files are valid")
        return True


def test_retrieval_config_structure() -> bool:
    """Test retrieval configuration structure."""
    print("🔍 Testing retrieval configuration structure...")
    
    retrieval_dir = Path('pipelines/configs/retrieval')
    if not retrieval_dir.exists():
        print("❌ Retrieval configs directory not found")
        return False
    
    required_fields = ['retrieval_pipeline']
    required_retriever_fields = ['type', 'top_k']
    
    config_count = 0
    for config_file in retrieval_dir.glob('*.yml'):
        config_count += 1
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check top-level fields
        for field in required_fields:
            if field not in config:
                print(f"❌ {config_file.name}: Missing {field}")
                return False
        
        # Check retriever fields
        retriever = config.get('retrieval_pipeline', {}).get('retriever', {})
        for field in required_retriever_fields:
            if field not in retriever:
                print(f"❌ {config_file.name}: Missing retriever.{field}")
                return False
        
        # Check that retriever type is valid
        valid_types = ['dense', 'sparse', 'hybrid']
        if retriever.get('type') not in valid_types:
            print(f"❌ {config_file.name}: Invalid retriever type")
            return False
    
    if config_count == 0:
        print("❌ No retrieval configs found")
        return False
    
    print(f"✅ All {config_count} retrieval configs have valid structure")
    return True


def test_google_embeddings_in_configs() -> bool:
    """Test that Google embeddings are properly configured."""
    print("🔍 Testing Google embeddings configuration...")
    
    retrieval_dir = Path('pipelines/configs/retrieval')
    if not retrieval_dir.exists():
        print("❌ Retrieval configs directory not found")
        return False
    
    google_configs = 0
    for config_file in retrieval_dir.glob('*.yml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Look for Google embeddings configuration
        retriever = config.get('retrieval_pipeline', {}).get('retriever', {})
        embedding = retriever.get('embedding', {})
        
        if embedding:
            dense_config = embedding.get('dense', {})
            if dense_config.get('provider') == 'google':
                google_configs += 1
                
                # Validate Google-specific fields
                required_google_fields = ['model', 'dimensions', 'api_key_env']
                for field in required_google_fields:
                    if field not in dense_config:
                        print(f"❌ {config_file.name}: Missing Google embedding field: {field}")
                        return False
                
                # Check model format
                model = dense_config.get('model', '')
                if not model.startswith('models/'):
                    print(f"❌ {config_file.name}: Invalid Google model format: {model}")
                    return False
    
    print(f"✅ Found {google_configs} configs with valid Google embeddings")
    return True


def test_main_config_structure() -> bool:
    """Test main configuration file structure."""
    print("🔍 Testing main configuration structure...")
    
    main_config_path = Path('config.yml')
    if not main_config_path.exists():
        print("❌ Main config file not found")
        return False
    
    try:
        with open(main_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load main config: {e}")
        return False
    
    required_sections = ['llm', 'qdrant', 'agent_retrieval']
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing main config section: {section}")
            return False
    
    # Check agent_retrieval section
    agent_retrieval = config.get('agent_retrieval', {})
    if 'config_path' not in agent_retrieval:
        print("❌ Missing agent_retrieval.config_path")
        return False
    
    # Check that the referenced config file exists
    config_path = agent_retrieval.get('config_path')
    if not Path(config_path).exists():
        print(f"❌ Referenced config file not found: {config_path}")
        return False
    
    print("✅ Main configuration structure is valid")
    return True


def run_config_validation_tests() -> bool:
    """Run all configuration validation tests."""
    print("🔧 Configuration Validation Tests")
    print("=" * 40)
    
    tests = [
        ("YAML Validity", test_yaml_validity),
        ("Retrieval Config Structure", test_retrieval_config_structure),
        ("Google Embeddings Config", test_google_embeddings_in_configs),
        ("Main Config Structure", test_main_config_structure),
    ]
    
    passed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 25)
        
        if test_func():
            passed += 1
        else:
            failed_tests.append(test_name)
    
    total = len(tests)
    print("\n" + "=" * 40)
    print("📊 CONFIGURATION VALIDATION RESULTS")
    print("=" * 40)
    
    if passed == total:
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        return True
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        print("Failed tests:")
        for test in failed_tests:
            print(f"  • {test}")
        return False


def main():
    """Main function."""
    success = run_config_validation_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
