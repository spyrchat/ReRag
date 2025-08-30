#!/usr/bin/env python3
"""
Example demonstrating how to use retriever configuration files.
Shows the recommended approach for configuring and using retrievers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from components.retrieval_pipeline import RetrievalPipelineFactory
from pipelines.configs.retriever_config_loader import RetrieverConfigLoader, load_retriever_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_config_usage():
    """Example: Basic usage of retriever configuration files."""
    
    logger.info("=== Basic Configuration Usage ===")
    
    # Load a specific retriever configuration
    try:
        dense_config = load_retriever_config("dense")
        logger.info(f"Dense retriever config loaded: {dense_config['retriever']['type']}")
        
        hybrid_config = load_retriever_config("hybrid")  
        logger.info(f"Hybrid retriever config loaded: {hybrid_config['retriever']['type']}")
        
    except Exception as e:
        logger.error(f"Error loading configs: {e}")


def example_list_available_configs():
    """Example: List all available configuration files."""
    
    logger.info("=== Available Retriever Configurations ===")
    
    # Method 1: Using the config loader directly
    loader = RetrieverConfigLoader()
    available_configs = loader.get_available_configs()
    logger.info(f"Available configs (via loader): {available_configs}")
    
    # Method 2: Using the factory method
    factory_configs = RetrievalPipelineFactory.list_available_retrievers()
    logger.info(f"Available configs (via factory): {factory_configs}")


def example_create_pipeline_from_config():
    """Example: Create pipeline directly from configuration file."""
    
    logger.info("=== Creating Pipeline from Configuration ===")
    
    try:
        # Create a hybrid pipeline from configuration
        pipeline = RetrievalPipelineFactory.create_from_retriever_config("hybrid")
        logger.info(f"Created pipeline with {len(pipeline.components)} components")
        logger.info(f"Component names: {[comp.component_name for comp in pipeline.components]}")
        
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")


def example_merge_with_global_config():
    """Example: Merge retriever config with global pipeline settings."""
    
    logger.info("=== Merging with Global Configuration ===")
    
    # Global configuration (e.g., from main pipeline config)
    global_config = {
        "qdrant": {
            "collection_name": "production_collection",
            "host": "localhost",
            "port": 6333
        },
        "performance": {
            "enable_caching": True,
            "cache_ttl": 3600
        },
        "logging": {
            "level": "DEBUG"
        }
    }
    
    try:
        # Create pipeline with merged configuration
        pipeline = RetrievalPipelineFactory.create_from_retriever_config(
            "semantic", 
            global_config
        )
        
        logger.info("Created semantic pipeline with merged configuration")
        
    except Exception as e:
        logger.error(f"Failed to create merged pipeline: {e}")


def example_config_validation():
    """Example: Configuration validation and error handling."""
    
    logger.info("=== Configuration Validation ===")
    
    loader = RetrieverConfigLoader()
    
    # Test loading all configs to check for issues
    all_configs = loader.load_all_configs()
    
    for retriever_type, config in all_configs.items():
        try:
            # Validate that each config can create a retriever
            retriever = RetrievalPipelineFactory._create_retriever(
                config['retriever'], config
            )
            logger.info(f"✓ {retriever_type} config is valid: {retriever.component_name}")
            
        except Exception as e:
            logger.warning(f"✗ {retriever_type} config has issues: {e}")


def example_custom_config_directory():
    """Example: Using a custom configuration directory."""
    
    logger.info("=== Custom Configuration Directory ===")
    
    # You can use a different config directory
    custom_config_dir = "/path/to/custom/configs"
    
    try:
        # This would load from a custom directory
        # loader = RetrieverConfigLoader(custom_config_dir)
        # configs = loader.get_available_configs()
        
        logger.info("Custom config directory example (would load from custom path)")
        
    except Exception as e:
        logger.info(f"Custom config example: {e}")


def example_runtime_config_modification():
    """Example: Modifying configuration at runtime."""
    
    logger.info("=== Runtime Configuration Modification ===")
    
    try:
        # Load base configuration
        config = load_retriever_config("dense")
        
        # Modify settings at runtime
        config['retriever']['top_k'] = 20  # Increase results
        config['retriever']['score_threshold'] = 0.1  # Add filtering
        config['performance']['batch_size'] = 64  # Optimize performance
        
        # Create retriever with modified config
        retriever = RetrievalPipelineFactory._create_retriever(
            config['retriever'], config
        )
        
        logger.info(f"Created retriever with modified config: top_k={config['retriever']['top_k']}")
        
    except Exception as e:
        logger.error(f"Runtime modification example failed: {e}")


if __name__ == "__main__":
    logger.info("🔧 Retriever Configuration Examples")
    logger.info("=" * 50)
    
    # Run all examples
    example_basic_config_usage()
    print()
    
    example_list_available_configs()
    print()
    
    example_create_pipeline_from_config()
    print()
    
    example_merge_with_global_config()
    print()
    
    example_config_validation()
    print()
    
    example_custom_config_directory()
    print()
    
    example_runtime_config_modification()
    
    logger.info("=" * 50)
    logger.info("✅ Configuration examples completed!")
    
    # Show the recommended file structure
    logger.info("\n📁 Recommended Configuration Structure:")
    logger.info("pipelines/")
    logger.info("├── configs/")
    logger.info("│   ├── retrievers/")
    logger.info("│   │   ├── dense_retriever.yml")
    logger.info("│   │   ├── sparse_retriever.yml")
    logger.info("│   │   ├── hybrid_retriever.yml")
    logger.info("│   │   └── semantic_retriever.yml")
    logger.info("│   ├── retriever_config_loader.py")
    logger.info("│   └── stackoverflow_hybrid.yml  # Existing")
    logger.info("└── ...")
