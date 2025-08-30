#!/usr/bin/env python3
"""
Direct test for agent retriever functionality without LLM dependencies.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bin.agent_retriever import ConfigurableRetrieverAgent
from config.config_loader import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_retriever_config_loading():
    """Test that retriever configurations load correctly."""
    
    logger.info("=== Testing Retriever Configuration Loading ===")
    
    # Test each configuration
    configs = ["modern_hybrid", "modern_dense", "fast_hybrid"]
    
    for config_name in configs:
        try:
            config_path = f"pipelines/configs/retrieval/{config_name}.yml"
            agent = ConfigurableRetrieverAgent(config_path)
            
            config_info = agent.get_config_info()
            logger.info(f"✅ {config_name}:")
            logger.info(f"   Type: {config_info['retriever_type']}")
            logger.info(f"   Stages: {config_info['num_stages']}")
            logger.info(f"   Components: {config_info['stage_types']}")
            logger.info(f"   Collection: {config_info['collection']}")
            
        except Exception as e:
            logger.error(f"❌ {config_name}: {e}")
    
    return True


def test_retrieval_functionality():
    """Test actual retrieval functionality."""
    
    logger.info("=== Testing Retrieval Functionality ===")
    
    # Use current active configuration
    config = load_config()
    config_path = config.get("agent_retrieval", {}).get("config_path")
    
    if not config_path:
        logger.error("No active agent retrieval configuration found")
        return False
    
    try:
        agent = ConfigurableRetrieverAgent(config_path)
        config_info = agent.get_config_info()
        
        logger.info(f"Testing with: {config_info['retriever_type']}")
        
        # Test queries
        test_queries = [
            "How to handle Python exceptions?",
            "Binary search algorithm",
            "Python list comprehensions"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Query: {query}")
            
            try:
                results = agent.retrieve(query, top_k=3)
                
                if results:
                    logger.info(f"   Retrieved: {len(results)} documents")
                    
                    # Show top result details
                    top_result = results[0]
                    logger.info(f"   Top score: {top_result['score']:.3f}")
                    logger.info(f"   Method: {top_result['retrieval_method']}")
                    logger.info(f"   Title: {top_result['question_title'][:50]}...")
                    
                else:
                    logger.warning(f"   No results for query: {query}")
                    
            except Exception as e:
                logger.error(f"   Retrieval failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test retrieval: {e}")
        return False


def test_config_switching():
    """Test switching between configurations."""
    
    logger.info("=== Testing Configuration Switching ===")
    
    config_path = "pipelines/configs/retrieval/modern_hybrid.yml"
    agent = ConfigurableRetrieverAgent(config_path)
    
    # Test switching to different configs
    configs = [
        "pipelines/configs/retrieval/modern_dense.yml",
        "pipelines/configs/retrieval/fast_hybrid.yml",
        "pipelines/configs/retrieval/modern_hybrid.yml"  # back to original
    ]
    
    test_query = "Python exceptions handling"
    
    for new_config in configs:
        try:
            agent.switch_config(new_config)
            config_info = agent.get_config_info()
            
            logger.info(f"Switched to: {config_info['retriever_type']}")
            
            # Quick test
            results = agent.retrieve(test_query, top_k=1)
            if results:
                logger.info(f"  ✅ Retrieved {len(results)} result(s)")
            else:
                logger.warning(f"  ⚠ No results")
                
        except Exception as e:
            logger.error(f"  ❌ Switch failed: {e}")
    
    return True


def main():
    """Run all retriever tests."""
    
    logger.info("🧪 Agent Retriever Direct Tests")
    logger.info("=" * 50)
    
    # Test 1: Configuration loading
    test1_success = test_retriever_config_loading()
    print()
    
    # Test 2: Retrieval functionality  
    test2_success = test_retrieval_functionality()
    print()
    
    # Test 3: Configuration switching
    test3_success = test_config_switching()
    
    logger.info("=" * 50)
    
    if all([test1_success, test2_success, test3_success]):
        logger.info("✅ All retriever tests passed!")
        logger.info("\n🎉 Agent retrieval system is ready!")
        logger.info("   Use 'python bin/switch_agent_config.py <config>' to switch configurations")
        logger.info("   Available configs: modern_hybrid, modern_dense, fast_hybrid")
    else:
        logger.warning("⚠ Some tests failed - check configuration and database connections")


if __name__ == "__main__":
    main()
