#!/usr/bin/env python3
"""
Retrieval Pipeline CLI - Use any YAML configuration
Usage: python bin/retrieval_pipeline.py --config pipelines/configs/retrieval/basic_dense.yml --query "How to handle exceptions in Python?"
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from components.retrieval_pipeline import RetrievalPipelineFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def run_retrieval(config: dict, query: str, top_k: int = 5) -> list:
    """Run retrieval with the specified configuration."""
    logger.info(f"Creating pipeline from configuration...")
    
    # Create pipeline from config
    pipeline = RetrievalPipelineFactory.create_from_config(config)
    
    logger.info(f"Pipeline components: {[c.component_name for c in pipeline.components]}")
    
    # Run retrieval
    logger.info(f"Running query: '{query}'")
    results = pipeline.run(query, k=top_k)
    
    return results


def display_results(results: list, show_content: bool = False):
    """Display retrieval results in a nice format."""
    print(f"\n🔍 Found {len(results)} results:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        labels = result.document.metadata.get('labels', {})
        
        print(f"\n{i}. Score: {result.score:.4f} | Method: {result.retrieval_method}")
        
        # Show question title if available
        title = labels.get('title', 'N/A')
        if title != 'N/A':
            print(f"   📝 Question: {title}")
        
        # Show tags if available
        tags = labels.get('tags', [])
        if tags:
            print(f"   🏷️  Tags: {', '.join(tags[:5])}")  # Show first 5 tags
        
        # Show enhancement info if available
        if result.metadata.get('enhanced'):
            quality = result.metadata.get('answer_quality', 'unknown')
            print(f"   ✨ Enhanced (Quality: {quality})")
        
        # Show content if requested
        if show_content:
            content = result.document.page_content[:200] + "..." if len(result.document.page_content) > 200 else result.document.page_content
            print(f"   📄 Content: {content}")
        
        print("-" * 80)


def list_available_configs():
    """List all available configuration files."""
    config_dir = Path("pipelines/configs/retrieval")
    
    if not config_dir.exists():
        print("❌ No retrieval configurations found")
        return
    
    print("\n📋 Available configurations:")
    print("=" * 50)
    
    configs = list(config_dir.glob("*.yml"))
    for config_file in sorted(configs):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract pipeline info
            pipeline_info = config.get('retrieval_pipeline', {})
            retriever_type = pipeline_info.get('retriever', {}).get('type', 'unknown')
            stages = pipeline_info.get('stages', [])
            
            print(f"\n📁 {config_file.name}")
            print(f"   Retriever: {retriever_type}")
            print(f"   Stages: {len(stages)} components")
            
            if stages:
                stage_types = [stage.get('type', 'unknown') for stage in stages]
                print(f"   Pipeline: {retriever_type} → {' → '.join(stage_types)}")
            
        except Exception as e:
            print(f"❌ Error reading {config_file.name}: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run retrieval pipeline with specified YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use basic dense retrieval
  python bin/retrieval_pipeline.py --config pipelines/configs/retrieval/basic_dense.yml --query "Python exceptions"
  
  # Use advanced pipeline with reranking
  python bin/retrieval_pipeline.py --config pipelines/configs/retrieval/advanced_reranked.yml --query "binary search algorithm"
  
  # Show full content of results
  python bin/retrieval_pipeline.py --config pipelines/configs/retrieval/experimental.yml --query "metaclasses" --show-content
  
  # List available configurations
  python bin/retrieval_pipeline.py --list-configs
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Search query'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of results to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--show-content',
        action='store_true',
        help='Show document content in results'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configuration files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # List configs and exit
        if args.list_configs:
            list_available_configs()
            return
        
        # Validate required arguments
        if not args.config:
            print("❌ Error: --config is required (or use --list-configs to see available options)")
            parser.print_help()
            return
        
        if not args.query:
            print("❌ Error: --query is required")
            parser.print_help()
            return
        
        print(f"🚀 Running retrieval pipeline")
        print(f"📋 Config: {args.config}")
        print(f"🔍 Query: {args.query}")
        print(f"📊 Top-K: {args.top_k}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Run retrieval
        results = run_retrieval(config, args.query, args.top_k)
        
        # Display results
        display_results(results, show_content=args.show_content)
        
        print(f"\n✅ Retrieval completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
