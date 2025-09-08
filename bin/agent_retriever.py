#!/usr/bin/env python3
"""
Agent wrapper for retrieval pipeline with configurable YAML.
Simple interface for agents to use any retrieval configuration.
"""

from components.retrieval_pipeline import RetrievalPipelineFactory, RetrievalResult
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


logger = logging.getLogger(__name__)


class ConfigurableRetrieverAgent:
    """
    Agent that can use any YAML configuration for retrieval.
    Provides a simple interface for agents to retrieve documents using 
    configurable pipelines without needing to know implementation details.
    """

    def __init__(self, config_path: str, cache_pipeline: bool = True):
        """
        Initialize agent with a specific configuration.

        Args:
            config_path (str): Path to YAML configuration file
            cache_pipeline (bool): Whether to cache the pipeline for reuse
        """
        self.config_path = config_path
        self.cache_pipeline = cache_pipeline
        self._pipeline = None
        self._config = None

        # Load configuration
        self._load_config()

        logger.info(
            f"ConfigurableRetrieverAgent initialized with config: {config_path}")

    def _load_config(self):
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_file = Path(self.config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}")

        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)

        logger.info(f"Loaded configuration: {self.config_path}")

    def _get_pipeline(self):
        """
        Get or create the retrieval pipeline.

        Returns:
            RetrievalPipeline: Configured retrieval pipeline
        """
        if self._pipeline is None or not self.cache_pipeline:
            logger.info("Creating retrieval pipeline from configuration...")
            self._pipeline = RetrievalPipelineFactory.create_from_config(
                self._config)

            components = [c.component_name for c in self._pipeline.components]
            logger.info(f"Pipeline components: {components}")

        return self._pipeline

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.

        Args:
            query (str): Search query
            top_k (int): Number of results to return

        Returns:
            List[Dict[str, Any]]: List of dictionaries with document information
                containing rank, score, content, metadata, etc.
        """
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")

        # Get pipeline and run retrieval
        pipeline = self._get_pipeline()
        results = pipeline.run(query, k=top_k)

        # Convert to simple dictionary format for agents
        documents = []
        for i, result in enumerate(results):
            labels = result.document.metadata.get('labels', {})

            doc_info = {
                'rank': i + 1,
                'score': result.score,
                'content': result.document.page_content,
                'retrieval_method': result.retrieval_method,
                'question_title': labels.get('title', ''),
                'tags': labels.get('tags', []),
                'external_id': labels.get('external_id', ''),
                'enhanced': result.metadata.get('enhanced', False),
                'answer_quality': result.metadata.get('answer_quality', ''),
                'metadata': result.metadata
            }
            documents.append(doc_info)

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        pipeline_config = self._config.get('retrieval_pipeline', {})
        retriever_config = pipeline_config.get('retriever', {})
        stages = pipeline_config.get('stages', [])

        return {
            'config_path': self.config_path,
            'retriever_type': retriever_config.get('type', 'unknown'),
            'retriever_top_k': retriever_config.get('top_k', 5),
            'num_stages': len(stages),
            'stage_types': [stage.get('type', 'unknown') for stage in stages],
            'embedding_strategy': self._config.get('embedding_strategy', 'unknown'),
            'collection': self._config.get('qdrant', {}).get('collection', 'unknown')
        }

    def switch_config(self, new_config_path: str):
        """
        Switch to a different configuration.

        Args:
            new_config_path: Path to new YAML configuration
        """
        logger.info(
            f"Switching configuration from {self.config_path} to {new_config_path}")

        self.config_path = new_config_path
        self._config = None
        self._pipeline = None  # Force recreation

        self._load_config()
        logger.info("Configuration switched successfully")


def get_agent_with_config(config_name: str) -> ConfigurableRetrieverAgent:
    """
    Convenience function to get an agent with a named configuration.

    Args:
        config_name: Name of config file (e.g., 'basic_dense' for basic_dense.yml)

    Returns:
        ConfigurableRetrieverAgent instance
    """
    config_path = f"pipelines/configs/retrieval/{config_name}.yml"
    return ConfigurableRetrieverAgent(config_path)


def demo_agent_usage():
    """Demonstrate how to use the configurable agent."""
    print("🤖 Configurable Retriever Agent Demo")
    print("=" * 50)

    # Test queries
    queries = [
        "How to handle Python exceptions?",
        "Binary search algorithm implementation",
        "What are Python metaclasses?"
    ]

    # Test different configurations
    configs = [
        "basic_dense",
        "advanced_reranked",
        "experimental"
    ]

    for config_name in configs:
        print(f"\n📋 Testing configuration: {config_name}")
        print("-" * 40)

        try:
            # Create agent with specific config
            agent = get_agent_with_config(config_name)

            # Show config info
            config_info = agent.get_config_info()
            print(f"Retriever: {config_info['retriever_type']}")
            print(
                f"Stages: {config_info['num_stages']} ({', '.join(config_info['stage_types'])})")

            # Test a query
            query = queries[0]
            results = agent.retrieve(query, top_k=2)

            print(f"\nQuery: {query}")
            print(f"Results: {len(results)}")

            for doc in results[:1]:  # Show top result
                print(
                    f"  Score: {doc['score']:.3f} | Method: {doc['retrieval_method']}")
                print(f"  Question: {doc['question_title'][:50]}...")

        except Exception as e:
            print(f"❌ Error with {config_name}: {e}")


if __name__ == "__main__":
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO)

    # Run demonstration
    demo_agent_usage()

    print("\n" + "=" * 50)
    print("💡 Usage Examples:")
    print("=" * 50)
    print("""
# Simple usage
agent = get_agent_with_config('basic_dense')
results = agent.retrieve('Python exceptions', top_k=5)

# Switch configurations dynamically
agent.switch_config('pipelines/configs/retrieval/advanced_reranked.yml')
results = agent.retrieve('same query', top_k=5)

# Get config information
info = agent.get_config_info()
print(f"Using {info['retriever_type']} with {info['num_stages']} stages")
    """)
