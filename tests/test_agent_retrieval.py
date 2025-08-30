#!/usr/bin/env python3
"""
Test the updated LangGraph agent with configurable retrieval pipeline.
Demonstrates how to switch between different retrieval configurations.
"""

from agent.schema import AgentState
from agent.graph import graph
import yaml
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_with_config(config_name: str, query: str):
    """Test agent with a specific retrieval configuration."""
    print(f"\n{'='*60}")
    print(f"🤖 Testing Agent with {config_name} Configuration")
    print('='*60)

    try:
        # Update the retrieval config in the main config
        main_config_path = Path("config.yml")
        with open(main_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update retrieval config path
        if 'agent_retrieval' not in config:
            config['agent_retrieval'] = {}
        config["agent_retrieval"][
            "config_path"] = f"pipelines/configs/retrieval/{config_name}.yml"
        config["agent_retrieval"]["active_config"] = config_name

        # Write updated config (temporarily)
        backup_config = config.copy()
        with open(main_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create initial state
        initial_state = AgentState(
            question=query,
            reference_date="2024-01-01",
            next_node="retriever",  # Go directly to retriever for this test
            chat_history=[]
        )

        print(f"Query: {query}")
        print(f"Config: {config_name}")

        # Run the agent
        logger.info(f"Running agent with query: {query}")
        result = graph.invoke(initial_state)

        # Display results
        print(f"\n📊 Results:")
        print(f"Context length: {len(result.get('context', ''))}")

        if "retrieval_metadata" in result:
            metadata = result["retrieval_metadata"]
            print(f"Retrieved documents: {metadata.get('num_results', 0)}")
            print(
                f"Retrieval method: {metadata.get('retrieval_method', 'unknown')}")
            print(f"Top score: {metadata.get('top_result_score', 0):.3f}")

            pipeline_info = metadata.get('pipeline_config', {})
            print(
                f"Pipeline: {pipeline_info.get('retriever_type', 'unknown')} with {pipeline_info.get('num_stages', 0)} stages")
            print(
                f"Components: {', '.join(pipeline_info.get('stage_types', []))}")

        if "retrieved_documents" in result and result["retrieved_documents"]:
            doc = result["retrieved_documents"][0]
            print(f"\nTop result preview:")
            print(f"  Score: {doc.metadata.get('score', 0):.3f}")
            print(
                f"  Method: {doc.metadata.get('retrieval_method', 'unknown')}")
            print(
                f"  Question: {doc.metadata.get('question_title', 'N/A')[:50]}...")
            print(f"  Enhanced: {doc.metadata.get('enhanced', False)}")

        if "error" in result:
            print(f"❌ Error: {result['error']}")

        # Restore original config
        with open(main_config_path, 'w') as f:
            yaml.dump(backup_config, f, default_flow_style=False)

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_agent_configurations():
    """Demonstrate agent with different retrieval configurations."""
    print("🚀 LangGraph Agent with Configurable Retrieval Pipeline")
    print("Testing different retrieval configurations...")

    # Test queries
    queries = [
        "How to handle Python exceptions properly?",
        "What is binary search algorithm?",
        "Explain Python metaclasses"
    ]

    # Test different configurations
    configs = [
        "modern_hybrid",        # Modern hybrid with RRF and reranking
        "modern_dense",         # Dense retrieval with neural reranking
        "fast_hybrid"           # Fast hybrid for speed
    ]

    for query in queries[:1]:  # Test with first query only for demo
        for config in configs:
            result = test_agent_with_config(config, query)

            if result and not result.get("error"):
                print("✅ Success")
            else:
                print("❌ Failed")

    # Show how to change configuration easily
    print(f"\n{'='*60}")
    print("💡 How to Change Retrieval Configuration")
    print('='*60)
    print("""
1. Use the switch utility:
   python bin/switch_agent_config.py modern_dense

2. Or edit config.yml:
   agent_retrieval:
     config_path: "pipelines/configs/retrieval/modern_dense.yml"
     active_config: "modern_dense"

3. Available configurations:
   - modern_hybrid.yml        # Advanced hybrid with RRF and reranking
   - modern_dense.yml         # Dense retrieval with neural reranking  
   - fast_hybrid.yml          # Fast hybrid optimized for speed

4. Agent automatically uses the configured pipeline!
    """)


if __name__ == "__main__":
    demo_agent_configurations()
