#!/usr/bin/env python3
"""
Test script for the configurable agent retriever node.
Demonstrates how the agent uses different YAML-configured retrieval pipelines.
"""
from config.config_loader import load_config
from agent.nodes.retriever import make_configurable_retriever
from agent.schema import AgentState
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_agent_retriever_node():
    """
    Test the agent retriever node with different configurations.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("Testing Agent Retriever Node")
    print("=" * 50)

    # Load main config to see current retrieval config
    config = load_config()
    current_config = config.get("retrieval", {}).get("config_path", "Unknown")
    print(f"Current retrieval config: {current_config}")

    # Create the retriever function using the current config
    try:
        retriever_node = make_configurable_retriever()
        print("Successfully created configurable retriever node")
    except Exception as e:
        print(f"Failed to create retriever node: {e}")
        return False

    # Test with a sample question
    test_question = "How to handle Python exceptions and error handling best practices?"

    print(f"\nTesting retrieval with question: '{test_question}'")

    # Create agent state
    state = AgentState(
        question=test_question,
        retrieval_top_k=5  # Override default top_k
    )

    try:
        # Run the retriever node
        result_state = retriever_node(state)

        # Check results
        documents = result_state.get("retrieved_documents", [])
        metadata = result_state.get("retrieval_metadata", {})

        print(f"\nRetrieval Results:")
        print(f"   Documents retrieved: {len(documents)}")
        print(
            f"   Retrieval method: {metadata.get('retrieval_method', 'Unknown')}")

        if "pipeline_config" in metadata:
            pipeline_info = metadata["pipeline_config"]
            print(
                f"   Pipeline type: {pipeline_info.get('retriever_type', 'Unknown')}")
            print(
                f"   Pipeline stages: {pipeline_info.get('stage_types', [])}")

        if "scores" in metadata:
            print(
                f"   Score range: {metadata['scores'].get('min_score', 0):.3f} - {metadata['scores'].get('max_score', 0):.3f}")

        # Show a sample document
        if documents:
            sample_doc = documents[0]
            print(f"\nSample Document:")
            print(f"   Content preview: {sample_doc.page_content[:200]}...")
            if hasattr(sample_doc, 'metadata'):
                print(f"   Metadata keys: {list(sample_doc.metadata.keys())}")

        print("\nAgent retriever node test completed successfully!")
        return True

    except Exception as e:
        print(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_configs():
    """
    Display instructions for testing different configurations.
    Shows CLI commands to switch between retrieval pipeline configs.
    """
    print("\nTesting Configuration Switching")
    print("=" * 50)

    # This would typically be done via the CLI tool
    print("To test different configurations, use:")
    print("   python bin/switch_agent_config.py --list")
    print("   python bin/switch_agent_config.py basic_dense")
    print("   python test_agent_retriever_node.py")
    print("   python bin/switch_agent_config.py advanced_reranked")
    print("   python test_agent_retriever_node.py")


if __name__ == "__main__":
    print("Agent Retriever Node Testing Suite")
    print("=" * 60)

    success = test_agent_retriever_node()

    if success:
        test_different_configs()
        print("\nAll tests completed! Your agent retriever is working correctly.")
    else:
        print("\nTests failed. Please check your configuration and setup.")
        sys.exit(1)
