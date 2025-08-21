#!/usr/bin/env python3
"""
Simple demo showing how to use the LangGraph agent with different retrieval configurations.
This demonstrates the complete integration of configurable retrieval pipelines with the agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
from agent.schema import AgentState

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def demo_agent_retrieval():
    """Simple demo of agent retrieval capabilities."""
    print("🤖 LangGraph Agent with Configurable Retrieval Pipeline")
    print("=" * 60)
    
    try:
        # Import here to avoid issues if dependencies aren't available
        from agent.graph import graph
        
        # Create a simple test state
        test_state = AgentState(
            question="How to handle Python exceptions properly?",
            reference_date="2024-01-01",
            next_node="retriever",  # Go directly to retriever
            chat_history=[]
        )
        
        print(f"📝 Query: {test_state['question']}")
        print(f"🔧 Current config: Advanced reranked pipeline")
        print("\n🔍 Running retrieval...")
        
        # Run the agent (just the retriever part)
        result = graph.invoke(test_state)
        
        # Display results
        print(f"\n✅ Retrieval completed!")
        print(f"📄 Context length: {len(result.get('context', ''))}")
        
        if "retrieval_metadata" in result:
            metadata = result["retrieval_metadata"]
            print(f"📊 Retrieved: {metadata.get('num_results', 0)} documents")
            print(f"🎯 Method: {metadata.get('retrieval_method', 'unknown')}")
            print(f"⭐ Top score: {metadata.get('top_result_score', 0):.3f}")
            
            if "pipeline_config" in metadata:
                pipeline = metadata["pipeline_config"]
                print(f"🔧 Pipeline: {pipeline.get('retriever_type')} with {pipeline.get('num_stages', 0)} stages")
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Make sure langgraph and other dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_configuration_options():
    """Show how to switch between different configurations."""
    print(f"\n{'='*60}")
    print("⚙️  How to Switch Retrieval Configurations")
    print('='*60)
    
    print("""
🔧 Available Configurations:

1. basic_dense.yml          → Simple dense retrieval (fast)
2. advanced_reranked.yml    → Dense + CrossEncoder reranking (current)  
3. hybrid_multistage.yml    → Hybrid + multi-stage reranking (best quality)
4. experimental.yml         → For testing new components

🔄 To Switch Configuration:

Method 1 - Using the CLI tool:
   python bin/switch_agent_config.py basic_dense
   
Method 2 - Edit config.yml directly:
   retrieval:
     config_path: "pipelines/configs/retrieval/basic_dense.yml"

Method 3 - Create custom configuration:
   1. Copy an existing config: cp basic_dense.yml my_config.yml
   2. Edit my_config.yml to add/remove components
   3. Switch: python bin/switch_agent_config.py my_config
    """)
    
    print("💡 Benefits:")
    print("   ✅ Switch pipelines without changing code")
    print("   ✅ A/B test different retrieval strategies") 
    print("   ✅ Add rerankers, filters, enhancers easily")
    print("   ✅ Production-ready configuration management")


if __name__ == "__main__":
    print("🚀 Testing LangGraph Agent Integration")
    
    success = demo_agent_retrieval()
    
    if success:
        print("\n🎉 Integration successful!")
        show_configuration_options()
    else:
        print("\n❌ Integration test failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure Qdrant is running: docker-compose up -d")
        print("   2. Check that collections exist: python bin/qdrant_inspector.py list")
        print("   3. Verify dependencies: pip install langgraph langchain-openai")
    
    print(f"\n{'='*60}")
    print("✅ Your agent now uses configurable retrieval pipelines!")
    print("🔄 Change configs with: python bin/switch_agent_config.py <config_name>")
