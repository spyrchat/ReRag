"""
Self-RAG Agent Graph with Iterative Refinement.

Pipeline Flow:
1. Query Analyzer - Breaks down query into logical steps
2. Retriever - Retrieves relevant documents
3. Self-RAG Generator - Generates answer with verification loop
   - Generates initial answer
   - Verifies for hallucinations
   - Refines if needed (up to max_iterations)
4. Benchmark Logger - Logs execution for benchmarking

Key Difference from Standard Graph:
- Replaces standard generator with self-correcting generator
- Adds verification feedback loop
- Tracks refinement iterations
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from langgraph.graph import StateGraph, END
from agent.nodes.query_analyzer import make_query_analyzer
from agent.nodes.retriever import retriever
from agent.nodes.self_rag_generator import make_self_rag_generator
from agent.nodes.benchmark_logger import make_benchmark_logger_node, initialize_benchmark_logger
from agent.schema import AgentState
from config.config_loader import load_config
from config.llm_factory import create_llm
import os

# Load configuration
config = load_config("config.yml")

# Setup LLM using factory (supports OpenAI, Ollama, etc.)
llm_cfg = config["llm"]
llm = create_llm(llm_cfg)

# Setup nodes
query_analyzer = make_query_analyzer(llm)

# Get max iterations from config (default: 3)
max_iterations = config.get("self_rag", {}).get("max_iterations", 3)
self_rag_generator = make_self_rag_generator(llm, max_iterations=max_iterations)

# Initialize benchmark logger
benchmark_enabled = os.getenv("BENCHMARK_MODE", "false").lower() == "true"
benchmark_enabled = config.get("benchmark", {}).get("enabled", benchmark_enabled)
initialize_benchmark_logger(
    output_dir="logs/benchmark",
    enabled=benchmark_enabled
)
benchmark_logger = make_benchmark_logger_node()

# Build the self-RAG agent graph
builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("query_analyzer", query_analyzer)
builder.add_node("retriever", retriever)
builder.add_node("self_rag_generator", self_rag_generator)
builder.add_node("benchmark_logger", benchmark_logger)

# Set entry point
builder.set_entry_point("query_analyzer")

# Define linear flow with self-RAG
builder.add_edge("query_analyzer", "retriever")
builder.add_edge("retriever", "self_rag_generator")
builder.add_edge("self_rag_generator", "benchmark_logger")
builder.add_edge("benchmark_logger", END)

# Compile the graph
graph = builder.compile()

# Print graph info
print("\n" + "=" * 70)
print("SELF-RAG AGENT INITIALIZED")
print("=" * 70)
print(f"LLM Provider: {llm_cfg.get('provider', 'unknown')}")
print(f"LLM Model: {llm_cfg.get('model', 'unknown')}")
print(f"Max Refinement Iterations: {max_iterations}")
print(f"Benchmark Logging: {'ENABLED' if benchmark_enabled else 'DISABLED'}")
print("\nPipeline Flow:")
print("1. Query Analyzer → Breaks down query")
print("2. Retriever → Fetches relevant documents")
print("3. Self-RAG Generator → Generate + Verify + Refine loop")
print("   - Generates initial answer")
print("   - Verifies for hallucinations")
print("   - Refines if issues detected (up to max_iterations)")
print("4. Benchmark Logger → Saves execution data")
print("\nSelf-RAG Features:")
print("✓ Automatic hallucination detection")
print("✓ Iterative refinement with verification feedback")
print("✓ Tracks convergence and corrections")
print("=" * 70 + "\n")


def print_graph():
    """Print ASCII visualization of the agent graph."""
    try:
        print("\n" + "=" * 70)
        print("SELF-RAG AGENT GRAPH VISUALIZATION")
        print("=" * 70)
        print(graph.get_graph().draw_ascii())
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Could not print graph visualization: {e}")


def save_graph_image(output_path: str = "agent_graph_self_rag.png"):
    """
    Save graph visualization as an image file.
    Requires: pip install grandalf pygraphviz
    
    Args:
        output_path: Path to save the image (supports .png, .svg, .jpg)
    """
    try:
        from PIL import Image
        import io
        
        # Get the graph visualization as PNG bytes
        png_bytes = graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        
        print(f"✓ Graph visualization saved to: {output_path}")
        return output_path
    except ImportError:
        print("⚠ To save graph images, install: pip install grandalf pillow")
        return None
    except Exception as e:
        print(f"✗ Could not save graph image: {e}")
        return None


def get_mermaid_diagram() -> str:
    """
    Get Mermaid diagram representation of the graph.
    Can be pasted into https://mermaid.live/ for visualization.
    
    Returns:
        Mermaid diagram as string
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        print(f"Could not generate Mermaid diagram: {e}")
        return ""
