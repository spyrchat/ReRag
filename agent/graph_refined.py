"""
Refined RAG Agent Graph with Multi-Stage Pipeline.

Pipeline Flow:
1. Query Analyzer - Breaks down query into logical steps
2. Router - Decides if retrieval is needed
3. Retriever (conditional) - Retrieves relevant documents
4. Generator - Generates faithful answer
5. Memory Updater - Updates conversation memory (optional)
6. Benchmark Logger - Logs execution for benchmarking
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from langgraph.graph import StateGraph, END
from agent.nodes.query_analyzer import make_query_analyzer
from agent.nodes.retriever import retriever
from agent.nodes.generator import make_generator
from agent.nodes.memory_updater import memory_updater
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

# Get generator prompt style from config (default: "strict")
prompt_style = config.get("generation", {}).get("prompt_style", "strict")
generator = make_generator(llm, prompt_style=prompt_style)

# Initialize benchmark logger (check environment variable or config)
benchmark_enabled = os.getenv("BENCHMARK_MODE", "false").lower() == "true"
benchmark_enabled = config.get("benchmark", {}).get(
    "enabled", benchmark_enabled)
initialize_benchmark_logger(
    output_dir="logs/benchmark",
    enabled=benchmark_enabled
)
benchmark_logger = make_benchmark_logger_node()

# Build the refined agent graph
builder = StateGraph(AgentState)

# Add all nodes (removed router - always retrieve)
builder.add_node("query_analyzer", query_analyzer)
builder.add_node("retriever", retriever)
builder.add_node("generator", generator)
builder.add_node("memory_updater", memory_updater)
builder.add_node("benchmark_logger", benchmark_logger)

# Set entry point
builder.set_entry_point("query_analyzer")

# Define linear flow (always retrieve)
builder.add_edge("query_analyzer", "retriever")
builder.add_edge("retriever", "generator")
# builder.add_edge("generator", "memory_updater")
# builder.add_edge("memory_updater", "benchmark_logger")
builder.add_edge("generator", "benchmark_logger")
builder.add_edge("benchmark_logger", END)

# Compile the graph
graph = builder.compile()

# Print graph info
print("\n" + "=" * 70)
print("REFINED RAG AGENT INITIALIZED")
print("=" * 70)
print(f"LLM Provider: {llm_cfg.get('provider', 'unknown')}")
print(f"LLM Model: {llm_cfg.get('model', 'unknown')}")
print(f"Benchmark Logging: {'ENABLED' if benchmark_enabled else 'DISABLED'}")
print("\nPipeline Flow:")
print("1. Query Analyzer → Breaks down query")
print("2. Retriever → Fetches documents (optional)")
print("3. Generator → Creates answer")
print("4. Memory Updater → Updates history")
print("5. Benchmark Logger → Saves execution data")
print("=" * 70 + "\n")


def print_graph():
    """Print ASCII visualization of the agent graph."""
    try:
        print("\n" + "=" * 70)
        print("RAG AGENT GRAPH VISUALIZATION")
        print("=" * 70)
        print(graph.get_graph().draw_ascii())
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Could not print graph visualization: {e}")


def save_graph_image(output_path: str = "agent_graph_refined.png"):
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
