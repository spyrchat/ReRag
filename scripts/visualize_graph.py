#!/usr/bin/env python3
"""
Graph Visualization Utility for RAG Agent.

Generates visual representations of the agent graphs in multiple formats:
- ASCII art (terminal-friendly)
- Mermaid diagram (web-friendly, paste into https://mermaid.live/)
- PNG/SVG images (requires additional dependencies)

Usage:
    python scripts/visualize_graph.py --mode standard
    python scripts/visualize_graph.py --mode self-rag
    python scripts/visualize_graph.py --mode both --output-dir output/graphs
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_graph(mode: str, output_dir: str = "output/graphs"):
    """
    Visualize the agent graph.
    
    Args:
        mode: "standard" or "self-rag"
        output_dir: Directory to save output files
    """
    print(f"\n{'='*70}")
    print(f"VISUALIZING {mode.upper()} AGENT GRAPH")
    print(f"{'='*70}\n")
    
    # Import the appropriate graph
    if mode == "self-rag":
        from agent.graph_self_rag import graph, print_graph, save_graph_image, get_mermaid_diagram
        graph_name = "self_rag"
    else:
        from agent.graph_refined import graph, print_graph, save_graph_image, get_mermaid_diagram
        graph_name = "refined"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Print ASCII visualization
    print("\n📊 ASCII Visualization:")
    print("-" * 70)
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"❌ ASCII visualization failed: {e}")
    
    # 2. Generate Mermaid diagram
    print("\n📊 Mermaid Diagram:")
    print("-" * 70)
    try:
        mermaid = get_mermaid_diagram()
        mermaid_file = os.path.join(output_dir, f"agent_graph_{graph_name}.mmd")
        
        with open(mermaid_file, "w") as f:
            f.write(mermaid)
        
        print(f"✅ Mermaid diagram saved to: {mermaid_file}")
        print(f"📝 Paste into https://mermaid.live/ to visualize\n")
        print("Preview:")
        print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)
    except Exception as e:
        print(f"❌ Mermaid generation failed: {e}")
    
    # 3. Save as image (PNG)
    print("\n📊 Image Export:")
    print("-" * 70)
    try:
        png_file = os.path.join(output_dir, f"agent_graph_{graph_name}.png")
        
        # Try to save as PNG
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(png_file, "wb") as f:
            f.write(png_bytes)
        
        print(f"✅ PNG image saved to: {png_file}")
    except ImportError as e:
        print(f"⚠️  PNG export requires additional packages:")
        print(f"   pip install grandalf")
        print(f"   (optional: pip install pygraphviz for better layouts)")
    except Exception as e:
        print(f"❌ Image export failed: {e}")
    
    # 4. Graph statistics
    print("\n📊 Graph Statistics:")
    print("-" * 70)
    try:
        graph_obj = graph.get_graph()
        nodes = list(graph_obj.nodes.keys())
        edges = graph_obj.edges
        
        print(f"Nodes ({len(nodes)}):")
        for node in nodes:
            print(f"  - {node}")
        
        print(f"\nEdges ({len(edges)}):")
        for edge in edges:
            print(f"  - {edge}")
    except Exception as e:
        print(f"❌ Statistics generation failed: {e}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RAG Agent Graph Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize standard RAG graph
  python scripts/visualize_graph.py --mode standard
  
  # Visualize Self-RAG graph
  python scripts/visualize_graph.py --mode self-rag
  
  # Visualize both graphs
  python scripts/visualize_graph.py --mode both
  
  # Custom output directory
  python scripts/visualize_graph.py --mode both --output-dir my_graphs
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["standard", "self-rag", "both"],
        default="standard",
        help="Which graph to visualize"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output/graphs",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--format",
        choices=["all", "ascii", "mermaid", "image"],
        default="all",
        help="Output format (default: all)"
    )
    
    args = parser.parse_args()
    
    # Visualize requested graphs
    if args.mode == "both":
        visualize_graph("standard", args.output_dir)
        visualize_graph("self-rag", args.output_dir)
    else:
        visualize_graph(args.mode, args.output_dir)
    
    print("✅ Visualization complete!")
    print(f"📁 Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
