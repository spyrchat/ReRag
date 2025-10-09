#!/usr/bin/env python3
"""
Quick demo of graph visualization capabilities.

Run this to see how to use the built-in visualization functions.
"""

from agent.graph_self_rag import graph, print_graph, save_graph_image, get_mermaid_diagram

print("=" * 70)
print("LANGGRAPH VISUALIZATION DEMO")
print("=" * 70)

# Option 1: ASCII visualization (best for terminal)
print("\n1️⃣  ASCII Visualization (Terminal-Friendly):\n")
print(graph.get_graph().draw_ascii())

# Option 2: Mermaid diagram (paste into https://mermaid.live/)
print("\n2️⃣  Mermaid Diagram (Copy and paste into https://mermaid.live/):\n")
mermaid = get_mermaid_diagram()
print(mermaid)

# Save Mermaid to file
with open("output/agent_graph.mmd", "w") as f:
    f.write(mermaid)
print("\n✅ Saved to: output/agent_graph.mmd")

# Option 3: PNG image (requires dependencies)
print("\n3️⃣  PNG Image Export:")
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("output/agent_graph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ Saved to: output/agent_graph.png")
except Exception as e:
    print(f"⚠️  PNG export failed: {e}")
    print("   Install dependencies: pip install grandalf")

# Option 4: Graph structure info
print("\n4️⃣  Graph Structure:")
graph_obj = graph.get_graph()
print(f"   Nodes: {list(graph_obj.nodes.keys())}")
print(f"   Edges: {graph_obj.edges}")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
