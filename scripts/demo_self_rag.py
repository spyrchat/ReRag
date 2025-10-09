#!/usr/bin/env python3
"""
Demo script to visualize Self-RAG verification loop.
Shows what happens at each iteration step.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from agent.graph_self_rag import graph
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

console = Console()


def visualize_self_rag_process(question: str):
    """
    Run a query and visualize the Self-RAG verification loop.
    """
    console.print(
        "\n[bold cyan]═══ Self-RAG Process Visualization ═══[/bold cyan]\n")

    # Show the question
    console.print(Panel(
        f"[yellow]{question}[/yellow]",
        title="📝 User Question",
        border_style="yellow"
    ))

    # Run the agent
    state = {
        "question": question,
        "chat_history": []
    }

    console.print("\n[dim]Running agent...[/dim]\n")
    final_state = graph.invoke(state)

    # Extract metadata
    metadata = final_state.get("self_rag_metadata", {})
    verification = final_state.get("verification", {})
    answer = final_state.get("answer", "")

    # Show iteration breakdown
    if "iteration_history" in metadata:
        console.print(Panel(
            "[bold]Self-RAG Verification Loop[/bold]",
            style="cyan"
        ))

        for i, iteration in enumerate(metadata["iteration_history"], 1):
            action = iteration.get("action", "unknown")
            ver = iteration.get("verification", {})
            is_faithful = ver.get("is_faithful", False)
            hallucination = ver.get("hallucination_detected", False)
            severity = ver.get("severity", "unknown")

            # Iteration header
            if action == "initial":
                console.print(
                    f"\n[bold green]Iteration {i}: Initial Generation[/bold green]")
            else:
                console.print(
                    f"\n[bold yellow]Iteration {i}: Refinement[/bold yellow]")

            # Verification results
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            faithful_icon = "✅" if is_faithful else "❌"
            halluc_icon = "⚠️" if hallucination else "✓"

            table.add_row("Faithful to Context",
                          f"{faithful_icon} {is_faithful}")
            table.add_row("Hallucination Detected",
                          f"{halluc_icon} {hallucination}")
            table.add_row("Severity", severity)
            table.add_row("Confidence", f"{ver.get('confidence', 0):.2f}")

            console.print(table)

            # Show issues if any
            if ver.get("issues"):
                console.print("\n[bold red]Issues Found:[/bold red]")
                for issue in ver["issues"]:
                    console.print(f"  • {issue}")

            # Show decision
            if is_faithful:
                console.print(
                    f"\n[bold green]→ Decision: Accept answer[/bold green]")
                break
            elif i < len(metadata["iteration_history"]):
                console.print(
                    f"\n[bold yellow]→ Decision: Refine answer[/bold yellow]")
            else:
                console.print(
                    f"\n[bold red]→ Decision: Max iterations reached[/bold red]")

    # Final summary
    console.print("\n")
    console.print(Panel(
        f"""[bold]Summary[/bold]
        
Total Iterations: {metadata.get('iterations', 1)}
Converged: {'✅ Yes' if metadata.get('converged') else '❌ No'}
Hallucinations Corrected: {'✅ Yes' if metadata.get('hallucinations_corrected') else '✓ None detected'}
Max Iterations Reached: {'⚠️ Yes' if metadata.get('max_iterations_reached') else '✓ No'}""",
        title="📊 Self-RAG Metrics",
        border_style="green" if metadata.get('converged') else "yellow"
    ))

    # Show final answer
    console.print("\n")
    console.print(Panel(
        f"[white]{answer}[/white]",
        title="✨ Final Answer",
        border_style="green"
    ))


def compare_modes(question: str):
    """
    Compare Standard RAG vs Self-RAG for the same question.
    """
    console.print(
        "\n[bold magenta]═══ Standard RAG vs Self-RAG Comparison ═══[/bold magenta]\n")

    # Standard RAG
    console.print("[bold cyan]Mode 1: Standard RAG[/bold cyan]")
    console.print("[dim]No verification loop, single generation[/dim]\n")

    from agent.graph_refined import graph as standard_graph
    state = {"question": question, "chat_history": []}
    standard_result = standard_graph.invoke(state)
    standard_answer = standard_result.get("answer", "")

    console.print(Panel(
        f"[white]{standard_answer[:300]}...[/white]" if len(
            standard_answer) > 300 else f"[white]{standard_answer}[/white]",
        title="Standard RAG Answer",
        border_style="blue"
    ))

    # Self-RAG
    console.print("\n[bold cyan]Mode 2: Self-RAG[/bold cyan]")
    console.print("[dim]With verification loop and refinement[/dim]\n")

    state = {"question": question, "chat_history": []}
    selfrag_result = graph.invoke(state)
    selfrag_answer = selfrag_result.get("answer", "")
    metadata = selfrag_result.get("self_rag_metadata", {})

    console.print(Panel(
        f"[white]{selfrag_answer[:300]}...[/white]" if len(
            selfrag_answer) > 300 else f"[white]{selfrag_answer}[/white]",
        title=f"Self-RAG Answer ({metadata.get('iterations', 1)} iterations)",
        border_style="green"
    ))

    # Comparison table
    console.print("\n[bold]Comparison:[/bold]")
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Standard RAG", style="blue")
    table.add_column("Self-RAG", style="green")

    table.add_row(
        "Verification",
        "❌ None",
        f"✅ {metadata.get('iterations', 1)} iteration(s)"
    )
    table.add_row(
        "Hallucination Check",
        "❌ No",
        "✅ Yes" if metadata.get('converged') else "⚠️ Attempted"
    )
    table.add_row(
        "Refinement",
        "❌ No",
        "✅ Yes" if metadata.get('hallucinations_corrected') else "✓ Not needed"
    )
    table.add_row(
        "Answer Length",
        f"{len(standard_answer)} chars",
        f"{len(selfrag_answer)} chars"
    )

    console.print(table)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Self-RAG verification process")
    parser.add_argument(
        "question",
        nargs="?",
        default="How do I prevent SQL injection in Python?",
        help="Question to ask the agent"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Standard RAG vs Self-RAG"
    )

    args = parser.parse_args()

    try:
        if args.compare:
            compare_modes(args.question)
        else:
            visualize_self_rag_process(args.question)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
