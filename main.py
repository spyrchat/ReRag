"""
Main application entry point for the RAG agent.
Provides an interactive chat interface for the LangGraph agent with configurable retrieval.
Supports both standard and Self-RAG modes.
"""

import argparse
import os
from logs.utils.logger import get_logger

logger = get_logger("chat")


def main():
    """
    Main chat loop for the RAG agent.
    Handles user input, agent invocation, and response display.
    """
    parser = argparse.ArgumentParser(description="RAG Agent Chat Interface")
    parser.add_argument(
        "--mode",
        choices=["standard", "self-rag"],
        default="standard",
        help="Agent mode: standard (default) or self-rag (with verification loop)"
    )
    parser.add_argument("--query", help="Single query mode (non-interactive)")
    args = parser.parse_args()

    # Load the appropriate graph based on mode
    if args.mode == "self-rag":
        from agent.graph_self_rag import graph
        logger.info("Using Self-RAG mode with verification loop")
        print("[Mode: Self-RAG - Iterative refinement enabled]")
    else:
        from agent.graph_refined import graph
        logger.info("Using standard RAG mode")
        print("[Mode: Standard RAG]")

    if args.query:
        # Single query mode
        state = {
            "question": args.query,
            "chat_history": []
        }

        try:
            final_state = graph.invoke(state)
            answer = final_state.get("answer", "[No answer returned]")
            print(f"Query: {args.query}")
            print(f"Answer: {answer}")

            if "error" in final_state:
                logger.error(f"Execution error: {final_state['error']}")
                print(f"[Error occurred: {final_state['error']}]")

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            print(f"[Error: Agent failed to process your request: {e}]")

        return

    # Interactive mode
    chat_history = []

    print("RAG Agent - Interactive Chat")
    print("Type 'exit' or 'quit' to end the conversation")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state = {
            "question": user_input,
            "chat_history": chat_history
        }

        try:
            final_state = graph.invoke(state)
            answer = final_state.get("answer", "[No answer returned]")
            chat_history = final_state.get("chat_history", [])

            print("\n---")
            print(f"Agent: {answer}")

            # Log to file
            logger.info(f"User: {user_input}")
            logger.info(f"Agent: {answer}")

            if "error" in final_state:
                logger.error(f"Execution error: {final_state['error']}")
                print(f"[Error occurred: {final_state['error']}]")

            print("---\n")

        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            print(f"[Error: Agent failed to process your request: {e}]")
            print("---\n")


if __name__ == "__main__":
    main()
