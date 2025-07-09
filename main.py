from agent.graph import graph
from logs.utils.logger import get_logger

logger = get_logger("chat")

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    state = {
        "question": user_input,
        "chat_history": chat_history
    }

    final_state = graph.invoke(state)
    answer = final_state.get("answer", "[No answer returned]")
    chat_history = final_state.get("chat_history", [])

    print("\n---")
    print(f"Agent: {answer}")

    # Log to file
    logger.info(f"User: {user_input}")
    logger.info(f"Agent: {answer}")

    # if "context" in final_state:
    #     context = final_state["context"]
    #     print("\n[Retrieved Context]")
    #     print(context)
    #     logger.info(
    #         f"Retrieved Context:\n{context[:500]}{'...' if len(context) > 500 else ''}")

    if "sql" in final_state:
        logger.info(f"Generated SQL: {final_state['sql']}")

    if "error" in final_state:
        logger.error(f"Execution error: {final_state['error']}")

    print("---\n")
