from agent.graph import graph

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

    if "context" in final_state:
        print("\n[Retrieved Context]")
        print(final_state["context"])
    print("---\n")
