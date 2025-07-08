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
    answer = final_state.get("answer")
    chat_history = final_state.get("chat_history", [])

    print(f"Agent: {answer}")
