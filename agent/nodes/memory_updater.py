from typing import Dict, Any
AgentState = Dict[str, Any]


def memory_updater(state: AgentState) -> AgentState:
    history = state.get("chat_history", [])
    question = state.get("question", "")
    answer = state.get("answer", "")

    history += [f"User: {question}", f"Assistant: {answer}"]
    state["chat_history"] = history[-20:]
    return state
