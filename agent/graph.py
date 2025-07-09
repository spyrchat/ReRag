from langgraph.graph import StateGraph
from agent.nodes.query_interpreter import query_interpreter
from agent.nodes.sql_planner import sql_planner
from agent.nodes.sql_executor import sql_executor
from agent.nodes.retriever import retriever
from agent.nodes.generator import make_generatorgenerator
from agent.nodes.memory_updater import memory_updater
from agent.schema import AgentState
from config import load_config
builder = StateGraph(AgentState)


builder.add_node("query_interpreter", query_interpreter)
builder.add_node("sql_planner", sql_planner)
builder.add_node("sql_executor", sql_executor)
builder.add_node("retriever", retriever)
builder.add_node("generator", generator)
builder.add_node("memory_updater", memory_updater)
builder.set_entry_point("query_interpreter")

builder.add_conditional_edges("query_interpreter", lambda state: state["next_node"], {
    "sql_planner": "sql_planner",
    "retriever": "retriever",
    "generator": "generator",  # <-- added for direct answer path
})


builder.add_edge("sql_planner", "sql_executor")
builder.add_edge("sql_executor", "generator")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", "memory_updater")
graph = builder.compile()
