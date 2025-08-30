from langgraph.graph import StateGraph
from agent.nodes.query_interpreter import make_query_interpreter
from agent.nodes.retriever import make_configurable_retriever
from agent.nodes.generator import make_generator
from agent.nodes.memory_updater import memory_updater
from agent.schema import AgentState
from config.config_loader import load_config
from langchain_openai import ChatOpenAI

# Load config
config = load_config("config.yml")

# Setup LLM
llm_cfg = config["llm"]
llm = ChatOpenAI(model=llm_cfg.get("model", "gpt-4.1-mini"),
                 temperature=llm_cfg.get("temperature", 0.0))

# Setup configurable retriever node
retrieval_config_path = config.get("agent_retrieval", {}).get(
    "config_path", "pipelines/configs/retrieval/modern_hybrid.yml")
retriever = make_configurable_retriever(config_path=retrieval_config_path)

# Setup other nodes
generator = make_generator(llm)
query_interpreter = make_query_interpreter(llm)

# Build the graph
builder = StateGraph(AgentState)
builder.add_node("query_interpreter", query_interpreter)
builder.add_node("retriever", retriever)
builder.add_node("generator", generator)
builder.add_node("memory_updater", memory_updater)
builder.set_entry_point("query_interpreter")

builder.add_conditional_edges("query_interpreter", lambda state: state["next_node"], {
    "retriever": "retriever",
    "generator": "generator",
})

builder.add_edge("retriever", "generator")
builder.add_edge("generator", "memory_updater")
graph = builder.compile()
