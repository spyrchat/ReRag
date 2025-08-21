from langgraph.graph import StateGraph
from agent.nodes.query_interpreter import make_query_interpreter
from agent.nodes.sql_planner import make_sql_planner
from agent.nodes.sql_executor import make_sql_executor
from agent.nodes.retriever import make_configurable_retriever
from agent.nodes.generator import make_generator
from agent.nodes.memory_updater import memory_updater
from agent.schema import AgentState
from config.config_loader import load_config
from langchain_openai import ChatOpenAI
from database.postgres_controller import PostgresController

# Load config
config = load_config("config.yml")

# Setup LLM
llm_cfg = config["llm"]
llm = ChatOpenAI(model=llm_cfg.get("model", "gpt-4.1-mini"),
                 temperature=llm_cfg.get("temperature", 0.0))

# Setup SQL database
sql_db = PostgresController()

# Setup configurable retriever node
retrieval_config_path = config.get("retrieval", {}).get("config_path", "pipelines/configs/retrieval/basic_dense.yml")
retriever = make_configurable_retriever(config_path=retrieval_config_path)
sql_executor = make_sql_executor(sql_db)
sql_planner = make_sql_planner(llm)
# Setup other nodes
generator = make_generator(llm)
query_interpreter = make_query_interpreter(llm)

# Build the graph
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
    "generator": "generator",
})

builder.add_edge("sql_planner", "sql_executor")
builder.add_edge("sql_executor", "generator")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", "memory_updater")
graph = builder.compile()
