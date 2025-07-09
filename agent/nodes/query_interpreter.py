import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# === Setup Logging ===
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/query_interpreter.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def query_interpreter(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    reference_date = state.get("reference_date") or datetime.now(
        timezone.utc).date().isoformat()

    prompt = f"""
You are a planner agent for a modular RAG pipeline.

Your job is to:
1. Understand the user's intent.
2. Decide if the answer requires accessing structured SQL tables, unstructured document chunks, both, or NEITHER (i.e., can be answered directly with no retrieval).
3. Output a plan, a query type ("sql", "text", "none"), and a next_node to route to.

Today's date is: {reference_date}

Question: {question}

---

Respond in valid JSON using this format:
{{
  "query_type": "sql" | "text" | "none",
  "next_node": "sql_planner" | "retriever" | "merge" | "generator",
  "plan": ["Step 1: ...", "Step 2: ..."],
  "reasoning": "..."
}}

Examples:
# Example 1: Direct answer (no DB needed)
{{
  "query_type": "none",
  "next_node": "generator",
  "plan": ["Recognize this as a chitchat or general info question.", "Answer directly."],
  "reasoning": "No retrieval required for this question."
}}

# Example 2: SQL needed
{{
  "query_type": "sql",
  "next_node": "sql_planner",
  "plan": ["Detect that SQL is needed.", "Route to sql_planner."],
  "reasoning": "This question is about structured data."
}}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        parsed = json.loads(content)
    except Exception as e:
        logging.error("Error in query_interpreter: %s", str(e))
        logging.error("LLM response: %s",
                      response.content if 'response' in locals() else 'None')
        # Add 'generator' as a fallback if parsing fails
        parsed = {
            "query_type": "none",
            "next_node": "generator",
            "plan": ["Fallback to generator: answer directly."],
            "reasoning": "Failed to parse model response. Defaulting to direct answer."
        }

    logging.info("=== Question ===")
    logging.info(question)
    logging.info("=== Parsed Result ===")
    logging.info(json.dumps(parsed, indent=2))

    return {
        "question": question,
        "reference_date": reference_date,
        **parsed
    }
