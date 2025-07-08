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
2. Decide if the answer requires accessing structured SQL tables, unstructured document chunks, or both.
3. Output a plan, a query type ("sql", "text"), and a next_node to route to.

Today's date is: {reference_date}

Question: {question}

---

Respond in valid JSON using this format:
{{
  "query_type": "sql" | "text",
  "next_node": "sql_planner" | "retriever" | "merge",
  "plan": ["Step 1: ...", "Step 2: ..."],
  "reasoning": "..."
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
        parsed = {
            "query_type": "text",
            "next_node": "retriever",
            "plan": ["Fallback to text retriever."],
            "reasoning": "Failed to parse model response."
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
