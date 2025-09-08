import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger

logger = get_logger(__name__)

# Optional: Move prompt to config for full modularity.
QUERY_INTERPRETER_PROMPT = """
You are a planner agent for a modular RAG pipeline.

Your job is to:
1. Understand the user's intent.
2. Decide if the answer requires accessing unstructured document chunks or can be answered directly without retrieval.
3. Output a plan, a query type ("text" or "none"), and a next_node to route to.

Today's date is: {reference_date}

Question: {question}

---

Respond in valid JSON using this format:
{{
  "query_type": "text" | "none",
  "next_node": "retriever" | "generator",
  "plan": ["Step 1: ...", "Step 2: ..."],
  "reasoning": "..."
}}

Examples:
# Example 1: Direct answer (no retrieval needed)
{{
  "query_type": "none",
  "next_node": "generator",
  "plan": ["Recognize this as a chitchat or general info question.", "Answer directly."],
  "reasoning": "No retrieval required for this question."
}}

# Example 2: Document retrieval needed
{{
  "query_type": "text",
  "next_node": "retriever",
  "plan": ["Detect that document search is needed.", "Route to retriever."],
  "reasoning": "This question requires searching through documents."
}}
"""


def make_query_interpreter(llm):
    """
    Factory to return a query_interpreter node with the provided LLM.
    """
    def query_interpreter(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        reference_date = state.get("reference_date") or datetime.now(
            timezone.utc).date().isoformat()

        prompt = QUERY_INTERPRETER_PROMPT.format(
            question=question, reference_date=reference_date
        )

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            parsed = json.loads(content)
        except Exception as e:
            logger.error("Error in query_interpreter: %s", str(e))
            logger.error("LLM response: %s",
                         response.content if 'response' in locals() else 'None')
            parsed = {
                "query_type": "none",
                "next_node": "generator",
                "plan": ["Fallback to generator: answer directly."],
                "reasoning": "Failed to parse model response. Defaulting to direct answer."
            }

        logger.info("=== Question ===")
        logger.info(question)
        logger.info("=== Parsed Result ===")
        logger.info(json.dumps(parsed, indent=2))

        return {
            "question": question,
            "reference_date": reference_date,
            **parsed
        }
    return query_interpreter
