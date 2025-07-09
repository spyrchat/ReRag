import os
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import logging
import json

logging.basicConfig(level=logging.INFO)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

sql_prompt = PromptTemplate.from_template(
    """You are a text-to-SQL translator.

The available table is:
table_assets(id, doc_id, page_number, table_json, caption, created_at)

Today's date is: {reference_date}

User question:
{question}

---

Respond ONLY with valid SQL syntax, no explanation or markdown."""
)

def sql_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    reference_date = state.get("reference_date", "2025-01-01")
    prompt = sql_prompt.format(question=question, reference_date=reference_date)

    try:
        response = llm.invoke(prompt)
        sql = response.content.strip()
        logging.info(f"[SQL Planner] SQL: {sql}")
        return {
            **state,
            "sql": sql
        }
    except Exception as e:
        logging.error(f"SQL planner failed: {e}")
        return {
            **state,
            "sql": None,
            "sql_error": str(e)
        }
