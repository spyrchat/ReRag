from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger

logger = get_logger(__name__)

# Example default prompt (could also load from config for full modularity)
DEFAULT_SQL_PROMPT = """
You are a text-to-SQL translator.

The available table is:
table_assets(id, doc_id, page_number, table_json, caption, created_at)

Today's date is: {reference_date}

User question:
{question}

---

Respond ONLY with valid SQL syntax, no explanation or markdown.
"""


def make_sql_planner(llm, sql_prompt: PromptTemplate = None):
    """
    Returns a SQL planner node function using the provided LLM and prompt.
    """
    prompt = sql_prompt or PromptTemplate.from_template(DEFAULT_SQL_PROMPT)

    def sql_planner(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        reference_date = state.get("reference_date", "2025-01-01")
        formatted_prompt = prompt.format(
            question=question, reference_date=reference_date
        )

        try:
            response = llm.invoke(formatted_prompt)
            sql = response.content.strip()
            logger.info(f"[SQL Planner] SQL generated: {sql}")
            return {
                **state,
                "sql": sql
            }
        except Exception as e:
            logger.error(f"[SQL Planner] Failed to generate SQL: {e}")
            return {
                **state,
                "sql": None,
                "sql_error": str(e)
            }
    return sql_planner
