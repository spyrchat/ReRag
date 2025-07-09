import logging
from typing import Dict, Any
from database.postgres_controller import PostgresController

logging.basicConfig(level=logging.INFO)

def sql_executor(state: Dict[str, Any]) -> Dict[str, Any]:
    sql = state.get("sql")
    if not sql:
        return {
            **state,
            "answer": "No SQL was generated."
        }

    db = PostgresController()
    try:
        with db.get_session() as session:
            result = session.execute(sql)
            rows = result.fetchall()
            if not rows:
                return {**state, "answer": "No results found."}

            columns = result.keys()
            output = [dict(zip(columns, row)) for row in rows]

            return {
                **state,
                "answer": str(output)
            }

    except Exception as e:
        logging.error(f"[SQL Executor] SQL error: {e}")
        return {
            **state,
            "answer": f"SQL execution failed: {str(e)}"
        }