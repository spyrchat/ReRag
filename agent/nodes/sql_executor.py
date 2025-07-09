from typing import Dict, Any
from database.postgres_controller import PostgresController
from logs.utils.logger import get_logger

logger = get_logger("sql_executor")


def sql_executor(state: Dict[str, Any]) -> Dict[str, Any]:
    sql = state.get("sql")
    if not sql:
        logger.warning("[SQL Executor] No SQL found in state.")
        return {
            **state,
            "answer": "No SQL was generated."
        }

    logger.info(f"[SQL Executor] Executing SQL: {sql}")
    db = PostgresController()

    try:
        with db.get_session() as session:
            result = session.execute(sql)
            rows = result.fetchall()

            if not rows:
                logger.info("[SQL Executor] No rows returned.")
                return {**state, "answer": "No results found."}

            columns = result.keys()
            output = [dict(zip(columns, row)) for row in rows]

            logger.info(f"[SQL Executor] Returned {len(rows)} rows.")
            return {
                **state,
                "answer": str(output)
            }

    except Exception as e:
        logger.error(f"[SQL Executor] SQL error: {e}")
        return {
            **state,
            "answer": f"SQL execution failed: {str(e)}"
        }
