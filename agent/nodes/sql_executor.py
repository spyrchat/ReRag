from typing import Dict, Any
from logs.utils.logger import get_logger

logger = get_logger(__name__)


def make_sql_executor(db):
    """
    Returns an SQL executor node with the provided DB controller injected.
    """
    def sql_executor(state: Dict[str, Any]) -> Dict[str, Any]:
        sql = state.get("sql")
        if not sql:
            logger.warning("[SQL Executor] No SQL found in state.")
            return {
                **state,
                "answer": "No SQL was generated."
            }

        logger.info(f"[SQL Executor] Executing SQL: {sql}")

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
    return sql_executor
