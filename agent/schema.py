from typing import TypedDict, List, Optional, Union


class AgentState(TypedDict, total=False):
    question: str
    reference_date: str
    next_node: str
    sql: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    chat_history: List[str]
