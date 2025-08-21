from typing import TypedDict, List, Optional, Union, Dict, Any
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    question: str
    reference_date: str
    next_node: str
    sql: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    chat_history: List[str]
    
    # Enhanced retrieval fields
    retrieved_documents: Optional[List[Document]]  # Full document objects with metadata
    retrieval_metadata: Optional[Dict[str, Any]]   # Pipeline info, scores, etc.
    retrieval_top_k: Optional[int]                 # Override default top_k
    error: Optional[str]                           # Error messages
