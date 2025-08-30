from typing import TypedDict, List, Optional, Union, Dict, Any
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    Agent state schema that defines all possible state variables for the LangGraph agent.

    Attributes:
        question (str): The user's input question
        reference_date (str): Reference date for temporal queries
        next_node (str): Next node to execute in the agent graph
        context (str, optional): Contextual information for response generation
        answer (str, optional): Final answer to return to user
        chat_history (List[str]): Previous conversation history

        # Enhanced retrieval fields for configurable pipeline integration
        retrieved_documents (List[Document], optional): Full document objects with metadata
        retrieval_metadata (Dict[str, Any], optional): Pipeline info, scores, method details
        retrieval_top_k (int, optional): Override default top_k for dynamic result count
        error (str, optional): Error messages from any processing stage
    """
    question: str
    reference_date: str
    next_node: str
    context: Optional[str]
    answer: Optional[str]
    chat_history: List[str]

    # Enhanced retrieval fields
    # Full document objects with metadata
    retrieved_documents: Optional[List[Document]]
    # Pipeline info, scores, etc.
    retrieval_metadata: Optional[Dict[str, Any]]
    retrieval_top_k: Optional[int]                 # Override default top_k
    error: Optional[str]                           # Error messages
