from typing import TypedDict, List, Optional, Union, Dict, Any
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    Agent state schema for refined RAG pipeline.

    Attributes:
        # User input
        question (str): The user's input question
        chat_history (List[str]): Previous conversation history
        
        # Query analysis
        query_analysis (str): LLM analysis breaking down the query
        query_type (str): Type of query (technical/general/clarification)
        
        # Routing decision
        needs_retrieval (bool): Whether database retrieval is needed
        routing_decision (str): Routing decision reasoning
        
        # Retrieval data
        context (str, optional): Retrieved context for answer generation
        retrieved_documents (List[Document], optional): Full document objects with metadata
        retrieval_metadata (Dict[str, Any], optional): Pipeline info, scores, method details
        retrieval_top_k (int, optional): Override default top_k
        
        # Generation
        answer (str, optional): Final answer to return to user
        generation_mode (str, optional): How answer was generated (context/direct/error)
        
        # Control flow
        next_node (str): Next node to execute in the agent graph
        
        # Metadata
        reference_date (str): Reference date for temporal queries
        error (str, optional): Error messages from any processing stage
    """
    # User input
    question: str
    chat_history: List[str]
    
    # Query analysis
    query_analysis: Optional[str]
    query_type: Optional[str]
    
    # Routing
    needs_retrieval: Optional[bool]
    routing_decision: Optional[str]
    
    # Retrieval
    context: Optional[str]
    retrieved_documents: Optional[List[Document]]
    retrieval_metadata: Optional[Dict[str, Any]]
    retrieval_top_k: Optional[int]
    
    # Generation
    answer: Optional[str]
    generation_mode: Optional[str]
    
    # Control
    next_node: str
    
    # Metadata
    reference_date: Optional[str]
    error: Optional[str]
