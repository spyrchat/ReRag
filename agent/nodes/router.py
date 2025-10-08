"""
Router Node - Decides whether to retrieve from database or answer directly.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from logs.utils.logger import get_logger

logger = get_logger(__name__)


def make_router(llm):
    """
    Factory to create a routing decision node.
    
    This node determines:
    1. Whether the query needs database retrieval
    2. Routes to either retriever or direct answer generator
    
    Args:
        llm: Language model instance
        
    Returns:
        function: Router node function
    """
    
    # Prompt for routing decision
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing agent for a StackOverflow Q&A system.

Your database contains:
- StackOverflow questions and answers about software engineering
- Code examples, debugging solutions, best practices
- Technical explanations for programming concepts
- Framework and library usage examples

Your task: Decide if retrieving from the database would help answer the question.

**Retrieve from database** if the question is about:
- Specific programming problems or errors
- How to implement something in code
- Best practices for software development
- Technical explanations of programming concepts
- Framework/library usage
- Code examples or snippets
- Debugging issues

**Answer directly** (without retrieval) if the question is:
- Simple greetings or chitchat
- General knowledge not related to programming
- Meta questions about the system itself
- Questions already fully answered in conversation history
- Very simple factual questions not requiring code context

Respond with EXACTLY one word:
- "RETRIEVE" if database retrieval would help
- "DIRECT" if you can answer directly without retrieval

Think carefully, but respond with only one word."""),
        ("human", """Question: {question}

Query Analysis: {query_analysis}

Decision:""")
    ])
    
    chain = routing_prompt | llm
    
    def router(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the query to retriever or direct answering.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with routing decision
        """
        question = state["question"]
        query_analysis = state.get("query_analysis", "No analysis available")
        
        logger.info(f"[Router] Making routing decision for: {question[:100]}...")
        
        try:
            # Get LLM decision
            response = chain.invoke({
                "question": question,
                "query_analysis": query_analysis
            })
            
            decision = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            # Parse decision
            needs_retrieval = "RETRIEVE" in decision
            
            if needs_retrieval:
                next_node = "retriever"
                logger.info("[Router] Decision: RETRIEVE - Routing to retriever")
            else:
                next_node = "generator"
                logger.info("[Router] Decision: DIRECT - Routing to generator")
            
            return {
                **state,
                "needs_retrieval": needs_retrieval,
                "routing_decision": decision,
                "next_node": next_node
            }
            
        except Exception as e:
            logger.error(f"[Router] Routing failed: {str(e)}")
            # Fallback: assume retrieval is needed for safety
            logger.warning("[Router] Fallback: Routing to retriever")
            return {
                **state,
                "needs_retrieval": True,
                "routing_decision": "RETRIEVE (fallback)",
                "next_node": "retriever"
            }
    
    return router


def router_condition(state: Dict[str, Any]) -> str:
    """
    Conditional edge function for routing.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name based on routing decision
    """
    next_node = state.get("next_node", "generator")
    logger.info(f"[RouterCondition] Routing to: {next_node}")
    return next_node
