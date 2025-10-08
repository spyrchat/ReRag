"""
Query Analyzer Node - Breaks down user queries into logical analysis steps.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from logs.utils.logger import get_logger

logger = get_logger(__name__)


def make_query_analyzer(llm):
    """
    Factory to create a query analyzer node.
    
    This node:
    1. Analyzes the user's query
    2. Breaks it down into logical reasoning steps
    3. Identifies key concepts and information needs
    
    Args:
        llm: Language model instance
        
    Returns:
        function: Query analyzer node function
    """
    
    # Prompt for query analysis and decomposition
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing technical questions and breaking them down into logical steps.

Your task:
1. Analyze the user's question carefully
2. Identify the key concepts and information needs
3. Break down the question into 2-4 logical reasoning steps
4. Determine what type of answer is needed

For technical questions about:
- Software engineering, programming, code
- Specific technologies, frameworks, libraries
- Debugging, errors, best practices
- Technical implementation details

Format your analysis as:
**Query Type**: [technical/general/clarification]
**Key Concepts**: [list main technical concepts]
**Reasoning Steps**:
1. [First step of analysis]
2. [Second step of analysis]
3. [Additional steps if needed]
**Information Needs**: [What specific information would help answer this]

Be concise and focused."""),
        ("human", "{question}")
    ])
    
    chain = analysis_prompt | llm
    
    def query_analyzer(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and decompose the user's query.
        
        Args:
            state: Current agent state with 'question'
            
        Returns:
            Updated state with query analysis
        """
        question = state["question"]
        logger.info(f"[QueryAnalyzer] Analyzing query: {question[:100]}...")
        
        try:
            # Get LLM analysis
            response = chain.invoke({"question": question})
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            # Extract query type (simple heuristic)
            query_type = "technical"
            if "Query Type" in analysis:
                if "general" in analysis.lower():
                    query_type = "general"
                elif "clarification" in analysis.lower():
                    query_type = "clarification"
            
            logger.info(f"[QueryAnalyzer] Query type: {query_type}")
            logger.info(f"[QueryAnalyzer] Analysis complete")
            
            return {
                **state,
                "query_analysis": analysis,
                "query_type": query_type,
                "next_node": "router"  # Always go to router next
            }
            
        except Exception as e:
            logger.error(f"[QueryAnalyzer] Analysis failed: {str(e)}")
            # Fallback: treat as technical query
            return {
                **state,
                "query_analysis": f"Unable to analyze query: {str(e)}",
                "query_type": "technical",
                "next_node": "router"
            }
    
    return query_analyzer
