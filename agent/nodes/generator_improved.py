"""
Improved Generator Node - Generates faithful responses based on context.
Supports both context-based and direct answering modes.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from logs.utils.logger import get_logger

logger = get_logger(__name__)


def make_generator(llm):
    """
    Factory to create a generator node with faithful answer generation.
    
    This generator:
    1. Checks if context is available from retrieval
    2. Uses appropriate prompting strategy
    3. Generates faithful, grounded answers
    
    Args:
        llm: Language model instance
        
    Returns:
        function: Generator node function
    """
    
    # Prompt for context-based generation (with retrieval)
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert software engineering assistant specialized in providing accurate, helpful answers based on StackOverflow content.

**CRITICAL**: Your answers must be FAITHFUL to the provided context.

Guidelines:
1. **Primary Source**: Base your answer on the retrieved context
   - Directly use information from the context
   - Synthesize multiple context pieces if available
   - Maintain accuracy to the source material

2. **Context Insufficient**: If context doesn't fully answer the question
   - Answer what you can from context
   - Clearly state what's missing
   - Suggest related information if helpful

3. **Answer Format**:
   - Be clear and concise
   - Include code examples from context when relevant
   - Use proper formatting (markdown, code blocks)
   - Cite or reference when using specific examples

4. **Never**:
   - Contradict the provided context
   - Make up information not in context
   - Claim context says something it doesn't

If no context is provided, politely indicate that."""),
        ("human", """**Query Analysis**:
{query_analysis}

**Retrieved Context** (from StackOverflow):
{context}

**User Question**: {question}

**Your Answer** (based primarily on the context above):""")
    ])
    
    # Prompt for direct answering (without retrieval)
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful software engineering assistant.

You are answering without specific retrieved context. Use your general knowledge to provide:
- Clear, accurate information
- Helpful guidance
- Honest acknowledgment of limitations
- Suggestions for where to find more information

Be concise and professional."""),
        ("human", """**Query Analysis**:
{query_analysis}

**User Question**: {question}

**Your Answer**:""")
    ])

    context_chain = context_prompt | llm
    direct_chain = direct_prompt | llm
    
    def generator(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer based on available context.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated answer
        """
        question = state["question"]
        context = state.get("context", "")
        query_analysis = state.get("query_analysis", "No analysis available")
        needs_retrieval = state.get("needs_retrieval", False)
        
        has_context = context and context.strip() and context != "No information available."
        
        logger.info(f"[Generator] Generating answer (has_context={has_context})")
        
        try:
            if has_context:
                # Use context-based generation
                logger.info(f"[Generator] Using {len(context)} chars of context")
                response = context_chain.invoke({
                    "question": question,
                    "context": context,
                    "query_analysis": query_analysis
                })
            else:
                # Use direct generation
                logger.info("[Generator] No context available, using direct generation")
                response = direct_chain.invoke({
                    "question": question,
                    "query_analysis": query_analysis
                })
            
            final_answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            logger.info("[Generator] Answer generated successfully")
            
            return {
                **state,
                "answer": final_answer,
                "generation_mode": "context" if has_context else "direct"
            }
            
        except Exception as e:
            logger.error(f"[Generator] Generation failed: {str(e)}")
            fallback_answer = (
                "I apologize, but I encountered an error while generating a response. "
                "Please try rephrasing your question or try again later."
            )
            return {
                **state,
                "answer": fallback_answer,
                "error": f"Generator failed: {str(e)}",
                "generation_mode": "error"
            }
    
    return generator
