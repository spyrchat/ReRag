from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger

from langchain_openai import ChatOpenAI

logger = get_logger("generator")

# Main prompt: Strict grounding to context (recommended for accuracy)
generator_prompt = PromptTemplate.from_template(
    """You are an expert technical assistant specializing in software development and programming.

Your task is to answer the user's question using ONLY the information provided in the context below. The context contains relevant excerpts from Stack Overflow posts and technical documentation.

Query Analysis:
{query_analysis}

Context:
{context}

Question:
{question}

Instructions:
1. Use the query analysis above to understand what the user needs and follow the reasoning steps
2. Answer ONLY based on the provided context - do not use external knowledge
3. Address the key concepts identified in the analysis
4. If the context contains code examples, include them in your answer
5. If the context discusses multiple approaches, mention the key differences
6. If the context doesn't fully answer the question, acknowledge what's missing
7. Be concise but complete - aim for clarity over verbosity
8. Use technical terminology appropriately
9. If the context is insufficient or irrelevant, say: "Based on the provided context, I cannot fully answer this question."

Provide your answer in clear, professional language:"""
)

# Alternative: More conversational prompt (allows some inference)
generator_prompt_conversational = PromptTemplate.from_template(
    """You are a helpful programming assistant with expertise in software development.

Use the following context from Stack Overflow posts to answer the user's technical question. Synthesize the information and provide practical guidance.

Query Analysis:
{query_analysis}

Context:
{context}

Question:
{question}

Provide a clear, helpful answer that:
- Follows the reasoning steps from the query analysis
- Addresses the key concepts identified
- Includes relevant code examples from the context
- Explains the reasoning behind solutions
- Mentions any important caveats or considerations

Answer:"""
)

# Alternative: Citation-focused prompt (for research/verification use cases)
generator_prompt_with_citations = PromptTemplate.from_template(
    """You are a technical documentation assistant. Answer the question using information from the provided Stack Overflow excerpts.

Question:
{question}

Query Analysis:
{query_analysis}

Available Context:
{context}

Instructions:
- Use the query analysis to understand the user's needs and key concepts
- Base your answer strictly on the provided context
- If multiple solutions exist, compare their trade-offs
- Indicate when information is incomplete
- Reference specific parts of the context when relevant (e.g., "According to the provided answer...")
- Structure your response clearly with bullet points or numbered lists when appropriate

Answer:"""
)


def make_generator(llm, prompt_style: str = "strict"):
    """
    Returns a generator node function with the provided LLM injected.

    Args:
        llm: Language model instance
        prompt_style: One of "strict", "conversational", or "citations" (default: "strict")
    """
    # Select prompt based on style
    prompts = {
        "strict": generator_prompt,
        "conversational": generator_prompt_conversational,
        "citations": generator_prompt_with_citations
    }

    selected_prompt = prompts.get(prompt_style, generator_prompt)
    logger.info(f"[Generator] Using prompt style: {prompt_style}")

    def generator(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        query_analysis = state.get("query_analysis", "No analysis available")

        # Log query analysis for debugging
        logger.info(f"[Generator] Query analysis received: {query_analysis[:200]}..." if len(query_analysis) > 200 else f"[Generator] Query analysis: {query_analysis}")

        if "context" in state:
            context = state["context"]
            logger.info("[Generator] Generating from retrieved context.")
        elif "answer" in state:
            context = state["answer"]
            logger.info("[Generator] Generating from existing answer.")
        else:
            context = "No information available."
            logger.warning("[Generator] No context available.")

        try:
            prompt = selected_prompt.format(
                context=context,
                question=question,
                query_analysis=query_analysis)
            response = llm.invoke(prompt)
            final_answer = response.content.strip()

            logger.info("[Generator] Answer generated.")
            return {
                **state,
                "answer": final_answer
            }
        except Exception as e:
            logger.error(f"[Generator] LLM invocation failed: {str(e)}")
            return {
                **state,
                "answer": "I'm sorry, I couldn't generate an answer due to an internal error."
            }
    return generator
