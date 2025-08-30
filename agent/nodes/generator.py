from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger

from langchain_openai import ChatOpenAI

logger = get_logger("generator")

generator_prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Use the given context to answer the user's question.

Context:
{context}

Question:
{question}

Answer in clear, professional natural language.
"""
)


def make_generator(llm):
    """
    Returns a generator node function with the provided LLM injected.
    """
    def generator(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]

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
            prompt = generator_prompt.format(
                context=context, question=question)
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
