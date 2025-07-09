from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger

logger = get_logger("generator")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

generator_prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Use the given context to answer the user's question.

Context:
{context}

Question:
{question}

Answer in clear, professional natural language.
"""
)


def generator(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]

    if "context" in state:
        context = state["context"]
        logger.info(f"[Generator] Generating from retrieved context.")
    elif "answer" in state:
        context = state["answer"]
        logger.info(f"[Generator] Generating from SQL result.")
    else:
        context = "No information available."
        logger.warning(f"[Generator] No context or SQL answer available.")

    try:
        prompt = generator_prompt.format(context=context, question=question)
        response = llm.invoke(prompt)
        final_answer = response.content.strip()

        logger.info(f"[Generator] Answer generated.")
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
