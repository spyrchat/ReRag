from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

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
    elif "answer" in state:
        context = state["answer"]
    else:
        context = "No information available."

    prompt = generator_prompt.format(context=context, question=question)
    response = llm.invoke(prompt)
    final_answer = response.content.strip()

    return {
        **state,
        "answer": final_answer
    }
