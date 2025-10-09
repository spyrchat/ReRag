from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import json


class LLMJudge:
    """
    A unified interface for LLM-as-a-Judge evaluations using different LLM providers.

    This class abstracts over provider-specific LLM implementations (e.g., OpenAI, Anthropic, Cohere)
    and exposes a consistent method for evaluating prompts and returning structured JSON responses.

    Supported providers:
        - "openai": via langchain_openai
        - "anthropic": via langchain_anthropic
        - "cohere": via langchain_cohere

    Example usage:
        judge = LLMJudge(provider="openai", model_name="gpt-4.1")
        result = judge.evaluate(prompt)
    """

    def __init__(self, provider: str, model_name: str):
        """
        Initializes the LLMJudge with the selected provider and model.

        Args:
            provider (str): One of {"openai", "anthropic", "cohere"}.
            model_name (str): Model identifier as expected by the LangChain wrapper.

        Raises:
            ValueError: If the provider is not supported.
        """
        if provider == "openai":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        elif provider == "anthropic":
            self.llm = ChatAnthropic(model=model_name, temperature=0)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluates a prompt using the selected LLM and parses the response as JSON.

        Args:
            prompt (str): The full string prompt to send to the model.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary containing the evaluation result.

        Raises:
            json.JSONDecodeError: If the LLM response is not valid JSON.
        """
        response = self.llm.invoke(prompt).content.strip()
        return json.loads(response)
