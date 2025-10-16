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

        # Try to extract JSON from code blocks if present
        if "```json" in response:
            # Extract content between ```json and ```
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            # Extract content between ``` and ```
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()

        # Try to parse the JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to find JSON object in the response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            # If all else fails, raise the original error with helpful context
            raise json.JSONDecodeError(
                f"Failed to parse LLM response as JSON. Response: {response[:200]}...",
                response,
                e.pos
            )
