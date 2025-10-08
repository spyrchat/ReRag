"""
LLM Factory for creating language model instances.
Supports multiple providers: OpenAI, Ollama, etc.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_llm(config: Dict[str, Any]):
    """
    Create an LLM instance based on configuration.

    Args:
        config: LLM configuration dictionary with 'provider', 'model', 'temperature'

    Returns:
        LangChain LLM instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If required package is not installed

    Example config:
        llm:
          provider: openai
          model: gpt-4o-mini
          temperature: 0.0

        llm:
          provider: ollama
          model: llama3.1
          temperature: 0.0
          base_url: http://localhost:11434  # optional
    """
    provider = config.get("provider", "openai").lower()
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.0)

    logger.info(
        f"Creating LLM: provider={provider}, model={model}, temp={temperature}")

    if provider == "openai":
        return _create_openai_llm(model, temperature, config)
    elif provider == "ollama":
        return _create_ollama_llm(model, temperature, config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _create_openai_llm(model: str, temperature: float, config: Dict[str, Any]):
    """Create OpenAI LLM instance."""
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            # Optional: add other OpenAI-specific params from config
            max_tokens=config.get("max_tokens"),
            timeout=config.get("timeout"),
            max_retries=config.get("max_retries", 2)
        )

        logger.info(f"✓ OpenAI LLM created: {model}")
        return llm

    except ImportError:
        raise ImportError(
            "langchain-openai is not installed. "
            "Install with: pip install langchain-openai"
        )


def _create_ollama_llm(model: str, temperature: float, config: Dict[str, Any]):
    """Create Ollama LLM instance."""
    try:
        from langchain_ollama import ChatOllama

        # Get base URL (default to localhost)
        base_url = config.get("base_url", "http://localhost:11434")

        llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            # Optional Ollama-specific parameters
            num_ctx=config.get("num_ctx"),  # Context window size
            num_predict=config.get("num_predict"),  # Max tokens to generate
            repeat_penalty=config.get("repeat_penalty"),
            top_k=config.get("top_k"),
            top_p=config.get("top_p"),
        )

        logger.info(f"✓ Ollama LLM created: {model} at {base_url}")
        return llm

    except ImportError:
        raise ImportError(
            "langchain-ollama is not installed. "
            "Install with: pip install langchain-ollama"
        )


def get_available_providers() -> list:
    """
    Get list of available LLM providers based on installed packages.

    Returns:
        List of available provider names
    """
    providers = []

    try:
        import langchain_openai
        providers.append("openai")
    except ImportError:
        pass

    try:
        import langchain_ollama
        providers.append("ollama")
    except ImportError:
        pass

    return providers


def validate_llm_config(config: Dict[str, Any]) -> bool:
    """
    Validate LLM configuration.

    Args:
        config: LLM configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    if "model" not in config:
        logger.error("LLM config missing 'model' field")
        return False

    provider = config.get("provider", "openai").lower()
    if provider not in ["openai", "ollama"]:
        logger.error(f"Unsupported provider: {provider}")
        return False

    available = get_available_providers()
    if provider not in available:
        logger.error(
            f"Provider '{provider}' not available. "
            f"Available providers: {available}"
        )
        return False

    return True
