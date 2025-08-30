import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    Load unified YAML configuration for the pipeline.

    Args:
        config_path: Path to the main configuration file

    Returns:
        Complete configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config {config_path}: {e}")


def get_retriever_config(config: Dict[str, Any], retriever_type: str) -> Dict[str, Any]:
    """
    Extract retriever-specific configuration from unified config.

    Args:
        config: Main configuration dictionary
        retriever_type: Type of retriever (dense, sparse, hybrid, semantic)

    Returns:
        Retriever-specific configuration

    Raises:
        ValueError: If retriever type not found
    """
    retrievers_config = config.get("retrievers", {})

    if retriever_type not in retrievers_config:
        available_types = list(retrievers_config.keys())
        raise ValueError(
            f"Retriever type '{retriever_type}' not found. Available: {available_types}")

    retriever_config = retrievers_config[retriever_type].copy()

    # Merge with global settings
    if "embedding" in config and "embedding" not in retriever_config:
        retriever_config["embedding"] = config["embedding"]

    if "qdrant" in config and "qdrant" not in retriever_config:
        retriever_config["qdrant"] = config["qdrant"]

    return retriever_config


def get_benchmark_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract benchmark configuration with defaults.

    Args:
        config: Main configuration dictionary

    Returns:
        Benchmark configuration
    """
    benchmark_config = config.get("benchmark", {})

    # Set defaults
    defaults = {
        "evaluation": {
            "k_values": [1, 5, 10, 20],
            "metrics": ["precision", "recall", "f1", "mrr", "ndcg"]
        },
        "retrieval": {
            "strategy": "hybrid",
            "top_k": 20,
            "search_params": {
                "score_threshold": 0.0
            }
        }
    }

    # Merge defaults with provided config
    for key, default_value in defaults.items():
        if key not in benchmark_config:
            benchmark_config[key] = default_value
        elif isinstance(default_value, dict):
            for sub_key, sub_default in default_value.items():
                if sub_key not in benchmark_config[key]:
                    benchmark_config[key][sub_key] = sub_default

    return benchmark_config


def get_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract retrieval pipeline configuration.

    Args:
        config: Main configuration dictionary

    Returns:
        Pipeline configuration
    """
    pipeline_config = config.get("retrieval_pipeline", {})

    # Set defaults
    if "default_retriever" not in pipeline_config:
        pipeline_config["default_retriever"] = "hybrid"

    if "components" not in pipeline_config:
        pipeline_config["components"] = [
            {"type": "retriever", "config": {
                "retriever_type": pipeline_config["default_retriever"]}}
        ]

    return pipeline_config


def load_config_with_overrides(config_path: str = "config.yml",
                               overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration with optional overrides.

    Args:
        config_path: Path to the main configuration file
        overrides: Optional dictionary of configuration overrides

    Returns:
        Configuration with overrides applied
    """
    config = load_config(config_path)

    if overrides:
        # Deep merge overrides
        config = _deep_merge(config, overrides)
        logger.info("Applied configuration overrides")

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if (key in result and
            isinstance(result[key], dict) and
                isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
