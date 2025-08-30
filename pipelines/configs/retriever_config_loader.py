"""
Retriever configuration loader utility.
Loads and validates retriever configurations from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class RetrieverConfigLoader:
    """
    Loads and manages retriever configurations.
    """

    def __init__(self, config_dir: str = None):
        """
        Initialize the config loader.

        Args:
            config_dir: Directory containing retriever configs (defaults to pipelines/configs/retrievers)
        """
        if config_dir is None:
            # Default to project's retriever configs directory
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / "pipelines" / "configs" / "retrievers"
        else:
            self.config_dir = Path(config_dir)

        logger.info(
            f"RetrieverConfigLoader initialized with config_dir: {self.config_dir}")

    def load_config(self, retriever_type: str) -> Dict[str, Any]:
        """
        Load configuration for a specific retriever type.

        Args:
            retriever_type: Type of retriever (dense, sparse, hybrid, semantic)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = self.config_dir / f"{retriever_type}_retriever.yml"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}")

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            logger.info(f"Loaded configuration for {retriever_type} retriever")

            # Validate basic structure
            self._validate_config(config, retriever_type)

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config {config_file}: {e}")

    def _validate_config(self, config: Dict[str, Any], retriever_type: str):
        """Validate configuration structure."""
        required_keys = ['retriever']

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate retriever type matches
        configured_type = config['retriever'].get('type')
        if configured_type != retriever_type:
            logger.warning(
                f"Config type mismatch: expected {retriever_type}, got {configured_type}")

    def get_available_configs(self) -> List[str]:
        """
        Get list of available retriever configurations.

        Returns:
            List of available retriever types
        """
        if not self.config_dir.exists():
            return []

        configs = []
        for file_path in self.config_dir.glob("*_retriever.yml"):
            retriever_type = file_path.stem.replace("_retriever", "")
            configs.append(retriever_type)

        return sorted(configs)

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available retriever configurations.

        Returns:
            Dictionary mapping retriever type to configuration
        """
        configs = {}

        for retriever_type in self.get_available_configs():
            try:
                configs[retriever_type] = self.load_config(retriever_type)
            except Exception as e:
                logger.warning(f"Failed to load {retriever_type} config: {e}")

        return configs

    def merge_with_global_config(self, retriever_config: Dict[str, Any],
                                 global_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge retriever-specific config with global pipeline config.

        Args:
            retriever_config: Retriever-specific configuration
            global_config: Global pipeline configuration

        Returns:
            Merged configuration with global config as base and retriever config taking precedence
        """
        merged_config = global_config.copy()

        # Update with retriever-specific settings
        for key, value in retriever_config.items():
            if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                # Deep merge for nested dictionaries
                merged_config[key].update(value)
            else:
                # Direct assignment for non-dict values or new keys
                merged_config[key] = value

        return merged_config


# Convenience function for easy access
def load_retriever_config(retriever_type: str, config_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to load a retriever configuration.

    Args:
        retriever_type: Type of retriever to load config for
        config_dir: Optional custom config directory

    Returns:
        Configuration dictionary
    """
    loader = RetrieverConfigLoader(config_dir)
    return loader.load_config(retriever_type)
