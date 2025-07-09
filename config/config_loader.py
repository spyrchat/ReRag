import yaml
from pathlib import Path


def load_config(config_path="config.yml"):
    """
    Load YAML configuration for the pipeline.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
