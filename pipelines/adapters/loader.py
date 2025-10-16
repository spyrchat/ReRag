"""
Dynamic adapter loader - load adapters from config without code changes.
Supports scaling to new adapters by simply adding them to config files.
"""
import importlib
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AdapterLoader:
    """
    Dynamically load dataset adapters from configuration.

    This allows adding new adapters without modifying code - just update config.
    """

    # Built-in adapter shortcuts for convenience (optional)
    ADAPTER_SHORTCUTS = {
        "stackoverflow": "pipelines.adapters.stackoverflow.StackOverflowAdapter"
    }

    @classmethod
    def load_adapter(
        cls,
        adapter_spec: str,
        dataset_path: str,
        version: str = "1.0.0",
        **kwargs
    ) -> Any:
        """
        Load an adapter dynamically from a specification string.

        Args:
            adapter_spec: Either a shortcut name or full module path
                Examples:
                - "stackoverflow" (shortcut)
                - "pipelines.adapters.stackoverflow.StackOverflowAdapter" (full path)
                - "my_custom_package.adapters.MyAdapter" (custom adapter)
            dataset_path: Path to dataset files
            version: Dataset version
            **kwargs: Additional arguments to pass to adapter constructor

        Returns:
            Instantiated adapter object

        Raises:
            ValueError: If adapter cannot be loaded
        """
        # Resolve shortcuts
        if adapter_spec in cls.ADAPTER_SHORTCUTS:
            full_path = cls.ADAPTER_SHORTCUTS[adapter_spec]
            logger.info(f"Resolved shortcut '{adapter_spec}' -> '{full_path}'")
        else:
            full_path = adapter_spec

        # Parse module path and class name
        try:
            module_path, class_name = full_path.rsplit(".", 1)
        except ValueError:
            raise ValueError(
                f"Invalid adapter specification: '{adapter_spec}'. "
                f"Expected format: 'module.path.ClassName' or a valid shortcut."
            )

        # Import module and get class
        try:
            logger.info(f"Loading adapter: {module_path}.{class_name}")
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Could not import adapter module '{module_path}': {e}\n"
                f"Available shortcuts: {list(cls.ADAPTER_SHORTCUTS.keys())}"
            )
        except AttributeError as e:
            raise ValueError(
                f"Module '{module_path}' does not have class '{class_name}': {e}"
            )

        # Instantiate adapter
        try:
            adapter = adapter_class(dataset_path, version, **kwargs)

            # Log success - use source_name if available, otherwise use name or class_name
            adapter_name = getattr(adapter, 'source_name', None) or getattr(
                adapter, 'name', class_name)
            adapter_version = getattr(adapter, 'version', version)
            logger.info(
                f"Successfully loaded adapter: {adapter_name} v{adapter_version}"
            )
            return adapter
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate adapter {class_name}: {e}\n"
                f"Check that the adapter constructor accepts (dataset_path, version, **kwargs)"
            )

    @classmethod
    def load_from_config(
        cls,
        config: Dict[str, Any],
        dataset_path: Optional[str] = None
    ) -> Any:
        """
        Load adapter from a configuration dictionary.

        Args:
            config: Configuration dict with 'dataset' section
                Example:
                    dataset:
                      adapter: "stackoverflow"  # or full path
                      path: "/path/to/data"     # optional if dataset_path provided
                      version: "1.0.0"          # optional
                      adapter_kwargs:           # optional
                        custom_param: "value"
            dataset_path: Override dataset path from config (optional)

        Returns:
            Instantiated adapter object

        """
        if "dataset" not in config:
            raise ValueError(
                "Config must contain 'dataset' section with adapter specification"
            )

        dataset_config = config["dataset"]

        # Get adapter specification
        adapter_spec = dataset_config.get("adapter")
        if not adapter_spec:
            raise ValueError(
                "Config must specify 'dataset.adapter' (e.g., 'stackoverflow' or full class path)"
            )

        # Get dataset path
        path = dataset_path or dataset_config.get("path")
        if not path:
            raise ValueError(
                "Dataset path must be specified in config or as function argument"
            )

        # Get version
        version = dataset_config.get("version", "1.0.0")

        # Get additional adapter kwargs
        adapter_kwargs = dataset_config.get("adapter_kwargs", {})

        # Load adapter
        return cls.load_adapter(
            adapter_spec=adapter_spec,
            dataset_path=path,
            version=version,
            **adapter_kwargs
        )

    @classmethod
    def register_shortcut(cls, name: str, full_path: str):
        """
        Register a custom adapter shortcut.

        This allows projects to define their own shortcuts without modifying this file.

        Args:
            name: Short name for the adapter
            full_path: Full module path (e.g., "my.module.MyAdapter")

        Example:
            >>> AdapterLoader.register_shortcut(
            ...     "my_adapter",
            ...     "my_project.adapters.MyCustomAdapter"
            ... )
            >>> adapter = AdapterLoader.load_adapter("my_adapter", "/path/to/data")
        """
        cls.ADAPTER_SHORTCUTS[name] = full_path
        logger.info(f"Registered adapter shortcut: {name} -> {full_path}")

    @classmethod
    def list_shortcuts(cls) -> Dict[str, str]:
        """
        Get all registered adapter shortcuts.

        Returns:
            Dictionary mapping shortcut names to full class paths
        """
        return cls.ADAPTER_SHORTCUTS.copy()


# Convenience function for simple use cases
def load_adapter(adapter_spec: str, dataset_path: str, version: str = "1.0.0", **kwargs) -> Any:
    """
    Convenience function to load an adapter.

    See AdapterLoader.load_adapter for full documentation.
    """
    return AdapterLoader.load_adapter(adapter_spec, dataset_path, version, **kwargs)
