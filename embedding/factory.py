import os
from typing import Optional, Union

from embedding.hf_embedder import HuggingFaceEmbedder
from embedding.bedrock_embeddings import TitanEmbedder


def get_embedder(name: Optional[str] = None, **kwargs) -> Union[HuggingFaceEmbedder, TitanEmbedder]:
    """
    Factory function that returns a LangChain-compatible embedder
    based on the specified name or environment variable DENSE_EMBEDDER.

    Supported embedders:
        - "hf"     -> HuggingFace SentenceTransformers
        - "titan"  -> Amazon Titan via Bedrock

    Args:
        name (Optional[str]): Name of the embedder to use ("hf" or "titan").
                              If None, will default to value from DENSE_EMBEDDER env var, or "hf".
        **kwargs: Optional keyword arguments for embedder configuration.

    Returns:
        HuggingFaceEmbedder or TitanEmbedder: An initialized embedder instance.

    Raises:
        ValueError: If the embedder name is unsupported.
    """
    name = (name or os.getenv("DENSE_EMBEDDER", "hf")).lower()

    if name == "hf":
        model_name = kwargs.get("model_name") or os.getenv(
            "HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbedder(model_name=model_name)

    elif name == "titan":
        model = kwargs.get("model") or os.getenv(
            "TITAN_MODEL", "amazon.titan-embed-text-v2:0"
        )
        region = kwargs.get("region") or os.getenv("TITAN_REGION", "us-east-1")
        return TitanEmbedder(model=model, region=region)

    else:
        raise ValueError(
            f"Unsupported embedder: '{name}'. Valid options: 'hf', 'titan'.")
