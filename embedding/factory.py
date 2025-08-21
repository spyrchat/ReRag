from embedding.hf_embedder import HuggingFaceEmbedder
from embedding.bedrock_embeddings import TitanEmbedder
from embedding.sparse_embedder import SparseEmbedder
from langchain_qdrant import FastEmbedSparse


def get_embedder(cfg: dict):
    """
    Factory to return a LangChain-compatible embedder instance, based on YAML config.

    Args:
        cfg (dict): Embedder configuration dictionary.

    Returns:
        A LangChain-compatible embedder object.
    """
    provider = cfg.get("provider", "hf").strip().lower()

    if provider == "hf":
        model_name = cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = cfg.get("device", "cpu")
        return HuggingFaceEmbedder(model_name=model_name, device=device)

    elif provider == "titan":
        model = cfg.get("model_name", "amazon.titan-embed-text-v2:0")
        region = cfg.get("region", "us-east-1")
        return TitanEmbedder(model=model, region=region)

    elif provider == "fastembed":
        model_name = cfg.get("model_name", "BAAI/bge-small-en-v1.5")
        return FastEmbedSparse(model_name=model_name)

    elif provider == "sparse":
        model_name = cfg.get("model_name", "BAAI/bge-base-en")
        device = cfg.get("device", "cpu")
        return SparseEmbedder(model_name=model_name, device=device)

    else:
        raise ValueError(
            f"Unsupported embedder provider: '{provider}'. Supported: hf, titan, fastembed, sparse"
        )
