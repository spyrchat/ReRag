import os
from embedding.hf_embedder import HuggingFaceEmbedder
from embedding.bedrock_embeddings import TitanEmbedder
import dotenv


def get_embedder(name: str = None, **kwargs):
    """
    Returns a LangChain-compatible embedder.
    The embedder name is read from the .env if not passed explicitly.
    """

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
        raise ValueError(f"Unsupported embedder: {name}")
