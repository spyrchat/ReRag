import os
from embedding.hf_embedder import HuggingFaceEmbedder
from embedding.bedrock_embeddings import TitanEmbedder
from embedding.sparse_embedder import SparseEmbedder
import dotenv

dotenv.load_dotenv()


def get_embedder(name: str = None, **kwargs):
    """
    Factory to return a LangChain-compatible embedder instance.

    Args:
        name (str, optional): Embedder name. If not provided, will fetch from ENV.
        kwargs: Additional model configuration.

    Returns:
        A LangChain-compatible embedder object.
    """

    name = (name or os.getenv("DENSE_EMBEDDER")
            or os.getenv("SPARSE_EMBEDDER")).strip().lower()

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

    elif name == "fastembed":
        model_name = kwargs.get("model_name") or os.getenv(
            "FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"
        )
        return SparseEmbedder(model_name=model_name)

    else:
        raise ValueError(
            f"Unsupported embedder name: '{name}'. Supported: hf, titan, fastembed")
