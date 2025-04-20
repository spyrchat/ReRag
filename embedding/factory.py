from embedding.hf_embedder import HuggingFaceEmbedder
from embedding.bedrock_embeddings import TitanEmbedder


def get_embedder(name: str, cached=False, **kwargs):
    if name == "hf":
        embedder = HuggingFaceEmbedder(**kwargs)
    elif name == "titan":
        embedder = TitanEmbedder(**kwargs)
    else:
        raise ValueError(f"Unsupported embedder: {name}")

    return embedder
