import os


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
        from embedding.embeddings import HuggingFaceEmbedder

        model_name = cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = cfg.get("device", "cpu")
        return HuggingFaceEmbedder(model_name=model_name, device=device)

    elif provider == "titan":
        from embedding.bedrock_embeddings import TitanEmbedder

        model = cfg.get("model_name", "amazon.titan-embed-text-v2:0")
        region = cfg.get("region", "us-east-1")
        return TitanEmbedder(model=model, region=region)

    elif provider == "fastembed":
        from langchain_qdrant import FastEmbedSparse

        model_name = cfg.get("model_name", "BAAI/bge-small-en-v1.5")
        return FastEmbedSparse(model_name=model_name)

    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model_name = cfg.get("model", "models/embedding-001")
        dimensions = cfg.get("dimensions") or cfg.get("output_dimensionality")
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Google embeddings")

        # Initialize with or without dimensions parameter
        embedding_kwargs = {
            "model": model_name,
            "google_api_key": api_key
        }

        if dimensions:
            embedding_kwargs["output_dimensionality"] = dimensions

        return GoogleGenerativeAIEmbeddings(**embedding_kwargs)

    elif provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        model_name = cfg.get("model", "voyage-3.5-lite")
        api_key = os.getenv("VOYAGE_API_KEY")

        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY environment variable is required for Voyage embeddings")

        # VoyageAI embeddings use native dimensions (1024 for voyage-3.5)
        # Dimension reduction can be handled via truncation if needed
        return VoyageAIEmbeddings(
            model=model_name,
            voyage_api_key=api_key
        )

    elif provider == "sparse":
        from embedding.sparse_embedder import SparseEmbedder

        # Support both 'model' and 'model_name' for consistency with other providers
        model_name = cfg.get("model") or cfg.get("model_name") or "Qdrant/bm25"
        device = cfg.get("device", "cpu")
        return SparseEmbedder(model_name=model_name, device=device)

    else:
        raise ValueError(
            f"Unsupported embedder provider: '{provider}'. Supported: hf, titan, fastembed, sparse, google, voyage"
        )
