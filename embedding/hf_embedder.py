from langchain_huggingface import HuggingFaceEmbeddings
from embedding.base_embedder import BaseEmbedder


class HuggingFaceEmbedder(HuggingFaceEmbeddings, BaseEmbedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        super().__init__(
            model_name=model_name,
            model_kwargs={"device": device},
        )
