from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


class HuggingFaceEmbedder(HuggingFaceEmbeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
