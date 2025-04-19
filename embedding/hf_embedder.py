from langchain.embeddings import HuggingFaceEmbeddings
from embedding.base_embedder import BaseEmbedder
from embedding.utils import batchify


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self.batch_size = batch_size

    def embed_texts(self, texts):
        embeddings = []
        for batch in batchify(texts, self.batch_size):
            embeddings.extend(self.model.embed_documents(batch))
        return embeddings
