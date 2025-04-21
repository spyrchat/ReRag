from qdrant_client import QdrantClient
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from typing import List, Tuple
from .base import BaseRetriever


class QdrantDenseRetriever:
    def __init__(
        self,
        embedding: Embeddings,
        vectorstore: QdrantVectorStore,
        top_k: int = 5,
    ):
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=k or self.top_k)

    def get_relevant_documents(self, query: str) -> List[Tuple[Document, float]]:
        return self.retrieve(query)
