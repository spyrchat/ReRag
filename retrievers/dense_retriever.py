from qdrant_client import QdrantClient
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from typing import List
from .base import BaseRetriever


class QdrantDenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedding: Embeddings,
        client: QdrantClient,
        collection_name: str,
        top_k: int = 10,
    ):
        self.top_k = top_k
        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense"
        )

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k or self.top_k)

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retrieve(query)
