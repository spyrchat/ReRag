from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_qdrant import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from typing import List, Optional


class QdrantDenseRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: Embeddings,
        top_k: int = 5,
        vector_name: str = "",  # MUST match what was used on insertion
        filters: Optional[dict] = None,
    ):
        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model,
            vector_name=vector_name,
        )
        self.top_k = top_k
        self.filters = filters

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(
            query=query,
            k=self.top_k,
            filter=self.filters
        )
