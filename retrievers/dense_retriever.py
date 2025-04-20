from typing import List, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient


class QdrantDenseRetriever:
    """
    A modern dense retriever for Qdrant using LangChain's updated QdrantVectorStore.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: Embeddings,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ):
        self.top_k = top_k
        self.filters = filters

        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
            vector_name="",
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.top_k, filter=self.filters)
