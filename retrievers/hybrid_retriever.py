from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from .base import BaseRetriever


class QdrantHybridRetriever(BaseRetriever):
    """
    A retriever that uses Qdrant's hybrid (dense + sparse) search.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_embedding: Embeddings,
        sparse_embedding: Embeddings,
        *,
        top_k: int = 5,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
    ):
        """
        Args:
            client:           an initialized QdrantClient
            collection_name:  the Qdrant collection to query
            dense_embedding:  the Embeddings instance for dense vectors
            sparse_embedding: the Embeddings instance for sparse vectors
            top_k:            number of hits to return by default
            dense_vector_name: name of the dense vector field in Qdrant
            sparse_vector_name: name of the sparse vector field in Qdrant
        """
        self.top_k = top_k
        self.vs = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense_embedding,
            vector_name=dense_vector_name,
            sparse_embedding=sparse_embedding,
            sparse_vector_name=sparse_vector_name,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def retrieve(
        self, query: str, *, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Returns top-k (Document, score) pairs according to hybrid search.
        """
        return self.vs.similarity_search_with_score(
            query,
            k=k or self.top_k
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Returns just the Documents (no scores).
        """
        hits = self.retrieve(query)
        return [doc for doc, _ in hits]
