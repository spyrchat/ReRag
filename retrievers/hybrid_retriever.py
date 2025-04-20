from typing import List
from langchain.schema import Document
from langchain.retrievers import BaseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        alpha: float = 0.5
    ):
        """
        Args:
            dense_retriever (BaseRetriever): Vector DB retriever (e.g., Qdrant).
            sparse_retriever (BaseRetriever): BM25 retriever.
            alpha (float): Fusion weight [0, 1]. 1 = all dense, 0 = all sparse.
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Get top-k from both
        dense_docs = self.dense.get_relevant_documents(query)
        sparse_docs = self.sparse.get_relevant_documents(query)

        # Merge by unique content
        all_docs = {doc.page_content: doc for doc in dense_docs}
        for doc in sparse_docs:
            if doc.page_content not in all_docs:
                all_docs[doc.page_content] = doc

        return list(all_docs.values())

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
