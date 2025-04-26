from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query (str): The input query string.
            k (int, optional): Number of documents to retrieve. Defaults to 5.

        Returns:
            List[Document]: List of retrieved documents.
        """
        pass

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Compatibility method for LangChain retriever interface.

        Args:
            query (str): The input query string.

        Returns:
            List[Document]: List of retrieved documents.
        """
        return self.retrieve(query=query)
