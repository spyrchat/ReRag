from abc import ABC, abstractmethod
from typing import List, Protocol
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class BaseVectorDB(ABC):
    @abstractmethod
    def init_collection(self, vector_size: int) -> None:
        """
        Initialize the vector collection with the specified vector size.

        Args:
            vector_size (int): The dimensionality of the vectors to be stored.
        """
        pass

    @abstractmethod
    def get_client(self) -> object:
        """
        Return the underlying database client.

        Returns:
            object: The database-specific client instance.
        """
        pass

    @abstractmethod
    def get_collection_name(self) -> str:
        """
        Return the name of the vector collection.

        Returns:
            str: The name of the collection.
        """
        pass

    @abstractmethod
    def insert_documents(
        self,
        documents: List[Document],
        embedding_function: Embeddings
    ) -> None:
        """
        Insert documents into the vector store after embedding them.

        Args:
            documents (List[Document]): List of documents to insert.
            embedding_function (Embeddings): Embedding model to use for vector generation.
        """
        pass

    @abstractmethod
    def as_langchain_vectorstore(
        self,
        embedding_function: Embeddings
    ) -> object:
        """
        Return the vector store as a LangChain-compatible vector store.

        Args:
            embedding_function (Embeddings): Embedding model to use for vector generation.

        Returns:
            object: LangChain-compatible vector store instance.
        """
        pass
