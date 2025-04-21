import os
import uuid
import logging
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantVectorDB:
    def __init__(self, vector_name: str = "dense"):
        """
        Initialize the QdrantVectorDB using environment variables for connection
        and collection parameters.

        Args:
            vector_name (str): The name of the vector field to use in Qdrant. Default is "dense".
        """
        load_dotenv(override=True)
        self.host: str = os.getenv("QDRANT_HOST")
        self.port: int = int(os.getenv("QDRANT_PORT"))
        self.api_key: str | None = os.getenv("QDRANT_API_KEY", None)
        self.collection_name: str = os.getenv("QDRANT_COLLECTION")
        self.vector_name: str = vector_name
        print(f"Qdrant collection: {self.collection_name}")

        self.client: QdrantClient = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, vector_size: int) -> None:
        """
        Initialize a Qdrant collection with the specified vector size.
        If the collection already exists, it is not modified.

        Args:
            vector_size (int): Dimensionality of the vectors to store.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                }
            )
            logger.info(
                f"Collection '{self.collection_name}' created with vector '{self.vector_name}'.")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def get_client(self) -> QdrantClient:
        """
        Get the Qdrant client instance.

        Returns:
            QdrantClient: The initialized Qdrant client.
        """
        return self.client

    def get_collection_name(self) -> str:
        """
        Get the name of the current Qdrant collection.

        Returns:
            str: The collection name.
        """
        return self.collection_name

    def insert_documents(
        self,
        documents: List[Document],
        embedding: Embeddings
    ) -> None:
        """
        Insert documents into the Qdrant collection after embedding them using the provided LangChain embedder.

        Args:
            documents (List[Document]): List of LangChain documents to embed and insert.
            embedding (Embeddings): LangChain-compatible embedding model.
        """
        vectorstore = self.as_langchain_vectorstore(embedding)
        ids = [str(uuid.uuid4()) for _ in documents]
        vectorstore.add_documents(documents=documents, ids=ids)
        logger.info(
            f"Inserted {len(documents)} documents into '{self.collection_name}'.")

    def as_langchain_vectorstore(self, embedding: Embeddings) -> QdrantVectorStore:
        """
        Return a LangChain-compatible Qdrant vector store instance.

        Args:
            embedding (Embeddings): A LangChain embedding model to use with the vector store.

        Returns:
            QdrantVectorStore: A LangChain-wrapped Qdrant vector store object.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embedding,
            vector_name=self.vector_name,
        )
