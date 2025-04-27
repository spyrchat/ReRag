import os
import uuid
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_qdrant import RetrievalMode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantVectorDB:
    def __init__(self):
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
        self.dense_vector_name = os.getenv("DENSE_VECTOR_NAME", "dense")
        self.sparse_vector_name = os.getenv("SPARSE_VECTOR_NAME", "sparse")

        print(f"Qdrant collection: {self.collection_name}")

        self.client: QdrantClient = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, dense_vector_size: int, sparse_vector_size: int) -> None:
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
                    "dense": VectorParams(
                        size=dense_vector_size,
                        distance=Distance.COSINE
                    ),
                    "sparse": VectorParams(
                        size=sparse_vector_size,
                        distance=Distance.COSINE
                    )
                }
            )

            logger.info(
                f"Collection '{self.collection_name}' created with vector '{self.dense_vector_name}' and sparse vector '{self.sparse_vector_name}'.")
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
        dense_embedder: Optional[Embeddings] = None,
        sparse_embedder: Optional[Embeddings] = None,
    ) -> None:
        """
        Insert documents into the Qdrant collection using dense, sparse, or hybrid embedding.

        Args:
            documents (List[Document]): Documents to insert.
            dense_embedding (Optional[Embeddings]): Dense embedder, if available.
            sparse_embedding (Optional[Embeddings]): Sparse embedder, if available.
        """
        vectorstore = self.as_langchain_vectorstore(
            dense_embedding=dense_embedder,
            sparse_embedding=sparse_embedder,
        )

        ids = [str(uuid.uuid4()) for _ in documents]

        vectorstore.add_documents(documents=documents, ids=ids)

        logger.info(
            f"Inserted {len(documents)} documents into '{self.collection_name}' with "
            f"{'dense' if dense_embedder else ''}"
            f"{' + sparse' if sparse_embedder else ''} embeddings."
        )

    def as_langchain_vectorstore(
        self,
        dense_embedding: Optional[Embeddings] = None,
        sparse_embedding: Optional[Embeddings] = None
    ) -> QdrantVectorStore:
        """
        Return a LangChain-compatible Qdrant vectorstore.
        Supports dense, sparse, or hybrid retrieval based on environment variable.

        Args:
            dense_embedding (Optional[Embeddings]): Dense embedding model.
            sparse_embedding (Optional[Embeddings]): Sparse embedding model.

        Returns:
            QdrantVectorStore
        """
        retrieval_strategy = os.getenv("EMBEDDING_STRATEGY", "dense").lower()

        if retrieval_strategy == "dense":
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=dense_embedding,
                vector_name=self.dense_vector_name,
                retrieval_mode="dense",
            )

        elif retrieval_strategy == "sparse":
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                sparse_embedding=sparse_embedding,
                sparse_vector_name=self.sparse_vector_name,
                retrieval_mode="sparse",
            )

        elif retrieval_strategy == "hybrid":
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                vector_name=self.dense_vector_name,
                sparse_vector_name=f"{self.sparse_vector_name}_sparse",
                retrieval_mode="hybrid",
            )

        else:
            raise ValueError(
                f"Invalid EMBEDDING_STRATEGY: {retrieval_strategy}")
