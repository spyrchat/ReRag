from qdrant_client.http.models import VectorParams, SparseVectorParams
from qdrant_client import models as qmodels
from qdrant_client.http.models import Distance
import os
import uuid
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from .base import BaseVectorDB
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantVectorDB(BaseVectorDB):
    def __init__(self):
        load_dotenv(override=True)
        self.host: str = os.getenv("QDRANT_HOST")
        self.port: int = int(os.getenv("QDRANT_PORT"))
        self.api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
        self.collection_name: str = os.getenv("QDRANT_COLLECTION")
        self.dense_vector_name: str = os.getenv("DENSE_VECTOR_NAME", "dense")
        self.sparse_vector_name: str = os.getenv(
            "SPARSE_VECTOR_NAME", "sparse")

        logger.info(f"Qdrant collection: {self.collection_name}")
        logger.info(f"Dense vector: {self.dense_vector_name}")
        logger.info(f"Sparse vector: {self.sparse_vector_name}")

        self.client: QdrantClient = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, dense_vector_size: int) -> None:
        """
        Create (or recreate) the collection for dense and sparse vectors.
        """
        if self.client.collection_exists(self.collection_name):
            logger.info(
                f"Collection '{self.collection_name}' already exists. Recreating..."
            )
            self.client.delete_collection(self.collection_name)

        # Create with separate configs for dense & sparse
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                # only your dense side here
                self.dense_vector_name: VectorParams(
                    size=dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                # only your sparse side here, using SparseVectorParams
                self.sparse_vector_name: SparseVectorParams(
                    index=qmodels.SparseIndexParams(on_disk=False)
                )
            },
        )

        logger.info(
            f"Collection '{self.collection_name}' created with "
            f"dense vector '{self.dense_vector_name}' and sparse vector "
            f"'{self.sparse_vector_name}'."
        )

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
        vectorstore = self.as_langchain_vectorstore(
            dense_embedding=dense_embedder,
            sparse_embedding=sparse_embedder,
        )

        ids = [str(uuid.uuid4()) for _ in documents]
        vectorstore.add_documents(documents=documents, ids=ids)

        logger.info(
            f"Inserted {len(documents)} documents into '{self.collection_name}' "
            f"({'dense' if dense_embedder else ''}{' + sparse' if sparse_embedder else ''})."
        )

    def as_langchain_vectorstore(
        self,
        dense_embedding: Optional[Embeddings] = None,
        sparse_embedding: Optional[Embeddings] = None,
    ) -> QdrantVectorStore:
        strategy = os.getenv("EMBEDDING_STRATEGY", "dense").lower()

        if strategy == "dense":
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=dense_embedding,
                vector_name=self.dense_vector_name,
                sparse_embedding=None,
                sparse_vector_name=self.sparse_vector_name,
                retrieval_mode=RetrievalMode.DENSE,
            )

        elif strategy == "sparse":
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                sparse_embedding=sparse_embedding,
                sparse_vector_name=self.sparse_vector_name,
                retrieval_mode=RetrievalMode.SPARSE,
            )

        elif strategy == "hybrid":
            return QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=dense_embedding,
                vector_name=self.dense_vector_name,
                sparse_embedding=sparse_embedding,
                sparse_vector_name=self.sparse_vector_name,
                retrieval_mode=RetrievalMode.HYBRID,
            )

        else:
            raise ValueError(f"Invalid EMBEDDING_STRATEGY: {strategy}")
