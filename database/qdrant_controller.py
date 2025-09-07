import os
import uuid
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from logs.utils.logger import get_logger
from .base import BaseVectorDB

logger = get_logger(__name__)


class QdrantVectorDB(BaseVectorDB):
    def __init__(self, strategy: str = "dense", config: Optional[Dict[str, Any]] = None):
        load_dotenv(override=True)
        self.strategy = strategy.lower()

        # Use config if provided, otherwise fall back to environment variables
        if config and "qdrant" in config:
            qdrant_config = config["qdrant"]
            self.host = qdrant_config.get(
                "host", os.getenv("QDRANT_HOST", "localhost"))
            self.port = int(qdrant_config.get(
                "port", os.getenv("QDRANT_PORT", "6333")))
            self.api_key = qdrant_config.get(
                "api_key", os.getenv("QDRANT_API_KEY"))
            self.collection_name = qdrant_config.get("collection", qdrant_config.get(
                "collection_name", os.getenv("QDRANT_COLLECTION")))
            self.dense_vector_name = qdrant_config.get(
                "dense_vector_name", os.getenv("DENSE_VECTOR_NAME", "dense"))
            self.sparse_vector_name = qdrant_config.get(
                "sparse_vector_name", os.getenv("SPARSE_VECTOR_NAME", "sparse"))
        else:
            # Fall back to environment variables
            self.host = os.getenv("QDRANT_HOST")
            self.port = int(os.getenv("QDRANT_PORT"))
            self.api_key = os.getenv("QDRANT_API_KEY")
            self.collection_name = os.getenv("QDRANT_COLLECTION")
            self.dense_vector_name = os.getenv("DENSE_VECTOR_NAME", "dense")
            self.sparse_vector_name = os.getenv("SPARSE_VECTOR_NAME", "sparse")

        logger.info(f"Qdrant collection: {self.collection_name}")
        logger.info(f"Dense vector: {self.dense_vector_name}")
        logger.info(f"Sparse vector: {self.sparse_vector_name}")

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, dense_vector_size: int) -> None:
        """
        Initialize (or re-create) a Qdrant collection for dense and sparse vectors.
        Deletes existing collection if already present.
        Args:
            dense_vector_size (int): The dimensionality of the dense vector.
        """
        if self.client.collection_exists(self.collection_name):
            logger.info(
                f"Collection '{self.collection_name}' already exists. Recreating..."
            )
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: VectorParams(
                    size=dense_vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
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
        Return the initialized Qdrant client instance.
        """
        return self.client

    def get_collection_name(self) -> str:
        """
        Return the name of the current Qdrant collection.
        """
        return self.collection_name

    def insert_documents(
        self,
        documents: List[Document],
        dense_embedder: Optional[Embeddings] = None,
        sparse_embedder: Optional[Embeddings] = None,
    ) -> None:
        """
        Insert a list of LangChain Documents into the configured Qdrant collection,
        initializing the collection if needed (using dense_embedder for dimension).
        Args:
            documents (List[Document]): The documents to insert.
            dense_embedder (Optional[Embeddings]): Embedder for dense vectors.
            sparse_embedder (Optional[Embeddings]): Embedder for sparse vectors.
        """
        # Initialize collection only if needed and if dense_embedder is provided
        if not self.client.collection_exists(self.collection_name) and dense_embedder:
            sample_embedding = dense_embedder.embed_query("test")
            dense_dim = len(sample_embedding)
            self.init_collection(dense_vector_size=dense_dim)

        vectorstore = self.as_langchain_vectorstore(
            dense_embedding=dense_embedder,
            sparse_embedding=sparse_embedder,
        )

        # Use external_id from metadata if available, otherwise generate UUID
        ids = []
        processed_documents = []
        for doc in documents:
            external_id = doc.metadata.get("external_id")
            if external_id:
                ids.append(str(external_id))
                # Ensure external_id is preserved in the document metadata
                # Create a copy of the document with external_id explicitly in metadata
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "external_id": str(external_id)}
                )
                processed_documents.append(doc_copy)
            else:
                generated_id = str(uuid.uuid4())
                ids.append(generated_id)
                # Add the generated ID to metadata as well
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "external_id": generated_id}
                )
                processed_documents.append(doc_copy)

        vectorstore.add_documents(documents=processed_documents, ids=ids)

        logger.info(
            f"Inserted {len(documents)} documents into '{self.collection_name}' "
            f"({'dense' if dense_embedder else ''}{' + sparse' if sparse_embedder else ''})."
        )

    def as_langchain_vectorstore(
        self,
        dense_embedding: Optional[Embeddings] = None,
        sparse_embedding: Optional[Embeddings] = None,
        strategy: Optional[str] = None
    ) -> QdrantVectorStore:
        """
        Returns a LangChain-compatible QdrantVectorStore based on the selected retrieval strategy.
        """
        strategy = (strategy or self.strategy or "dense").lower()

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
            logger.error(f"Invalid EMBEDDING_STRATEGY: {strategy}")
            raise ValueError(f"Invalid EMBEDDING_STRATEGY: {strategy}")
