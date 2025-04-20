from langchain.schema import Document
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.embeddings.base import Embeddings
from langchain_qdrant import QdrantVectorStore
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantVectorDB:
    def __init__(self):
        load_dotenv()
        self.host = os.getenv("QDRANT_HOST")
        self.port = int(os.getenv("QDRANT_PORT"))
        self.api_key = os.getenv("QDRANT_API_KEY", None)
        self.collection_name = os.getenv("QDRANT_COLLECTION")

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created.")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def get_client(self):
        return self.client

    def get_collection_name(self):
        return self.collection_name

    def insert_documents(
        self,
        documents: List[Document],
        embedding_function: Embeddings,
        vector_name: str = ""
    ):
        """Embed and insert documents into the Qdrant collection using LangChain integration."""
        QdrantVectorStore.from_documents(
            documents=documents,
            embeddings=embedding_function,
            client=self.client,
            collection_name=self.collection_name,
            vector_name=vector_name,
        )
        logger.info(
            f"Inserted {len(documents)} documents with embeddings into '{self.collection_name}' (vector: {vector_name}).")

    def insert_embeddings(
        self,
        documents: List[Document],
        vectors: List[List[float]],
        vector_name: str = "default"
    ):
        if len(documents) != len(vectors):
            raise ValueError("Number of documents and embeddings must match")

        payloads = []
        ids = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy()

            doc_id = metadata.get("doc_id")
            chunk_id = metadata.get("chunk_id", i)

            if doc_id is None:
                raise ValueError(
                    f"Missing 'doc_id' in metadata for document index {i}")

            metadata["chunk_id"] = chunk_id
            metadata["doc_id"] = doc_id
            metadata["text"] = doc.page_content

            payloads.append(metadata)
            ids.append(str(uuid.uuid4()))  # valid UUID for Qdrant

        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payloads,
            ids=ids,
            batch_size=64
        )

        logger.info(
            f"Inserted {len(documents)} vectors into '{self.collection_name}' under vector '{vector_name}' with UUID point IDs.")
