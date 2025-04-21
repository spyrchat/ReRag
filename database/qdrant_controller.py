from langchain.schema import Document
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.embeddings.base import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import PointStruct, VectorStruct
from uuid import uuid4
from qdrant_client.models import VectorParams, Distance, NamedVectorStruct, VectorParams
from embedding.utils import batchify
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantVectorDB:
    def __init__(self, vector_name: str = "dense"):
        load_dotenv()
        self.host = os.getenv("QDRANT_HOST")
        self.port = int(os.getenv("QDRANT_PORT"))
        self.api_key = os.getenv("QDRANT_API_KEY", None)
        self.collection_name = os.getenv("QDRANT_COLLECTION")
        self.vector_name = vector_name

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, vector_size: int, vector_name: str = "dense"):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    vector_name: VectorParams(
                        size=vector_size, distance=Distance.COSINE)
                }),

            logger.info(
                f"Collection '{self.collection_name}' created with vector '{vector_name}'.")
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
        batch_size: int = 256
    ):
        assert len(documents) == len(
            vectors), "Mismatch between documents and vectors"
        logger.info(
            f"Inserting {len(documents)} documents in batches of {batch_size} using vector name '{self.vector_name}'.")

        for i, (doc_batch, vec_batch) in enumerate(zip(
            batchify(documents, batch_size),
            batchify(vectors, batch_size)
        )):
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={self.vector_name: vector},
                    payload={**doc.metadata, "text": doc.page_content}
                )
                for doc, vector in zip(doc_batch, vec_batch)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

            logger.info(f"Inserted batch {i + 1} with {len(points)} points.")
