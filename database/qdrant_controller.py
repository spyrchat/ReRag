import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from database.base import BaseVectorDB
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QdrantVectorDB(BaseVectorDB):
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

    def insert_documents(self, documents: List[Document], embedding_function: Embeddings):
        """Embed and insert documents into the Qdrant collection."""
        LangchainQdrant.from_documents(
            documents=documents,
            embedding=embedding_function,
            client=self.client,
            collection_name=self.collection_name,
        )
        logger.info(
            f"Inserted {len(documents)} documents into '{self.collection_name}'.")

    def insert_embeddings(self, documents: List[Document], vectors: List[List[float]]):
        if len(documents) != len(vectors):
            raise ValueError("Number of documents and embeddings must match")

        from uuid import uuid4

        payloads = [
            {
                **doc.metadata,
                "text": doc.page_content
            }
            for doc in documents
        ]
        ids = [str(uuid4()) for _ in documents]

        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payloads,
            ids=ids,
            batch_size=64
        )

        logger.info(
            f"Inserted {len(documents)} embeddings into '{self.collection_name}'.")
