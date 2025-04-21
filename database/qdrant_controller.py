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
from embedding.utils import batchify

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
        print(f"Qdrant collection: {self.collection_name}")

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key or None,
        )

    def init_collection(self, vector_size: int):
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

    def get_client(self):
        return self.client

    def get_collection_name(self):
        return self.collection_name

    def insert_documents(
        self,
        documents: List[Document],
        embedding: Embeddings
    ):
        """Embed and insert documents using LangChain's QdrantVectorStore."""
        vectorstore = self.as_langchain_vectorstore(embedding)
        ids = [str(uuid.uuid4()) for _ in documents]
        vectorstore.add_documents(documents=documents, ids=ids)
        logger.info(
            f"Inserted {len(documents)} documents into '{self.collection_name}'.")

    def as_langchain_vectorstore(self, embedding: Embeddings) -> QdrantVectorStore:
        """Return a LangChain-compatible vectorstore."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embedding,
            vector_name=self.vector_name,
        )
