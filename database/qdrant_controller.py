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

    def insert_embeddings(self, documents, vectors, vector_name="dense"):
        if len(documents) != len(vectors):
            raise ValueError("Mismatched number of documents and vectors")

        points = []

        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            metadata = doc.metadata.copy()
            doc_id = metadata.get("doc_id")
            chunk_id = metadata.get("chunk_id", i)

            if not doc_id:
                raise ValueError(f"Missing doc_id for document {i}")

            metadata.update({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": doc.page_content
            })

            point_id = str(uuid4())  # Qdrant requires UUID or int
            point = PointStruct(
                id=point_id,
                payload=metadata,
                # allows hybrid if needed later
                vector={vector_name: vector}
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(
            f"Inserted {len(points)} points into Qdrant ({self.collection_name})")
