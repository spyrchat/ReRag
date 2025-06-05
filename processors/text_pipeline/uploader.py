from database.qdrant_controller import QdrantVectorDB
from typing import List
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class QdrantUploader:
    def __init__(self):
        self.db = QdrantVectorDB()

    def upload(
        self,
        docs: List[Document],
        dense_embedder: Embeddings,
        sparse_embedder: Embeddings,
    ):
        if not docs:
            logger.warning("No documents provided for upload")
            return

        try:
            self.db.insert_documents(
                documents=docs,
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
            )
        except Exception as e:
            logger.error(f"Failed to upload documents: {str(e)}")
            raise