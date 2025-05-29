from database.qdrant_controller import QdrantVectorDB
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class QdrantUploader:
    def __init__(self):
        self.db = QdrantVectorDB()

    def upload(
        self,
        docs: List[Document],
        dense_embedder: Embeddings,
        sparse_embedder: Embeddings,
    ):
        self.db.insert_documents(
            documents=docs,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
