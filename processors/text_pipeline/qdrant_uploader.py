from typing import List, Optional
from langchain_core.documents import Document
from database.qdrant_controller import QdrantVectorDB
from langchain_core.embeddings import Embeddings


class QdrantUploader:
    def __init__(self):
        self.db = QdrantVectorDB()

    def upload(
        self,
        docs: List[Document],
        dense_embedder: Optional[Embeddings] = None,
        sparse_embedder: Optional[Embeddings] = None,
    ):
        self.db.insert_documents(
            documents=docs,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
