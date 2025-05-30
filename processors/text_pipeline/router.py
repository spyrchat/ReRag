import os
from chunker import TextChunker
from embedder import Embedder
from uploader import QdrantUploader
from utils import prepare_documents
from processors.core.metadata import PageMetadata
from langchain.schema import Document


class TextRouter:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.uploader = QdrantUploader()
        self.strategy = os.getenv("EMBEDDING_STRATEGY", "dense").lower()

    def route(self, text: str, metadata: PageMetadata):
        doc = Document(page_content=text, metadata={
            "doc_id": metadata.doc_id,
            "page": metadata.page,
            "source": f"{metadata.doc_id}.pdf"
        })

        chunks = self.chunker.chunk([doc])
        enriched = prepare_documents([c.page_content for c in chunks], [doc])

        dense_model = self.embedder.dense_model if self.strategy in ("dense", "hybrid") else None
        sparse_model = self.embedder.sparse_model if self.strategy in ("sparse", "hybrid") else None

        self.uploader.upload(
            docs=enriched,
            dense_embedder=dense_model,
            sparse_embedder=sparse_model,
        )
