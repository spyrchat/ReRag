from core.metadata import PageMetadata
from text_pipeline.chunker import TextChunker
from text_pipeline.embedder import Embedder
from text_pipeline.qdrant_uploader import QdrantUploader


class TextRouter:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.uploader = QdrantUploader()

    def route(self, text: str, metadata: PageMetadata):
        chunks = self.chunker.chunk(text, metadata)
        dense_vectors = self.embedder.embed_dense([c.page_content for c in chunks])
        sparse_vectors = self.embedder.embed_sparse([c.page_content for c in chunks])
        self.uploader.upload(
            docs=chunks,
            dense_embedder=self.embedder.dense_model,
            sparse_embedder=self.embedder.sparse_model,
        )
