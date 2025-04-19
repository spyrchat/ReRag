from langchain.schema import Document
from embedding.base_embedder import BaseEmbedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Tuple, List


class EmbeddingPipeline:
    def __init__(self, embedder: BaseEmbedder, splitter: RecursiveCharacterTextSplitter):
        self.embedder = embedder
        self.splitter = splitter

    def run(self, docs: List[Document]) -> Tuple[List[str], List[List[float]]]:
        chunks = self.splitter.split(docs)
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder.embed_texts(texts)
        return texts, embeddings
