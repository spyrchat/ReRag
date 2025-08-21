from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


class TextChunker:
    def __init__(self, chunk_size=300, chunk_overlap=30):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        return chunks
