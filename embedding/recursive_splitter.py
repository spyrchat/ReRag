from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .splitter import BaseSplitter
from typing import List


class RecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
