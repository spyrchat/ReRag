from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseProcessor(ABC):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.paths: List[str] = []

    def add_file(self, path: str):
        self.paths.append(path)

    @abstractmethod
    def load_single(self, path: str) -> List[Document]:
        """Load one file (PDF, CSV, etc)."""
        pass

    def load(self) -> List[Document]:
        """Load all files added so far."""
        docs = []
        for path in self.paths:
            docs.extend(self.load_single(path))
        return docs

    def chunk(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

    def process(self) -> List[Document]:
        return self.chunk(self.load())
