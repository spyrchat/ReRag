from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseProcessor(ABC):
    def __init__(self, path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @abstractmethod
    def load(self) -> List[Document]:
        """Use a LangChain-compatible loader to return a list of Documents."""
        pass

    def chunk(self, docs: List[Document]) -> List[Document]:
        """Chunk documents using LangChain's RecursiveCharacterTextSplitter."""
        return self.splitter.split_documents(docs)

    def process(self) -> List[Document]:
        """Load and chunk the documents."""
        return self.chunk(self.load())
