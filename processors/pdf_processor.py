from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from processors.base import BaseProcessor


class PDFProcessor(BaseProcessor):
    def load(self) -> List[Document]:
        """Load PDF using LangChain's PyPDFLoader."""
        loader = PyPDFLoader(self.path)
        return loader.load()
