from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from processors.core.base import BaseProcessor


class PDFProcessor(BaseProcessor):
    def load_single(self, path: str) -> List[Document]:
        loader = PyPDFLoader(path)
        return loader.load()
